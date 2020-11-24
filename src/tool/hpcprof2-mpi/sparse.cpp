// -*-Mode: C++;-*-

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2020, Rice University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *

#include "sparse.hpp"

#include <lib/profile/util/log.hpp>
#include <lib/profile/mpi/all.hpp>

#include <lib/prof-lean/hpcrun-fmt.h>
#include <lib/prof-lean/id-tuple.h>
#include <lib/prof/pms-format.h>
#include <lib/prof/cms-format.h>


#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>
#include <assert.h>
#include <stdexcept> 

using namespace hpctoolkit;

SparseDB::SparseDB(const stdshim::filesystem::path& p) : dir(p), ctxMaxId(0), outputCnt(0) {
  if(dir.empty())
    util::log::fatal{} << "SparseDB doesn't allow for dry runs!";
  else
    stdshim::filesystem::create_directory(dir);
}

SparseDB::SparseDB(stdshim::filesystem::path&& p) : dir(std::move(p)), ctxMaxId(0), outputCnt(0) {
  if(dir.empty())
    util::log::fatal{} << "SparseDB doesn't allow for dry runs!";
  else
    stdshim::filesystem::create_directory(dir);
}

void SparseDB::notifyWavefront(DataClass d) noexcept {
  if(!d.hasContexts()) return;
  auto sig = contextWavefront.signal();

  std::map<unsigned int, std::reference_wrapper<const Context>> cs;
  src.contexts().citerate([&](const Context& c){
    auto id = c.userdata[src.identifier()];
    ctxMaxId = std::max(ctxMaxId, id);
    if(!cs.emplace(id, c).second)
      util::log::fatal() << "Duplicate Context identifier "
                         << c.userdata[src.identifier()] << "!";
  }, nullptr);

  contexts.reserve(cs.size());
  for(const auto& ic: cs) contexts.emplace_back(ic.second);

  ctxcnt = contexts.size();
}

void SparseDB::notifyThreadFinal(const Thread::Temporary& tt) {
  const auto& t = tt.thread();
  contextWavefront.wait();

  // Allocate the blobs needed for the final output
  std::vector<hpcrun_metricVal_t> values;
  std::vector<uint16_t> mids;
  std::vector<uint32_t> cids;
  std::vector<uint64_t> coffsets;
  coffsets.reserve(contexts.size() + 1);

  // Now stitch together each Context's results
  for(const Context& c: contexts) {
    if(auto accums = tt.accumulatorsFor(c)) {
      cids.push_back(c.userdata[src.identifier()]);
      coffsets.push_back(values.size());
      for(const auto& mx: accums->citerate()) {
        const auto& m = *mx.first;
        const auto& vv = mx.second;
        if(!m.scopes().has(MetricScope::function) || !m.scopes().has(MetricScope::execution))
          util::log::fatal{} << "Metric isn't function/execution!";
        const auto& ids = m.userdata[src.mscopeIdentifiers()];
        hpcrun_metricVal_t v;
        if(auto vex = vv.get(MetricScope::function)) {
          v.r = *vex;
          mids.push_back(ids.function);
          values.push_back(v);
        }
        if(auto vinc = vv.get(MetricScope::execution)) {
          v.r = *vinc;
          mids.push_back(ids.execution);
          values.push_back(v);
        }
      }
    }
  }

  //Add the extra ctx id and offset pair, to mark the end of ctx  - YUMENG
  cids.push_back(LastNodeEnd);
  coffsets.push_back(values.size());

  // Put together the sparse_metrics structure
  hpcrun_fmt_sparse_metrics_t sm;
  sm.id_tuple.length = 0; //always 0 here
  sm.num_vals = values.size();
  sm.num_cct_nodes = contexts.size();
  sm.num_nz_cct_nodes = coffsets.size() - 1; //since there is an extra end node 
  sm.values = values.data();
  sm.mids = mids.data();
  sm.cct_node_ids = cids.data();
  sm.cct_node_idxs = coffsets.data();

  // Set up the output temporary file.
  stdshim::filesystem::path outfile;
  int world_rank;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::ostringstream ss;
    ss << "tmp-" << world_rank << "."
       << outputCnt.fetch_add(1, std::memory_order_relaxed) << ".sparse-db";
    outfile = dir / ss.str();
  }
  std::FILE* of = std::fopen(outfile.c_str(), "wb");
  if(!of) util::log::fatal() << "Unable to open temporary sparse-db file for output!";

  // Spit it all out, and close up.
  if(hpcrun_fmt_sparse_metrics_fwrite(&sm, of) != HPCFMT_OK)
    util::log::fatal() << "Error writing out temporary sparse-db!";
  std::fclose(of);

  // Log the output for posterity
  outputs.emplace(&t, std::move(outfile));
}

void SparseDB::write()
{
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if(world_rank != 0) return;

  // Allocate the blobs needed for the final output
  std::vector<hpcrun_metricVal_t> values;
  std::vector<uint16_t> mids;
  std::vector<uint32_t> cids;
  std::vector<uint64_t> coffsets;
  coffsets.reserve(contexts.size() + 1);

  // Now stitch together each Context's results
  for(const Context& c: contexts) {
    const auto& stats = c.statistics();
    if(stats.size() > 0) {
      cids.push_back(c.userdata[src.identifier()]);
      coffsets.push_back(values.size());
    }
    for(const auto& mx: stats.citerate()) {
      const auto& m = *mx.first;
      if(!m.scopes().has(MetricScope::function) || !m.scopes().has(MetricScope::execution))
        util::log::fatal{} << "Metric isn't function/execution!";
      const auto& ids = m.userdata[src.mscopeIdentifiers()];
      const auto& vv = mx.second;
      size_t idx = 0;
      for(const auto& sp: m.partials()) {
        hpcrun_metricVal_t v;
        if(auto vex = vv.get(sp).get(MetricScope::function)) {
          v.r = *vex;
          mids.push_back((ids.function << 8) + idx);
          values.push_back(v);
        }
        if(auto vinc = vv.get(sp).get(MetricScope::execution)) {
          v.r = *vinc;
          mids.push_back((ids.execution << 8) + idx);
          values.push_back(v);
        }
        idx++;
      }
    }
  }

  //Add the extra ctx id and offset pair, to mark the end of ctx
  cids.push_back(LastNodeEnd);
  coffsets.push_back(values.size());

  // Put together the sparse_metrics structure
  hpcrun_fmt_sparse_metrics_t sm;
  //sm.tid = 0;
  sm.id_tuple.length = 0;
  sm.num_vals = values.size();
  sm.num_cct_nodes = contexts.size();
  sm.num_nz_cct_nodes = coffsets.size() - 1; //since there is an extra end node 
  sm.values = values.data();
  sm.mids = mids.data();
  sm.cct_node_ids = cids.data();
  sm.cct_node_idxs = coffsets.data();

  // Set up the output temporary file.
  summaryOut = dir / "tmp-summary.sparse-db";
  std::FILE* of = std::fopen(summaryOut.c_str(), "wb");
  if(!of) util::log::fatal() << "Unable to open temporary summary sparse-db file for output!";

  // Spit it all out, and close up.
  if(hpcrun_fmt_sparse_metrics_fwrite(&sm, of) != HPCFMT_OK)
    util::log::fatal() << "Error writing out temporary summary sparse-db!";
  std::fclose(of);
}

//***************************************************************************
// Work with bytes
//***************************************************************************
int SparseDB::writeAsByte4(uint32_t val, util::File::Instance& fh, MPI_Offset off){
  int shift = 0, num_writes = 0;
  char input[4];

  for (shift = 24; shift >= 0; shift -= 8) {
    input[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }

  fh.writeat(off, 4, input);
  return SPARSE_OK;
}

int SparseDB::writeAsByte8(uint64_t val, util::File::Instance& fh, MPI_Offset off){
  int shift = 0, num_writes = 0;
  char input[8];

  for (shift = 56; shift >= 0; shift -= 8) {
    input[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }

  fh.writeat(off, 8, input);
  return SPARSE_OK;
}

int SparseDB::writeAsByteX(std::vector<char>& val, size_t size, util::File::Instance& fh, MPI_Offset off){
  fh.writeat(off, size, val.data());
  return SPARSE_OK;
}

int SparseDB::readAsByte4(uint32_t& val, util::File::Instance& fh, MPI_Offset off){
  uint32_t v = 0;
  int shift = 0, num_reads = 0;
  char input[4];

  fh.readat(off, 4, input);
  
  for (shift = 24; shift >= 0; shift -= 8) {
    v |= ((uint32_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  val = v;
  return SPARSE_OK;

}

int SparseDB::readAsByte8(uint64_t& val, util::File::Instance& fh, MPI_Offset off){
  uint32_t v = 0;
  int shift = 0, num_reads = 0;
  char input[8];

  fh.readat(off, 8, input);
  
  for (shift = 56; shift >= 0; shift -= 8) {
    v |= ((uint64_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  val = v;
  return SPARSE_OK;

}

void SparseDB::interpretByte2(uint16_t& val, const char *input){
  uint16_t v = 0;
  int shift = 0, num_reads = 0;

  for (shift = 8; shift >= 0; shift -= 8) {
    v |= ((uint16_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  val = v;
}

void SparseDB::interpretByte4(uint32_t& val, const char *input){
  uint32_t v = 0;
  int shift = 0, num_reads = 0;

  for (shift = 24; shift >= 0; shift -= 8) {
    v |= ((uint32_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  val = v;
}

void SparseDB::interpretByte8(uint64_t& val, const char *input){
  uint64_t v = 0;
  int shift = 0, num_reads = 0;

  for (shift = 56; shift >= 0; shift -= 8) {
    v |= ((uint64_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  val = v;
}

void SparseDB::convertToByte2(uint16_t val, char* bytes){
  int shift = 0, num_writes = 0;

  for (shift = 8; shift >= 0; shift -= 8) {
    bytes[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }
}

void SparseDB::convertToByte4(uint32_t val, char* bytes){
  int shift = 0, num_writes = 0;

  for (shift = 24; shift >= 0; shift -= 8) {
    bytes[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }
}

void SparseDB::convertToByte8(uint64_t val, char* bytes){
  int shift = 0, num_writes = 0;

  for (shift = 56; shift >= 0; shift -= 8) {
    bytes[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }
}



//***************************************************************************
// profile.db  - YUMENG
//
///EXAMPLE
///HPCPROF-tmsdb_____
///[hdr:
///  (version: 0)
///]
///[Id tuples for 121 profiles
///  0[(SUMMARY: 0) ]
///  1[(RANK: 0) (THREAD: 0) ]
///  2[(RANK: 0) (THREAD: 1) ]
///  ...
///]
///[Profile informations for 72 profiles
///  0[(id_tuple_ptr: 23) (metadata_ptr: 0) (spare_one: 0) (spare_two: 0) (num_vals: 12984) (num_nzctxs: 8353) (starting location: 489559)]
///  1[(id_tuple_ptr: 31) (metadata_ptr: 0) (spare_one: 0) (spare_two: 0) (num_vals: 4422) (num_nzctxs: 3117) (starting location: 886225)]
///  ...
///]
///[thread 36
///  [metrics:
///  (NOTES: printed in file order, help checking if the file is correct)
///    (value: 2.8167, metric id: 1)
///    (value: 2.8167, metric id: 1)
///    ...
///  ]
///  [ctx indices:
///    (ctx id: 1, index: 0)
///    (ctx id: 7, index: 1)
///    ...
///  ]
///]
///...
//***************************************************************************
void SparseDB::assignSparseInputs(int world_rank)
{
  for(const auto& tp : outputs.citerate()){
    if(tp.first->attributes.idTuple().size() == 0) continue; //skip this profile
    //regular prof_info_idx starts with 1, 0 is for summary
    sparseInputs.emplace_back(tp.first->userdata[src.identifier()] + 1, tp.second.string());
  }

  if(world_rank == 0)
    sparseInputs.emplace_back(IDTUPLE_SUMMARY_PROF_INFO_IDX, summaryOut.string());
  
}

uint32_t SparseDB::getTotalNumProfiles(const uint32_t my_num_prof)
{
  uint32_t total_num_prof;
  MPI_Allreduce(&my_num_prof, &total_num_prof, 1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);
  return total_num_prof;
}

//---------------------------------------------------------------------------
// header
//---------------------------------------------------------------------------
void SparseDB::writePMSHdr(const uint32_t total_num_prof, const util::File& fh)
{
  if(mpi::World::rank() != 0) return;

  std::vector<char> hdr;
  
  hdr.insert(hdr.end(), HPCPROFILESPARSE_FMT_Magic, HPCPROFILESPARSE_FMT_Magic + HPCPROFILESPARSE_FMT_MagicLen);
  uint64_t cur_off = HPCPROFILESPARSE_FMT_MagicLen;

  hdr.emplace_back(HPCPROFILESPARSE_FMT_VersionMajor);
  hdr.emplace_back(HPCPROFILESPARSE_FMT_VersionMinor);
  cur_off += HPCPROFILESPARSE_FMT_VersionLen;

  hdr.resize(PMS_hdr_SIZE);
  convertToByte4(total_num_prof, hdr.data() + cur_off);
  cur_off += HPCPROFILESPARSE_FMT_NumProfLen;

  convertToByte2(HPCPROFILESPARSE_FMT_NumSec, hdr.data() + cur_off); 
  cur_off += HPCPROFILESPARSE_FMT_NumSecLen;

  convertToByte8(prof_info_sec_size, hdr.data() + cur_off);
  convertToByte8(prof_info_sec_ptr, hdr.data() + cur_off + HPCPROFILESPARSE_FMT_SecSizeLen);
  cur_off += HPCPROFILESPARSE_FMT_SecLen;

  convertToByte8(id_tuples_sec_size, hdr.data() + cur_off);
  convertToByte8(id_tuples_sec_ptr, hdr.data() + cur_off + HPCPROFILESPARSE_FMT_SecSizeLen);
  cur_off += HPCPROFILESPARSE_FMT_SecLen;
  assert(cur_off == prof_info_sec_ptr);
  
  auto fhi = fh.open(true);
  SPARSE_exitIfMPIError(writeAsByteX(hdr, PMS_hdr_SIZE, fhi, 0),
    __FUNCTION__ + std::string(": write the hdr wrong"));
  
}

//---------------------------------------------------------------------------
// profile offsets
//---------------------------------------------------------------------------
uint64_t SparseDB::getProfileSizes()
{
  uint64_t my_size = 0;
  for(const auto& tp: sparseInputs){
    struct stat buf;
    stat(tp.second.c_str(),&buf);
    my_size += (buf.st_size - PMS_prof_skip_SIZE);
    profile_sizes.emplace_back(buf.st_size - PMS_prof_skip_SIZE);    
  }
  return my_size;
}


uint64_t SparseDB::getMyOffset(const uint64_t my_size, const int rank)
{
  uint64_t my_offset;
  MPI_Exscan(&my_size, &my_offset, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0) my_offset = 0;
  return my_offset;
}


void SparseDB::getMyProfOffset(const uint32_t total_prof, const uint64_t my_offset, 
                               const int threads)
{
  prof_offsets.resize(profile_sizes.size());

  std::vector<uint64_t> tmp (profile_sizes.size());
  #pragma omp parallel for num_threads(threads)
  for(uint i = 0; i < tmp.size();i++)
    tmp[i] = profile_sizes[i];
  

  exscan<uint64_t>(tmp,threads);

  #pragma omp parallel for num_threads(threads) 
  for(uint i = 0; i < tmp.size();i++){
    if(i < tmp.size() - 1) assert(tmp[i] + profile_sizes[i] == tmp[i+1]);
    prof_offsets[i] = tmp[i] + my_offset + PMS_hdr_SIZE
      + (MULTIPLE_8(prof_info_sec_size)) + (MULTIPLE_8(id_tuples_sec_size)); 
  }

}


void SparseDB::workProfSizesOffsets(const int world_rank, const int total_prof, 
                                    const int threads)
{
  //work on class private variable profile_sizes
  uint64_t my_size = getProfileSizes();
  uint64_t my_off = getMyOffset(my_size, world_rank);
  //work on class private variable prof_offsets
  getMyProfOffset(total_prof, my_off, threads);
}


//---------------------------------------------------------------------------
// profile id tuples 
//---------------------------------------------------------------------------

std::vector<std::pair<uint16_t, uint64_t>>  SparseDB::getMyIdTuplesPairs()
{
  std::vector<std::pair<uint16_t, uint64_t>> pairs;
  
  uint64_t rank = mpi::World::rank();

  //each rank's pairs are led by a pair: {RANK_SPOT, rank number}
  pairs.emplace_back(RANK_SPOT, rank);

  //go through my rank's profiles' tuples and save them as pairs
  for(const auto& tp : outputs.citerate()){
    auto& ta = tp.first->attributes;
    uint16_t tuple_length = ta.idTuple().size();
    uint64_t prof_info_idx = tp.first->userdata[src.identifier()] + 1;
    if(tuple_length == 0) continue; //skip this profile

    pairs.emplace_back(tuple_length, prof_info_idx);
    for(auto id : ta.idTuple())
      pairs.emplace_back(id.kind, id.index);
  }

  if(rank != 0) return pairs;

  //don't forget summary file for rank 0
  pairs.emplace_back(IDTUPLE_SUMMARY_LENGTH, IDTUPLE_SUMMARY_PROF_INFO_IDX); 
  pairs.emplace_back(IDTUPLE_SUMMARY, IDTUPLE_SUMMARY_IDX); 
  return pairs;
}


std::vector<pms_id_tuple_t>
SparseDB::intPairs2Tuples(const std::vector<std::pair<uint16_t, uint64_t>>& all_pairs)
{
  std::vector<pms_id_tuple_t> tuples;
  uint i = 0;   //idx in all_pairs
  uint idx = 0; //idx in tuples
  uint cur_rank = -1; 
  while(i < all_pairs.size()){
    std::pair<uint16_t, uint64_t> p = all_pairs[i];

    //check if the following pairs belong to a new rank 
    if(p.first == RANK_SPOT){
      assert(p.second != cur_rank);
      cur_rank = p.second;
      i++;
      continue;
    }

    //create a new tuple
    id_tuple_t it;
    it.length = p.first;
    it.ids = (pms_id_t*)malloc(it.length * sizeof(pms_id_t));
    for(uint j = 0; j < it.length; j++){
      it.ids[j].kind = all_pairs[i+j+1].first;
      it.ids[j].index = all_pairs[i+j+1].second;
    }

    pms_id_tuple_t t;
    t.rank = cur_rank;
    t.all_at_root_idx = idx;
    t.prof_info_idx = p.second;
    t.idtuple = it;
    
    //update tuples, next idx in tuples, and next idx in all_pairs
    tuples.emplace_back(t);
    idx += 1;
    i += (1 + t.idtuple.length);
  }

  return tuples;
}



std::vector<std::pair<uint16_t, uint64_t>> SparseDB::gatherIdTuplesPairs(const int world_rank, const int world_size,
                                                                         const int threads, MPI_Datatype IntPairType, 
                                                                         const std::vector<std::pair<uint16_t, uint64_t>>& rank_pairs)
{
  //get the size of each ranks' pairs
  int rank_pairs_size = rank_pairs.size();
  std::vector<int> all_rank_pairs_sizes;
  if(world_rank == 0) all_rank_pairs_sizes.resize(world_size);
  MPI_Gather(&rank_pairs_size, 1, MPI_INT, all_rank_pairs_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  //get the displacement of all ranks' pairs
  std::vector<int> all_rank_pairs_disps; 
  std::vector<std::pair<uint16_t, uint64_t>> all_rank_pairs;
  if(world_rank == 0){
    all_rank_pairs_disps.resize(world_size);

    #pragma omp parallel for num_threads(threads)
    for(int i = 0; i<world_size; i++) all_rank_pairs_disps[i] = all_rank_pairs_sizes[i];
    exscan<int>(all_rank_pairs_disps,threads); 

    int total_size = all_rank_pairs_disps.back() + all_rank_pairs_sizes.back();
    all_rank_pairs.resize(total_size);
  }

  MPI_Gatherv(rank_pairs.data(),rank_pairs_size, IntPairType, \
    all_rank_pairs.data(), all_rank_pairs_sizes.data(), all_rank_pairs_disps.data(), IntPairType, 0, MPI_COMM_WORLD);

  return all_rank_pairs;

}


void SparseDB::scatterIdxPtrs(const std::vector<std::pair<uint32_t, uint64_t>>& idx_ptr_buffer, 
                              const size_t num_prof,const int world_size, const int world_rank,
                              const int threads)
{
  rank_idx_ptr_pairs.resize(num_prof);

  //get the size of each ranks' tuples
  std::vector<int> all_rank_tuples_sizes;
  if(world_rank == 0) all_rank_tuples_sizes.resize(world_size);
  MPI_Gather(&num_prof, 1, MPI_INT, all_rank_tuples_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  //get the displacement of all ranks' tuples
  std::vector<int> all_rank_tuples_disps; 
  if(world_rank == 0){
    all_rank_tuples_disps.resize(world_size);

    #pragma omp parallel for num_threads(threads)
    for(int i = 0; i<world_size; i++) all_rank_tuples_disps[i] = all_rank_tuples_sizes[i];
    exscan<int>(all_rank_tuples_disps,threads); 
  }

  //create a new Datatype for prof_info_idx and offset
  MPI_Datatype IdxPtrType = createPairType<uint32_t, uint64_t>(MPI_UINT32_T, MPI_UINT64_T);

  MPI_Scatterv(idx_ptr_buffer.data(), all_rank_tuples_sizes.data(), all_rank_tuples_disps.data(), \
    IdxPtrType, rank_idx_ptr_pairs.data(), num_prof, IdxPtrType, 0, MPI_COMM_WORLD);

}


void SparseDB::sortIdTuples(std::vector<pms_id_tuple_t>& all_tuples)
{
  struct {
    bool operator()(pms_id_tuple_t a, 
                    pms_id_tuple_t b) const
    {   
      id_tuple_t& a_tuple = a.idtuple;
      id_tuple_t& b_tuple = b.idtuple;
      uint16_t len = std::min(a_tuple.length, b_tuple.length);
      for(uint i = 0; i<len; i++){
        if(a_tuple.ids[i].kind != b_tuple.ids[i].kind){
          return a_tuple.ids[i].kind < b_tuple.ids[i].kind;
        }else{
          if(a_tuple.ids[i].index != b_tuple.ids[i].index){
            return a_tuple.ids[i].index < b_tuple.ids[i].index;
          }
        }
      }
      return a_tuple.length < b_tuple.length;
    }   
  }tupleComp;
  
  std::sort(all_tuples.begin(), all_tuples.end(), tupleComp);
}


void SparseDB::sortIdTuplesOnProfInfoIdx(std::vector<pms_id_tuple_t>& all_tuples)
{
  struct {
    bool operator()(pms_id_tuple_t a, 
                    pms_id_tuple_t b) const
    {   
      return a.prof_info_idx < b.prof_info_idx;
    }   
  }tupleComp;
  
  std::sort(all_tuples.begin(), all_tuples.end(), tupleComp);
}


void SparseDB::fillIdxPtrBuffer(std::vector<pms_id_tuple_t>& all_tuples,
                                std::vector<std::pair<uint32_t, uint64_t>>& buffer,
                                const int threads)
{
  assert(buffer.size() == all_tuples.size());
  std::vector<uint64_t> tupleSizes (all_tuples.size()+1,0);

  //notice the last entry in tupleSizes is still 0
  #pragma omp parallel for num_threads(threads)
  for(uint i = 0; i < all_tuples.size(); i++)
    tupleSizes[i] = PMS_id_tuple_len_SIZE + all_tuples[i].idtuple.length * PMS_id_SIZE;
  

  exscan<uint64_t>(tupleSizes,threads);

  #pragma omp parallel for num_threads(threads) 
  for(uint i = 0; i < all_tuples.size();i++){
    auto& t = all_tuples[i];
    buffer[t.all_at_root_idx] = {t.prof_info_idx, (tupleSizes[i] + id_tuples_sec_ptr)};
  }

}


void SparseDB::freeIdTuples(std::vector<pms_id_tuple_t>& all_tuples, const int threads)
{
  #pragma omp parallel for num_threads(threads)
  for(uint i = 0; i < all_tuples.size(); i++){
    free(all_tuples[i].idtuple.ids);
    all_tuples[i].idtuple.ids = NULL;
  }
}

 
std::vector<char> SparseDB::convertTuple2Bytes(const pms_id_tuple_t& tuple)
{
  uint16_t len = tuple.idtuple.length;
  std::vector<char> bytes(PMS_id_tuple_len_SIZE + len * PMS_id_SIZE);
  convertToByte2(len, bytes.data());
  char* byte_pos = bytes.data() + 2;
  for(uint i = 0; i < len; i++){
    auto& id = tuple.idtuple.ids[i];
    convertToByte2(id.kind, byte_pos);
    convertToByte8(id.index, byte_pos+2);
    byte_pos += PMS_id_SIZE;
  }
  return bytes;
}


void SparseDB::writeAllIdTuples(const std::vector<pms_id_tuple_t>& all_tuples, const util::File& fh)
{
  std::vector<char> bytes;
  for(auto& tuple : all_tuples)
  {
    std::vector<char> b = convertTuple2Bytes(tuple);
    bytes.insert(bytes.end(), b.begin(), b.end());
  }

  auto fhi = fh.open(true);
  SPARSE_exitIfMPIError(writeAsByteX(bytes, bytes.size(), fhi, id_tuples_sec_ptr),"error writing all tuples");
  
  //set class private variable 
  id_tuples_sec_size = bytes.size();
}


void SparseDB::workIdTuplesSection(const int world_rank, const int world_size, const int threads,
                                   const int num_prof, const util::File& fh)
{
  MPI_Datatype IntPairType = createPairType<uint16_t, uint64_t>(MPI_UINT16_T, MPI_UINT64_T);

  //will be used as a send buffer for {prof_info_idx, id_tuple_ptr}s
  std::vector<std::pair<uint32_t, uint64_t>> idx_ptr_buffer;

  //each rank collect its own pairs
  auto pairs = getMyIdTuplesPairs();

  //rank 0 gather all the pairs
  auto all_rank_pairs = gatherIdTuplesPairs(world_rank, world_size, threads, IntPairType, pairs);

  //rank 0 convert pairs to tuples and sort them
  if(world_rank == 0) {
    auto all_rank_tuples = intPairs2Tuples(all_rank_pairs);

    sortIdTuplesOnProfInfoIdx(all_rank_tuples);

    //write all the tuples
    writeAllIdTuples(all_rank_tuples, fh);
  
    //calculate the tuples' offset as ptrs
    idx_ptr_buffer.resize(all_rank_tuples.size());
    fillIdxPtrBuffer(all_rank_tuples, idx_ptr_buffer, threads);

    freeIdTuples(all_rank_tuples, threads);
  }

  //rank 0 sends back necessary info
  MPI_Bcast(&id_tuples_sec_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  scatterIdxPtrs(idx_ptr_buffer, num_prof, world_size, world_rank, threads);

}


//---------------------------------------------------------------------------
// get profile's real data (bytes)
//---------------------------------------------------------------------------
void SparseDB::updateCtxMids(const char* input, const uint64_t ctx_nzval_cnt,
                             std::set<uint16_t>& ctx_nzmids)
{
  for(uint m = 0; m < ctx_nzval_cnt; m++){
    uint16_t mid;
    interpretByte2(mid, input);
    ctx_nzmids.insert(mid); 
    input += (PMS_mid_SIZE + PMS_val_SIZE);
  }
}


void SparseDB::interpretOneCtxValCntMids(const char* val_cnt_input, const char* mids_input,
                                         std::vector<std::set<uint16_t>>& ctx_nzmids,
                                         std::vector<uint64_t>& ctx_nzval_cnts)
{
  uint32_t ctx_id;
  interpretByte4(ctx_id, val_cnt_input);
  
  // nzval_cnt
  uint64_t ctx_idx; 
  uint64_t next_ctx_idx;
  interpretByte8(ctx_idx, val_cnt_input + PMS_ctx_id_SIZE);
  interpretByte8(next_ctx_idx, val_cnt_input + PMS_ctx_id_SIZE + PMS_ctx_pair_SIZE);
  uint64_t ctx_nzval_cnt = next_ctx_idx - ctx_idx;
  ctx_nzval_cnts[CTX_VEC_IDX(ctx_id)] += ctx_nzval_cnt;

  // nz-mids
  mids_input += ctx_idx * (PMS_mid_SIZE + PMS_val_SIZE) + PMS_val_SIZE;
  updateCtxMids(mids_input, ctx_nzval_cnt, ctx_nzmids[CTX_VEC_IDX(ctx_id)]);

}


void SparseDB::collectCctMajorData(const uint32_t prof_info_idx, std::vector<char>& bytes,
                                   std::vector<uint64_t>& ctx_nzval_cnts, 
                                   std::vector<std::set<uint16_t>>& ctx_nzmids)
{ 
  assert(ctx_nzval_cnts.size() == ctx_nzmids.size());
  assert(ctx_nzval_cnts.size() > 0);

  uint64_t num_vals;
  uint32_t num_nzctxs;
  interpretByte8(num_vals,   bytes.data() + PMS_fake_id_tuple_SIZE);
  interpretByte4(num_nzctxs, bytes.data() + PMS_fake_id_tuple_SIZE + PMS_num_val_SIZE);
  uint64_t ctx_end_idx;
  interpretByte8(ctx_end_idx, bytes.data() + bytes.size() - PMS_ctx_idx_SIZE);
  if(num_vals != ctx_end_idx) 
    exitError("tmpDB file for thread " + std::to_string(prof_info_idx) + " is corrupted!");
  

  char* val_cnt_input = bytes.data() + PMS_prof_skip_SIZE + num_vals * (PMS_val_SIZE + PMS_mid_SIZE);
  char* mids_input = bytes.data() + PMS_prof_skip_SIZE;
  for(uint i = 0; i < num_nzctxs; i++){
    interpretOneCtxValCntMids(val_cnt_input, mids_input, ctx_nzmids, ctx_nzval_cnts);
    val_cnt_input += PMS_ctx_pair_SIZE;  
  }

}

//---------------------------------------------------------------------------
// write profiles 
//---------------------------------------------------------------------------
std::vector<char> SparseDB::buildOneProfInfoBytes(const std::vector<char>& partial_info_bytes, 
                                                  const uint64_t id_tuple_ptr, const uint64_t metadata_ptr,
                                                  const uint64_t spare_one_ptr, const uint64_t spare_two_ptr,
                                                  const uint64_t prof_offset)
{
  std::vector<char> info_bytes(PMS_prof_info_SIZE); 
  convertToByte8(id_tuple_ptr, info_bytes.data());
  convertToByte8(metadata_ptr, info_bytes.data() + PMS_id_tuple_ptr_SIZE);
  convertToByte8(spare_one_ptr, info_bytes.data() + PMS_id_tuple_ptr_SIZE + PMS_metadata_ptr_SIZE);
  convertToByte8(spare_two_ptr, info_bytes.data() + PMS_id_tuple_ptr_SIZE + PMS_metadata_ptr_SIZE + PMS_spare_one_SIZE);
  std::copy(partial_info_bytes.begin(), partial_info_bytes.end(),info_bytes.begin()+PMS_ptrs_SIZE);
  convertToByte8(prof_offset, info_bytes.data() + PMS_prof_info_SIZE - PMS_prof_offset_SIZE);
  return info_bytes;
}


void SparseDB::writeOneProfile(const std::pair<uint32_t, std::string>& tupleFn,
                               const MPI_Offset my_prof_offset, 
                               const std::pair<uint32_t,uint64_t>& prof_idx_off_pair,
                               std::vector<uint64_t>& ctx_nzval_cnts,
                               std::vector<std::set<uint16_t>>& ctx_nzmids,
                               util::File::Instance& fh)
{
  //get file name
  const std::string fn = tupleFn.second;

  //get all bytes from a profile
  std::ifstream input(fn.c_str(), std::ios::binary);
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(input)),
      (std::istreambuf_iterator<char>()));
  input.close();

  if(!keepTemps)
    stdshim::filesystem::remove(fn);

  //collect context local nonzero value counts and nz_mids from this profile
  if(tupleFn.first != IDTUPLE_SUMMARY_PROF_INFO_IDX)
    collectCctMajorData(prof_idx_off_pair.first, bytes, ctx_nzval_cnts, ctx_nzmids);
   
  //write profile info
  std::vector<char> partial_info (PMS_num_val_SIZE + PMS_num_nzctx_SIZE);
  std::copy(bytes.begin() + PMS_fake_id_tuple_SIZE, bytes.begin() + PMS_prof_skip_SIZE, partial_info.begin());
  // metadata_ptr, sparse_one/two are empty now, so 0,0,0
  std::vector<char> info = buildOneProfInfoBytes(partial_info, prof_idx_off_pair.second, 0, 0, 0, my_prof_offset);
  MPI_Offset info_off = PMS_hdr_SIZE + prof_idx_off_pair.first * PMS_prof_info_SIZE;
  fh.writeat(info_off, PMS_prof_info_SIZE, info.data());

  //write profile data
  fh.writeat(my_prof_offset, bytes.size() - PMS_prof_skip_SIZE, bytes.data() + PMS_prof_skip_SIZE);
}

// write all the profiles at the correct place, and collect data needed for cct.db 
// input: calculated prof_offsets, calculated profile_sizes, file handle, number of available threads
void SparseDB::writeProfiles(const util::File& fh,const int threads,
                             std::vector<uint64_t>& ctx_nzval_cnts,
                             std::vector<std::set<uint16_t>>& ctx_nzmids)
{

  assert(ctx_nzval_cnts.size() == ctx_nzmids.size());
  assert(ctx_nzval_cnts.size() > 0);
  assert(prof_offsets.size() == profile_sizes.size());

  std::vector<std::vector<std::set<uint16_t>> *> threads_ctx_nzmids(threads);

  #pragma omp declare reduction (vectorSum : std::vector<uint64_t> : \
      std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<uint64_t>()))\
      initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

  #pragma omp parallel num_threads(threads)
  {
    auto fhi = fh.open(true);

    std::set<uint16_t> empty;
    std::vector<std::set<uint16_t>> thread_ctx_nzmids (ctx_nzmids.size(), empty);
    threads_ctx_nzmids[omp_get_thread_num()] = &thread_ctx_nzmids;

    #pragma omp for reduction(vectorSum : ctx_nzval_cnts)
    for(uint i = 0; i<prof_offsets.size();i++){
      const std::pair<uint32_t, std::string>& tupleFn = sparseInputs[i];
      MPI_Offset my_prof_offset = prof_offsets[i];
      writeOneProfile(tupleFn, my_prof_offset, rank_idx_ptr_pairs[i], ctx_nzval_cnts, thread_ctx_nzmids, fhi);
    }

    // union non-zero metric ids collected from different threads
    #pragma omp for
    for(uint j = 0; j<ctx_nzmids.size(); j++){
      for(int t = 0; t < threads; t++){
        std::set_union(ctx_nzmids[j].begin(), ctx_nzmids[j].end(),
              (*threads_ctx_nzmids[t])[j].begin(), (*threads_ctx_nzmids[t])[j].end(),
              std::inserter(ctx_nzmids[j], ctx_nzmids[j].begin()));
      }
    }

  }//END of parallel region

}


void SparseDB::writeProfileMajor(const int threads, const int world_rank, 
                                 const int world_size, std::vector<uint64_t>& ctx_nzval_cnts,
                                 std::vector<std::set<uint16_t>>& ctx_nzmids)
{
  //
  // some private variables:
  // profile_sizes: vector of profile's own size
  // prof_offsets:  vector of final global offset
  // rank_idx_ptr_pairs: vector of (profile's idx at prof_info section : the ptr to its id_tuple)  
  //

  //set basic info
  util::File profile_major_f(dir / "profile.db", true);

  assignSparseInputs(world_rank);
  int my_num_prof = sparseInputs.size();
  uint32_t total_num_prof = getTotalNumProfiles(my_num_prof);

  //set hdr info, id_tuples_sec_size will be set in wordIdTuplesSection
  prof_info_sec_ptr = PMS_hdr_SIZE;
  prof_info_sec_size = total_num_prof * PMS_prof_info_SIZE;
  id_tuples_sec_ptr = prof_info_sec_ptr + (MULTIPLE_8(prof_info_sec_size));

  //write id_tuples, set id_tuples_sec_size 
  workIdTuplesSection(world_rank, world_size, threads, my_num_prof, profile_major_f);

  //write hdr
  writePMSHdr(total_num_prof,profile_major_f);
    
  //write rest profiles and corresponding prof_info
  workProfSizesOffsets(world_rank, total_num_prof, threads);
  writeProfiles(profile_major_f, threads, ctx_nzval_cnts, ctx_nzmids);
}

//***************************************************************************
// cct.db 
//
///EXAMPLE
///[Context informations for 220 Contexts
///  [(context id: 1) (num_vals: 72) (num_nzmids: 1) (starting location: 4844)]
///  [(context id: 3) (num_vals: 0) (num_nzmids: 0) (starting location: 5728)]
///  ...
///]
///[context 1
///  [metrics easy grep version:
///  (NOTES: printed in file order, help checking if the file is correct)
///    (value: 2.64331, thread id: 0)
///    (value: 2.62104, thread id: 1)
///    ...
///  ]
///  [metric indices:
///    (metric id: 1, index: 0)
///    (metric id: END, index: 72)
///  ]
///]
///...same [sparse metrics] for all rest ctxs 
//***************************************************************************

//---------------------------------------------------------------------------
// header
//---------------------------------------------------------------------------
void SparseDB::writeCMSHdr(util::File::Instance& cct_major_fi)
{
  std::vector<char> hdr;
  hdr.insert(hdr.end(), HPCCCTSPARSE_FMT_Magic, HPCCCTSPARSE_FMT_Magic + HPCCCTSPARSE_FMT_MagicLen);
  hdr.emplace_back(HPCCCTSPARSE_FMT_VersionMajor);
  hdr.emplace_back(HPCCCTSPARSE_FMT_VersionMinor);
  uint64_t cur_off = HPCCCTSPARSE_FMT_MagicLen + HPCCCTSPARSE_FMT_VersionLen;
  
  hdr.resize(CMS_hdr_SIZE);
  convertToByte4(ctxcnt, hdr.data() + cur_off); 
  cur_off += HPCCCTSPARSE_FMT_NumCtxLen;

  convertToByte2(HPCCCTSPARSE_FMT_NumSec, hdr.data() + cur_off); 
  cur_off += HPCCCTSPARSE_FMT_NumSecLen;

  uint64_t ctx_info_sec_ptr = cur_off + (HPCCCTSPARSE_FMT_NumSec-2) * HPCCCTSPARSE_FMT_SecLen;
  uint64_t ctx_info_sec_size = ctxcnt * CMS_ctx_info_SIZE;
  convertToByte8(ctx_info_sec_size, hdr.data() + cur_off);
  convertToByte8(ctx_info_sec_ptr, hdr.data() + cur_off + HPCCCTSPARSE_FMT_SecSizeLen);
  cur_off += HPCCCTSPARSE_FMT_SecLen;
  assert(cur_off == ctx_info_sec_ptr);

  SPARSE_exitIfMPIError(writeAsByteX(hdr, CMS_hdr_SIZE, cct_major_fi, 0),
    __FUNCTION__ + std::string(": write the hdr wrong"));
}


//---------------------------------------------------------------------------
// ctx info
//---------------------------------------------------------------------------
void SparseDB::writeCtxInfoSec(const std::vector<std::set<uint16_t>>& ctx_nzmids,
                               const std::vector<uint64_t>& ctx_off,
                               util::File::Instance& ofh)
{
  assert(ctxcnt == ctx_nzmids.size());

  std::vector<char> info_bytes (CMS_ctx_info_SIZE * ctxcnt);
  for(uint i = 0; i < ctxcnt; i++){
    uint16_t num_nzmids = (uint16_t)(ctx_nzmids[i].size() - 1);
    uint64_t num_vals = (num_nzmids == 0) ? 0 \
      : ((ctx_off[i+1] - ctx_off[i]) - (num_nzmids + 1) * CMS_m_pair_SIZE) / CMS_val_prof_idx_pair_SIZE;
    cms_ctx_info_t ci = {CTXID(i), num_vals, num_nzmids, ctx_off[i]};
    convertOneCtxInfo(ci, info_bytes.data() + i * CMS_ctx_info_SIZE);
  }

  ofh.writeat(CMS_hdr_SIZE, info_bytes.size(), info_bytes.data());

}

//---------------------------------------------------------------------------
// ctx offsets
//---------------------------------------------------------------------------
void SparseDB::unionMids(std::vector<std::set<uint16_t>>& ctx_nzmids, const int rank, 
                         const int num_proc, const int threads)
{
  assert(ctx_nzmids.size() > 0);

  /*
  LOGIC:                                      
  NOTE: ctx_nzmids is a vector of set, each set represents the metric ids for one context
  Step 1: turn ctx_nzmids to a long vector of metric ids, and use a stopper to differentiate ids for one contexts group
  Step 2: MPI_Gatherv to gather all long vectors to rank 0
    - get the size of the long vector and MPI_Gather to rank 0
    - rank 0 local exscan to get displacement for each rank's long vector
  Step 3: convert long vector (with all mids from all ranks) to rank 0's ctx_nzmids (global version now)
    - add the extra LastMidEnd to help marking the end location later for mid&offset pairs

    |----------|----------|       EACH RANK --\
    |context 0 | {0, 1, 2}|                   | 
    |----------|----------|      |---------------------------------------|
    |context 1 | {0, 2}   | ---> |0|1|2|stopper|0|2|stopper|0|1|3|stopper| --- \
    |----------|----------|      |---------------------------------------|     |
    |context 2 | {0, 1, 3}|                                                    |
    |----------|----------|                                                    |
                                                                               |
                                                                               |                                                                     
    RANK 0 --\                                                                 |
             |                                                                 |
    |-------------------------------------------------------------------|      |  
    |0|1|2|stopper|0|2|stopper|0|1|3|stopper|3|stopper|stopper|4|stopper|<--- /
    |-------------------------------------------------------------------|     
                                   |
                                   |
                                   |
                                  \|/
                       |----------|-------------------------|
                       |context 0 | {0, 1, 2, 3, LastMidEnd}|
        RANK 0 ----    |----------|-------------------------|
                       |context 1 | {0, 2, LastMidEnd}      | 
                       |----------|-------------------------|
                       |context 2 | {0, 1, 3, 4, LastMidEnd}|
                       |----------|-------------------------|                          
  */

  //STEP 1: convert to a long vector with stopper between mids for different contexts
  uint16_t stopper = -1;
  std::vector<uint16_t> rank_all_mids;
  for(auto ctx : ctx_nzmids){
    for(auto mid: ctx)
      rank_all_mids.emplace_back(mid);
    rank_all_mids.emplace_back(stopper);
  }

  //STEP 2: gather all rank_all_mids to rank 0
  //  prepare for later gatherv: get the size of each rank's rank_all_mids
  int rank_all_mids_size = rank_all_mids.size();
  std::vector<int> all_rank_mids_sizes;
  if(rank == 0) all_rank_mids_sizes.resize(num_proc);
  MPI_Gather(&rank_all_mids_size, 1, MPI_INT, all_rank_mids_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  //  prepare for later gatherv: get the displacement of each rank's rank_all_mids
  int total_size = 0;
  std::vector<int> all_rank_mids_disps; 
  std::vector<uint16_t> global_all_mids;
  if(rank == 0){
    all_rank_mids_disps.resize(num_proc);

    #pragma omp parallel for num_threads(threads)
    for(int i = 0; i<num_proc; i++) all_rank_mids_disps[i] = all_rank_mids_sizes[i];
    exscan<int>(all_rank_mids_disps,threads); 

    total_size = all_rank_mids_disps.back() + all_rank_mids_sizes.back();
    global_all_mids.resize(total_size);
  }

  //  gather all the rank_all_mids (i.e. ctx_nzmids) to root
  MPI_Gatherv(rank_all_mids.data(),rank_all_mids_size, MPI_UINT16_T,
    global_all_mids.data(), all_rank_mids_sizes.data(), all_rank_mids_disps.data(), MPI_UINT16_T, 0, MPI_COMM_WORLD);


  //STEP 3: turn the long vector global_all_mids back to rank 0's ctx_nzmids
  if(rank == 0){
    int num_stopper = 0;
    int num_ctx     = ctx_nzmids.size();
    for(int i = 0; i< total_size; i++) {
      uint16_t mid = global_all_mids[i];
      if(mid == stopper)
        num_stopper++;
      else
        ctx_nzmids[num_stopper % num_ctx].insert(mid); 
    }

    //  Add extra space for marking end location for the last mid
    #pragma omp parallel for num_threads(threads)
    for(uint i = 0; i<ctx_nzmids.size(); i++)
      ctx_nzmids[i].insert(LastMidEnd);
  }

}


// helper functions to help sum reduce a vector of things
void vSum ( uint64_t *, uint64_t *, int *, MPI_Datatype * );
void vSum(uint64_t *invec, uint64_t *inoutvec, int *len, MPI_Datatype *dtype)
{
    int i;
    for ( i=0; i<*len; i++ )
        inoutvec[i] += invec[i];
}


std::vector<uint64_t> SparseDB::getCtxOffset(const std::vector<uint64_t>& ctx_val_cnts, 
                                             const std::vector<std::set<uint16_t>>& ctx_nzmids,
                                             const int threads, const int rank)
{

  assert(ctx_val_cnts.size() == ctx_nzmids.size());

  std::vector<uint64_t> ctx_off (ctxcnt + 1);
  std::vector<uint64_t> local_ctx_off (ctxcnt + 1, 0);

  //get local sizes
  #pragma omp parallel for num_threads(threads)
  for(uint i = 0; i < ctx_val_cnts.size(); i++){
    local_ctx_off[i] = ctx_val_cnts[i] * CMS_val_prof_idx_pair_SIZE;
    //empty context also has LastMidEnd in ctx_nzmids, so if the size is 1, offet should not count that pair for calculation
    if(rank == 0 && ctx_nzmids[i].size() > 1) local_ctx_off[i] += ctx_nzmids[i].size() * CMS_m_pair_SIZE; 
  }

  //get local offsets
  exscan<uint64_t>(local_ctx_off,threads); 

  //sum up local offsets to get global offsets
  MPI_Op vectorSum;
  MPI_Op_create((MPI_User_function *)vSum, 1, &vectorSum);
  MPI_Allreduce(local_ctx_off.data(), ctx_off.data(), local_ctx_off.size(), MPI_UINT64_T, vectorSum, MPI_COMM_WORLD);
  MPI_Op_free(&vectorSum);

  return ctx_off;


}

//each rank is responsible for a group of ctxs
std::vector<uint32_t> SparseDB::getMyCtxs(const std::vector<uint64_t>& ctx_off,
                                          const int num_ranks, const int rank)
{
  assert(ctx_off.size() > 0);

  std::vector<uint32_t> my_ctxs;

  //split work among ranks by volume of ctxs
  uint64_t total_size = ctx_off.back();
  uint64_t max_size_per_rank = round(total_size/num_ranks);
  uint64_t my_start = rank * max_size_per_rank;
  uint64_t my_end = (rank == num_ranks - 1) ? total_size : (rank + 1) * max_size_per_rank;

  for(uint i = 1; i<ctx_off.size(); i++){
    if(ctx_off[i] > my_start && ctx_off[i] <= my_end) my_ctxs.emplace_back(CTXID((i-1)));
  }

  return my_ctxs;
}

void SparseDB::updateCtxOffset(const int threads, std::vector<uint64_t>& ctx_off)
{
  assert(ctx_off.size() == ctxcnt + 1);

  #pragma omp parallel for num_threads(threads)
  for(uint i = 0; i < ctxcnt + 1; i++){
    ctx_off[i] += (MULTIPLE_8(ctxcnt * CMS_ctx_info_SIZE)) + CMS_hdr_SIZE;
  }
}


//---------------------------------------------------------------------------
// get a list of profile info
//---------------------------------------------------------------------------
pms_profile_info_t SparseDB::profInfo(const char *input)
{
  pms_profile_info_t pi;
  interpretByte8(pi.id_tuple_ptr, input);
  interpretByte8(pi.metadata_ptr, input + PMS_id_tuple_ptr_SIZE);
  interpretByte8(pi.spare_one,    input + PMS_id_tuple_ptr_SIZE + PMS_metadata_ptr_SIZE);
  interpretByte8(pi.spare_two,    input + PMS_id_tuple_ptr_SIZE + PMS_metadata_ptr_SIZE + PMS_spare_one_SIZE);
  interpretByte8(pi.num_vals,     input + PMS_ptrs_SIZE);
  interpretByte4(pi.num_nzctxs,   input + PMS_ptrs_SIZE + PMS_num_val_SIZE);
  interpretByte8(pi.offset,       input + PMS_ptrs_SIZE + PMS_num_val_SIZE + PMS_num_nzctx_SIZE);
  return pi;
}


//read the Profile Information section of profile.db to get the list of profiles 
std::vector<pms_profile_info_t> SparseDB::profInfoList(const int threads, const util::File& fh)
{
  std::vector<pms_profile_info_t> prof_info;
  auto fhi = fh.open(false);

  //read the number of profiles
  uint32_t num_prof;
  SPARSE_exitIfMPIError(readAsByte4(num_prof,fhi, (HPCPROFILESPARSE_FMT_MagicLen + HPCPROFILESPARSE_FMT_VersionLen)),
     __FUNCTION__ + std::string(": cannot read the number of profiles"));
  num_prof--; //minus the summary profile

  //read the whole Profile Information section
  int count = num_prof * PMS_prof_info_SIZE; 
  char input[count];
  fhi.readat(PMS_hdr_SIZE + PMS_prof_info_SIZE, count, input); //skip one prof_info (summary)

  //interpret the section and store in a vector of pms_profile_info_t
  prof_info.resize(num_prof);
  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i<count; i += PMS_prof_info_SIZE){
    auto pi = profInfo(input + i);
    pi.prof_info_idx = i/PMS_prof_info_SIZE + 1; // the first one is summary profile
    prof_info[i/PMS_prof_info_SIZE] = std::move(pi);
  }

  return prof_info;
}


//---------------------------------------------------------------------------
// get all profiles' CtxIdIdxPairs
//---------------------------------------------------------------------------
SparseDB::PMS_CtxIdIdxPair SparseDB::ctxIdIdxPair(const char *input)
{
  PMS_CtxIdIdxPair ctx_pair;
  interpretByte4(ctx_pair.ctx_id,  input);
  interpretByte8(ctx_pair.ctx_idx, input + PMS_ctx_id_SIZE);
  return ctx_pair;
}

std::vector<SparseDB::PMS_CtxIdIdxPair> SparseDB::ctxIdIdxPairs(util::File::Instance& fh, const pms_profile_info_t pi)
{
  std::vector<PMS_CtxIdIdxPair> ctx_id_idx_pairs(pi.num_nzctxs + 1);
  if(pi.num_nzctxs == 0) return ctx_id_idx_pairs;

  //read the whole ctx_id_idx_pairs chunk
  int count = (pi.num_nzctxs + 1) * PMS_ctx_pair_SIZE; 
  char input[count];
  MPI_Offset ctx_pairs_offset = pi.offset + pi.num_vals * (PMS_val_SIZE + PMS_mid_SIZE);
  fh.readat(ctx_pairs_offset, count, input);

  //interpret the chunk and store accordingly
  for(int i = 0; i<count; i += PMS_ctx_pair_SIZE)
    ctx_id_idx_pairs[i/PMS_ctx_pair_SIZE] = std::move(ctxIdIdxPair(input + i));

  return ctx_id_idx_pairs;
}

std::vector<std::vector<SparseDB::PMS_CtxIdIdxPair>> 
SparseDB::getProfileCtxIdIdxPairs(const util::File& fh, const int threads,
                                  const std::vector<pms_profile_info_t>& prof_info)
{
  std::vector<std::vector<PMS_CtxIdIdxPair>> all_prof_ctx_pairs(prof_info.size());

  #pragma omp parallel for num_threads(threads)
  for(uint i = 0; i < prof_info.size(); i++){
    auto fhi = fh.open(false);
    all_prof_ctx_pairs[i] = std::move(ctxIdIdxPairs(fhi, prof_info[i]));
  }

  return all_prof_ctx_pairs;
}



//in a vector of PMS_CtxIdIdxPair, find one with target context id
//input: target context id, the vector we are searching through, 
//       length to search for (searching range index will be 0..length-1), 
//       whether already found previous one, the previous found context id (it no previously found, this var will not be used)
//output: found idx / SPARSE_END(already end of the vector, not found) / SPARSE_NOT_FOUND
//        (last argument) found ctx_id_idx_pair will be inserted if found 
int SparseDB::findOneCtxIdIdxPair(const uint32_t target_ctx_id,
                                  const std::vector<PMS_CtxIdIdxPair>& profile_ctx_pairs,
                                  const uint length, 
                                  const bool notfirst,
                                  const int found_ctx_idx, 
                                  std::vector<std::pair<uint32_t, uint64_t>>& my_ctx_pairs)
{
  int idx;

  if(notfirst){
    idx = 1 + found_ctx_idx;

    //the profile_ctx_pairs has been searched through
    if(idx == (int)length) return SPARSE_END;
    
    //the ctx_id at idx
    uint32_t prof_ctx_id = profile_ctx_pairs[idx].ctx_id;
    if(prof_ctx_id == target_ctx_id){
      //my_ctx_pairs.emplace(profile_ctx_pairs[idx].ctx_id,profile_ctx_pairs[idx].ctx_idx);
      my_ctx_pairs.emplace_back(profile_ctx_pairs[idx].ctx_id, profile_ctx_pairs[idx].ctx_idx);
      return idx;
    }else if(prof_ctx_id > target_ctx_id){
      return SPARSE_NOT_FOUND; //back to original since this might be next target
    }else{ //prof_ctx_id < target_ctx_id, should not happen
      std::ostringstream ss;
      ss << __FUNCTION__ << ": ctx id " << prof_ctx_id << " in a profile does not exist in the full ctx list while searching for " 
        << target_ctx_id << " with index " << idx;
      exitError(ss.str());
      return SPARSE_NOT_FOUND; //this should not be called if exit, but function requires return value
    }

  }else{
    PMS_CtxIdIdxPair target_ciip;
    target_ciip.ctx_id = target_ctx_id;
    idx = struct_member_binary_search(profile_ctx_pairs, target_ciip, &PMS_CtxIdIdxPair::ctx_id, length);
    if(idx >= 0){
      my_ctx_pairs.emplace_back(profile_ctx_pairs[idx].ctx_id, profile_ctx_pairs[idx].ctx_idx);
    }
    else if(idx == -1){
      idx = -1;  // JMA: Start future searches at idx 0
    }else if(idx < -1){
      idx = -2 - idx;
    }
    return idx;
  }
}


//from a group of PMS_CtxIdIdxPair of one profile, get the pairs related to a group of ctx_ids
//input: a vector of ctx_ids we are searching for, a vector of profile PMS_CtxIdIdxPairs we are searching through
//output: (last argument) a filled vector of PMS_CtxIdIdxPairs related to that group of ctx_ids
void SparseDB::findCtxIdIdxPairs(const std::vector<uint32_t>& ctx_ids,
                                 const std::vector<PMS_CtxIdIdxPair>& profile_ctx_pairs,
                                 std::vector<std::pair<uint32_t, uint64_t>>& my_ctx_pairs)
{
  assert(profile_ctx_pairs.size() > 1);

  uint n = profile_ctx_pairs.size() - 1; //since the last is LastNodeEnd
  int idx = -1; //index of current pair in profile_ctx_pairs
  bool notfirst = false;
  uint32_t target;

  for(uint i = 0; i<ctx_ids.size(); i++){
    target = ctx_ids[i];
    int ret = findOneCtxIdIdxPair(target, profile_ctx_pairs, n, notfirst, idx, my_ctx_pairs);
    if(ret == SPARSE_END) break;
    if(ret != SPARSE_NOT_FOUND){
      idx = ret;
      notfirst = true;
    } 
  }

  //add one extra context pair for later use
  //my_ctx_pairs.emplace(LastNodeEnd,profile_ctx_pairs[idx + 1].ctx_idx);
  my_ctx_pairs.emplace_back(LastNodeEnd, profile_ctx_pairs[idx + 1].ctx_idx);
  
  assert(my_ctx_pairs.size() <= ctx_ids.size() + 1);
}


// get the context id and index pairs for the group of contexts from one profile 
// input: prof_info for this profile, context ids we want, file handle
// output: found context id and idx pairs
int SparseDB::getMyCtxIdIdxPairs(const pms_profile_info_t& prof_info,
                                 const std::vector<uint32_t>& ctx_ids,
                                 const std::vector<PMS_CtxIdIdxPair>& prof_ctx_pairs,
                                 hpctoolkit::util::File::Instance& fh,
                                 //std::map<uint32_t, uint64_t>& my_ctx_pairs)
                                 std::vector<std::pair<uint32_t, uint64_t>>& my_ctx_pairs)
{
  if(prof_ctx_pairs.size() == 1) return SPARSE_ERR;

  findCtxIdIdxPairs(ctx_ids, prof_ctx_pairs, my_ctx_pairs);
  if(my_ctx_pairs.size() == 1) return SPARSE_ERR;

  return SPARSE_OK;
}





//---------------------------------------------------------------------------
// read and interpret one profie - ValMid
//---------------------------------------------------------------------------
// read all bytes for a group of contexts from one profile
void SparseDB::readValMidsBytes(const std::vector<uint32_t>& ctx_ids,
                                //std::map<uint32_t, uint64_t>& my_ctx_pairs,
                                std::vector<std::pair<uint32_t, uint64_t>>& my_ctx_pairs,
                                const pms_profile_info_t& prof_info,
                                util::File::Instance& fh,
                                std::vector<char>& bytes)
{

  uint64_t first_ctx_idx = my_ctx_pairs[0].second;
  uint64_t last_ctx_idx  = my_ctx_pairs.back().second;

  MPI_Offset val_mid_start_pos = prof_info.offset + first_ctx_idx * (PMS_val_SIZE + PMS_mid_SIZE);
  int val_mid_count = (last_ctx_idx - first_ctx_idx) * (PMS_val_SIZE + PMS_mid_SIZE);
  bytes.resize(val_mid_count);
  if(val_mid_count != 0) {
    fh.readat(val_mid_start_pos, val_mid_count, bytes.data());
  }

}

//create and return a new MetricValBlock
SparseDB::MetricValBlock SparseDB::createNewMetricValBlock(const hpcrun_metricVal_t val, 
                                                           const uint16_t mid,
                                                           const uint32_t prof_idx)
{
  MetricValBlock mvb;
  mvb.mid = mid;
  std::vector<std::pair<hpcrun_metricVal_t,uint32_t>> values_prof_idxs;
  values_prof_idxs.emplace_back(val, prof_idx);
  mvb.values_prof_idxs = values_prof_idxs;
  return mvb;
}

//create and return a new CtxMetricBlock
SparseDB::CtxMetricBlock SparseDB::createNewCtxMetricBlock(const hpcrun_metricVal_t val, 
                                                 const uint16_t mid,
                                                 const uint32_t prof_idx,
                                                 const uint32_t ctx_id)
{
  //create a new MetricValBlock for this mid, val, prof_idx
  MetricValBlock mvb = createNewMetricValBlock(val, mid, prof_idx);

  //create a vector of MetricValBlock for this context
  std::map<uint16_t, MetricValBlock> mvbs;
  mvbs.emplace(mid,mvb);

  //store it
  CtxMetricBlock cmb = {ctx_id, mvbs};
  return cmb;
}

//insert a pair of val and metric id to a CtxMetBlock they belong to (ctx id matches)
void SparseDB::insertValMidPair2OneCtxMetBlock(const hpcrun_metricVal_t val, 
                                               const uint16_t mid,
                                               const uint32_t prof_idx,
                                               CtxMetricBlock& cmb)
{
  //find if this mid exists
  std::map<uint16_t, MetricValBlock>& metric_blocks = cmb.metrics; 
  auto it = metric_blocks.find(mid);

  if(it != metric_blocks.end()){ //found mid
    it->second.values_prof_idxs.emplace_back(val, prof_idx);
  }else{ 
    metric_blocks.emplace(mid, createNewMetricValBlock(val, mid, prof_idx));
  }
}


//insert a triple of val, metric id and ctx_id to the correct place of ctx_met_blocks
void SparseDB::insertValMidCtxId2CtxMetBlocks(const hpcrun_metricVal_t val, 
                                              const uint16_t mid,
                                              const uint32_t prof_idx,
                                              const uint32_t ctx_id,
                                              CtxMetricBlock& cmb)
{

  //for single CtxMextricBlock
  assert(cmb.ctx_id == ctx_id);
  insertValMidPair2OneCtxMetBlock(val, mid, prof_idx, cmb);
  

}

//interpret the input bytes, assign value to val and metric id
void SparseDB::interpretOneValMidPair(const char *input,
                                      hpcrun_metricVal_t& val,
                                      uint16_t& mid)
{
  interpretByte8(val.bits, input);
  interpretByte2(mid,      input + PMS_val_SIZE);
}       

//interpret the bytes that has all val_mids for a group of contexts from one profile
//assign values accordingly to ctx_met_blocks
void SparseDB::interpretValMidsBytes(char *vminput,
                                     const uint32_t prof_idx,
                                     const std::pair<uint32_t,uint64_t>& ctx_pair,
                                     const uint64_t next_ctx_idx,
                                     const uint64_t first_ctx_idx,
                                     //std::map<uint32_t, uint64_t>& my_ctx_pairs,
                                     std::vector<std::pair<uint32_t, uint64_t>>& my_ctx_pairs,
                                     CtxMetricBlock& cmb)
{

  // for single CtxMetricBlock
  //uint64_t ctx_idx = my_ctx_pairs[ctx_id];
  uint32_t ctx_id = ctx_pair.first;
  uint64_t ctx_idx = ctx_pair.second;
  uint64_t num_val_this_ctx = next_ctx_idx - ctx_idx;

  char* ctx_met_input = vminput + (PMS_val_SIZE + PMS_mid_SIZE) * (ctx_idx - first_ctx_idx);
  for(uint i = 0; i < num_val_this_ctx; i++){
    hpcrun_metricVal_t val;
    uint16_t mid;
    interpretOneValMidPair(ctx_met_input, val, mid);

    insertValMidCtxId2CtxMetBlocks(val, mid, prof_idx, ctx_id, cmb);
    ctx_met_input += (PMS_val_SIZE + PMS_mid_SIZE);
  }

}


//---------------------------------------------------------------------------
// read and interpret ALL profies 
//---------------------------------------------------------------------------
//merge source CtxMetBlock to destination CtxMetBlock
//it has to be used for two blocks with the same ctx_id
void SparseDB::mergeCtxMetBlocks(const CtxMetricBlock& source,
                                 CtxMetricBlock& dest)
{
  assert(source.ctx_id == dest.ctx_id);

  std::map<uint16_t, MetricValBlock>& dest_metrics = dest.metrics;

  for(auto it = source.metrics.begin(); it != source.metrics.end(); it++){
    MetricValBlock source_m = it->second;
    const uint16_t& mid = source_m.mid;
    
    auto mvb_i = dest_metrics.find(mid);
    if(mvb_i == dest_metrics.end()){
      dest_metrics.emplace(mid,source_m);
    }else{
      mvb_i->second.values_prof_idxs.reserve(mvb_i->second.values_prof_idxs.size() + source_m.values_prof_idxs.size());
      mvb_i->second.values_prof_idxs.insert(mvb_i->second.values_prof_idxs.end(), source_m.values_prof_idxs.begin(), source_m.values_prof_idxs.end());
    }
  }
}


//merge all the CtxMetricBlocks from all the threads for one Ctx (one ctx_id) to dest
void SparseDB::mergeOneCtxAllThreadBlocks(const std::vector<std::map<uint32_t, CtxMetricBlock> *>& threads_ctx_met_blocks,
                                          CtxMetricBlock& dest)
{
  uint32_t ctx_id = dest.ctx_id;
  for(uint i = 0; i < threads_ctx_met_blocks.size(); i++){
    std::map<uint32_t, CtxMetricBlock>& thread_cmb = *threads_ctx_met_blocks[i];
    
    std::map<uint32_t, CtxMetricBlock>::iterator cmb_i = thread_cmb.find(ctx_id);   
    if(cmb_i != thread_cmb.end()){
      mergeCtxMetBlocks(cmb_i->second, dest);
    }
  }
}

//within each CtxMetricBlock sort based on metric id, within each MetricValBlock, sort based on thread id
void SparseDB::sortCtxMetBlocks(std::map<uint32_t, CtxMetricBlock>& ctx_met_blocks)
{

  //#pragma omp for  
  for(auto i = ctx_met_blocks.begin(); i != ctx_met_blocks.end(); i++){
    CtxMetricBlock& hcmb = i->second;

    for(auto j = hcmb.metrics.begin(); j != hcmb.metrics.end(); j++){
      MetricValBlock& mvb = j->second;

      std::sort(mvb.values_prof_idxs.begin(), mvb.values_prof_idxs.end(), 
        [](const std::pair<hpcrun_metricVal_t,uint32_t>& lhs, const std::pair<hpcrun_metricVal_t,uint32_t>& rhs) {
          return lhs.second < rhs.second;
        });  
      mvb.num_values = mvb.values_prof_idxs.size(); //update each mid's num_values
    }
  }

}

//read all the profiles and convert data to appropriate bytes for a group of contexts
//std::vector<std::pair<std::map<uint32_t, uint64_t>, std::vector<char>>>
std::vector<std::pair<std::vector<std::pair<uint32_t,uint64_t>>, std::vector<char>>>
SparseDB::readProfiles(const std::vector<uint32_t>& ctx_ids, 
                            const std::vector<pms_profile_info_t>& prof_info,
                            int threads,
                            const std::vector<std::vector<PMS_CtxIdIdxPair>>& all_prof_ctx_pairs,
                            const util::File& fh)
{

  std::vector<std::pair<std::vector<std::pair<uint32_t,uint64_t>>, std::vector<char>>> profiles_data (prof_info.size());

  //read all profiles for this ctx_ids group
  #pragma omp parallel for num_threads(threads) 
  for(uint i = 0; i < prof_info.size(); i++){
    auto fhi = fh.open(false);
    pms_profile_info_t pi = prof_info[i];
    std::vector<PMS_CtxIdIdxPair> prof_ctx_pairs = all_prof_ctx_pairs[i];
    
    std::vector<std::pair<uint32_t,uint64_t>> my_ctx_pairs;
    int ret = getMyCtxIdIdxPairs(pi, ctx_ids, prof_ctx_pairs, fhi, my_ctx_pairs);
  
    std::vector<char> vmbytes;
    if(ret == SPARSE_OK){
      readValMidsBytes(ctx_ids, my_ctx_pairs, pi, fhi, vmbytes);
    }

    profiles_data[i] = {std::move(my_ctx_pairs), std::move(vmbytes)};
  }
  
  return profiles_data;
}


//---------------------------------------------------------------------------
// convert ctx_met_blocks to correct bytes for writing
//---------------------------------------------------------------------------
//convert one MetricValBlock to bytes at bytes location
//return number of bytes converted
int SparseDB::convertOneMetricValBlock(const MetricValBlock& mvb,                                        
                                       char *bytes)
{
  char* bytes_pos = bytes;
  for(uint i = 0; i < mvb.values_prof_idxs.size(); i++){
    auto& pair = mvb.values_prof_idxs[i];
    convertToByte8(pair.first.bits, bytes_pos);
    convertToByte4(pair.second,     bytes_pos + CMS_val_SIZE);
    bytes_pos += CMS_val_prof_idx_pair_SIZE;
  }

  return (bytes_pos - bytes);
}

//convert ALL MetricValBlock of one CtxMetricBlock to bytes at bytes location
//return number of bytes converted
int SparseDB::convertCtxMetrics(std::map<uint16_t, MetricValBlock>& metrics,                                        
                                char *bytes)
{
  uint64_t num_vals = 0;

  char* mvb_pos = bytes; 
  for(auto i = metrics.begin(); i != metrics.end(); i++){
    MetricValBlock mvb = i->second;
    i->second.num_values = mvb.values_prof_idxs.size(); //if we use sort in readProfiles, we don't need this
    num_vals += mvb.values_prof_idxs.size();

    int bytes_converted = convertOneMetricValBlock(mvb, mvb_pos);
    mvb_pos += bytes_converted;
  }

  assert(num_vals == (uint)(mvb_pos - bytes)/CMS_val_prof_idx_pair_SIZE);

  return (mvb_pos - bytes);
}

//build metric id and idx pairs for one context as bytes to bytes location
//return number of bytes built
int SparseDB::buildCtxMetIdIdxPairsBytes(const std::map<uint16_t, MetricValBlock>& metrics,                                        
                                         char *bytes)
{
  char* bytes_pos = bytes;
  uint64_t m_idx = 0;
  for(auto i = metrics.begin(); i != metrics.end(); i++){
    uint16_t mid = i->first;
    convertToByte2(mid,   bytes_pos);
    convertToByte8(m_idx, bytes_pos + CMS_mid_SIZE);
    bytes_pos += CMS_m_pair_SIZE;
    m_idx += i->second.num_values;
  }

   uint16_t mid = LastMidEnd;
   convertToByte2(mid,   bytes_pos);
   convertToByte8(m_idx, bytes_pos + CMS_mid_SIZE);
   bytes_pos += CMS_m_pair_SIZE;

  return (bytes_pos - bytes);
}

//convert one CtxMetricBlock to bytes at bytes location
//also assigne value to num_nzmids and num_vals of this context
//return number of bytes converted
int SparseDB::convertOneCtxMetricBlock(const CtxMetricBlock& cmb,                                        
                                       char *bytes,
                                       uint16_t& num_nzmids,
                                       uint64_t& num_vals)
{
  std::map<uint16_t, MetricValBlock> metrics = cmb.metrics;
  num_nzmids = metrics.size();

  int bytes_converted = convertCtxMetrics(metrics, bytes);
  num_vals = bytes_converted/CMS_val_prof_idx_pair_SIZE;

  bytes_converted += buildCtxMetIdIdxPairsBytes(metrics, bytes + num_vals * CMS_val_prof_idx_pair_SIZE);

  return bytes_converted;
}

//convert one cms_ctx_info_t to bytes at bytes location
int SparseDB::convertOneCtxInfo(const cms_ctx_info_t& ctx_info,                                        
                                char *bytes)
{
  convertToByte4(ctx_info.ctx_id,   bytes);
  convertToByte8(ctx_info.num_vals,  bytes + CMS_ctx_id_SIZE);
  convertToByte2(ctx_info.num_nzmids,bytes + CMS_ctx_id_SIZE + CMS_num_val_SIZE);
  convertToByte8(ctx_info.offset,   bytes + CMS_ctx_id_SIZE + CMS_num_val_SIZE + CMS_num_nzmid_SIZE);
  
  return CMS_ctx_info_SIZE;
}


//convert one ctx (whose id is ctx_id), including info and metrics to bytes
//info will be converted to info_bytes, metrics will be converted to met_bytes
//info_byte_cnt and met_byte_cnt will be assigned number of bytes converted
void SparseDB::convertOneCtx(const uint32_t ctx_id, 
                             const uint64_t next_ctx_off,    
                             const CtxMetricBlock& cmb,                          
                             const uint64_t first_ctx_off,
                             cms_ctx_info_t& ci,
                             uint& met_byte_cnt,
                             char* met_bytes)

{

  met_byte_cnt += convertOneCtxMetricBlock(cmb, met_bytes + ci.offset - first_ctx_off, ci.num_nzmids, ci.num_vals);

  if(ci.num_nzmids != 0)
    if(ci.offset + ci.num_vals * CMS_val_prof_idx_pair_SIZE + (ci.num_nzmids+1) * CMS_m_pair_SIZE !=  next_ctx_off){
      printf("ctx_id %d, offset: %ld, num_vals: %ld, num_nzmids %d, next off %ld\n", ctx_id, ci.offset, ci.num_vals, ci.num_nzmids, next_ctx_off );
      exitError("collected cct data (num_vals:" + std::to_string(ci.num_vals) + " /num_nzmids:" + std::to_string(ci.num_nzmids) + ") were wrong !");
    }

}

//convert a group of contexts to appropriate bytes 
void SparseDB::ctxBlocks2Bytes(const CtxMetricBlock& cmb, 
                               const std::vector<uint64_t>& ctx_off, 
                               const uint32_t& ctx_id,
                               int threads,
                               std::vector<char>& metrics_bytes)
{
  assert(ctx_off.size() > 0);

  //uint64_t first_ctx_off =  ctx_off[CTX_VEC_IDX(ctx_ids[0])]; 
  uint64_t first_ctx_off =  ctx_off[CTX_VEC_IDX(ctx_id)]; 
  //uint info_byte_cnt = 0;
  uint met_byte_cnt  = 0;

  //for single cmb
  cms_ctx_info_t ci = {ctx_id, 0, 0, ctx_off[CTX_VEC_IDX(ctx_id)]};

  if(cmb.metrics.size() > 0)
    convertOneCtx(ctx_id, ctx_off[CTX_VEC_IDX(ctx_id)+1], cmb, first_ctx_off, ci, met_byte_cnt, metrics_bytes.data());

  if(met_byte_cnt  != metrics_bytes.size()) exitError("the count of metrics_bytes converted is not as expected" 
    + std::to_string(met_byte_cnt) + " != " + std::to_string(metrics_bytes.size()));

}


//given ctx_met_blocks, convert all and write everything for the group of contexts, to the ofh file 
void SparseDB::writeOneCtx(const uint32_t& ctx_id,
                           const std::vector<uint64_t>& ctx_off,
                           const CtxMetricBlock& cmb,
                           const int threads,
                           util::File::Instance& ofh)
{
  //std::vector<char> info_bytes (CMS_ctx_info_SIZE);

  int metric_bytes_size = ctx_off[CTX_VEC_IDX(ctx_id) + 1] - ctx_off[CTX_VEC_IDX(ctx_id)];
  std::vector<char> metrics_bytes (metric_bytes_size);
  
  //ctxBlocks2Bytes(cmb, ctx_off, ctx_id, threads, info_bytes, metrics_bytes);
  ctxBlocks2Bytes(cmb, ctx_off, ctx_id, threads, metrics_bytes);

  //MPI_Offset info_off = CMS_hdr_SIZE + CTX_VEC_IDX(ctx_id) * CMS_ctx_info_SIZE;
  //ofh.writeat(info_off, info_bytes.size(), info_bytes.data());

  MPI_Offset metrics_off = ctx_off[CTX_VEC_IDX(ctx_id)];
  ofh.writeat(metrics_off, metrics_bytes.size(), metrics_bytes.data());
}


//---------------------------------------------------------------------------
// read and write for all contexts in this rank's list
//---------------------------------------------------------------------------
//read a context group's data and write them out
void SparseDB::rwOneCtxGroup(const std::vector<uint32_t>& ctx_ids, 
                             const std::vector<pms_profile_info_t>& prof_info, 
                             const std::vector<uint64_t>& ctx_off, 
                             const int threads, 
                             const std::vector<std::vector<PMS_CtxIdIdxPair>>& all_prof_ctx_pairs,
                             const util::File& fh,
                             const util::File& ofh)
{
  if(ctx_ids.size() == 0) return;

  //read corresponding ctx_id_idx pairs and relevant ValMidsBytes
  std::vector<std::pair<std::vector<std::pair<uint32_t,uint64_t>>, std::vector<char>>> profiles_data =
    readProfiles(ctx_ids, prof_info, threads, all_prof_ctx_pairs, fh);

  struct nextCtx{
    uint32_t ctx_id;
    uint32_t prof_idx; 
    size_t cursor;

    //turn MaxHeap to MinHeap
    bool operator<(const nextCtx& a) const{
      if(ctx_id == a.ctx_id){
        return prof_idx > a.prof_idx;
      }
      return ctx_id > a.ctx_id;  
    }
  };

  uint first_ctx_off = ctx_off[CTX_VEC_IDX(ctx_ids.front())];
  uint total_ctx_ids_size = ctx_off[CTX_VEC_IDX(ctx_ids.back()) + 1] - first_ctx_off;
  uint thread_ctx_ids_size = round(total_ctx_ids_size/threads);

  std::vector<uint64_t> t_starts (threads, 0);
  std::vector<uint64_t> t_ends (threads, 0);
  int cur_thread = 0;

  //make sure first thread at least gets one ctx
  size_t cur_size = ctx_off[CTX_VEC_IDX(ctx_ids.front()) + 1] - first_ctx_off; //size of first ctx
  t_starts[cur_thread] = CTXID(0);
  if(threads > 1){
    for(uint i = 2; i <= ctx_ids.size(); i++){
      auto cid = (i == ctx_ids.size()) ? ctx_ids[i-1] + 1 : ctx_ids[i];
      auto cid_size = (ctx_off[CTX_VEC_IDX(cid)] - ctx_off[CTX_VEC_IDX(ctx_ids[i-1])]);

      if(cur_size > thread_ctx_ids_size){
        t_ends[cur_thread] = CTXID(i-1);
        cur_thread++;
        t_starts[cur_thread] = CTXID(i-1);
        cur_size = cid_size;

        if(cur_thread == threads - 1){
          t_ends[cur_thread] = CTXID(ctx_ids.size());
          break;
        } 
      }else{
        cur_size += cid_size;
      }
    }  
    if(cur_thread != threads-1) t_ends[cur_thread] = CTXID(ctx_ids.size());
  }else{
    t_ends[cur_thread] = CTXID(ctx_ids.size());
  }
  

  //for each ctx, find corresponding ctx_id_idx and bytes, and interpret
  #pragma omp parallel num_threads(threads)
  {
    auto ofhi = ofh.open(true);

    //each thread is responsible for a group of ctx_ids, idx from [my_start, my_end)
    int thread_num = omp_get_thread_num();
    uint my_start = t_starts[thread_num];
    uint my_end = t_ends[thread_num];

    if(my_start < my_end) {     
      //each thread sets up a heap to store <ctx_id, profile_idx, profile_cursor> for each profile
      std::vector<nextCtx> heap;
      heap.reserve(profiles_data.size());
      for(uint i = 0; i < profiles_data.size(); i++){
        uint32_t ctx_id = ctx_ids[my_start];
        std::vector<std::pair<uint32_t, uint64_t>>& ctx_id_idx_pairs = profiles_data[i].first;
        if(ctx_id_idx_pairs.empty()) continue;
        size_t cursor =lower_bound(ctx_id_idx_pairs.begin(),ctx_id_idx_pairs.end(), 
                  std::make_pair(ctx_id,std::numeric_limits<uint64_t>::min()), //Value to compare
                  [](const std::pair<uint32_t, uint64_t>& lhs, const std::pair<uint32_t, uint64_t>& rhs)      
                  { return lhs.first < rhs.first ;}) - ctx_id_idx_pairs.begin();
        heap.push_back({ctx_id_idx_pairs[cursor].first, i, cursor});
      }
      heap.shrink_to_fit();
      std::make_heap(heap.begin(), heap.end());

      while(1){
        //get the min ctx_id in the heap
        uint32_t ctx_id = heap.front().ctx_id;
        if(ctx_id > ctx_ids[my_end-1]) break;

        //a new CtxMetricBlock
        CtxMetricBlock cmb;
        cmb.ctx_id = ctx_id;

        while(heap.front().ctx_id == ctx_id){
          uint32_t prof_idx = heap.front().prof_idx;

          std::vector<char>& vmbytes = profiles_data[prof_idx].second;
          std::vector<std::pair<uint32_t, uint64_t>>& ctx_id_idx_pairs = profiles_data[prof_idx].first;
          
          uint64_t next_ctx_idx = ctx_id_idx_pairs[heap.front().cursor+1].second;
          uint64_t first_ctx_idx = ctx_id_idx_pairs[0].second;
          interpretValMidsBytes(vmbytes.data(), prof_info[prof_idx].prof_info_idx, ctx_id_idx_pairs[heap.front().cursor], next_ctx_idx, first_ctx_idx, ctx_id_idx_pairs, cmb);
          
          std::pop_heap(heap.begin(), heap.end());
          heap.back().cursor++;
          heap.back().ctx_id = ctx_id_idx_pairs[heap.back().cursor].first;
          std::push_heap(heap.begin(), heap.end());

        }

        writeOneCtx(ctx_id, ctx_off, cmb, threads, ofhi);

      }//END of while

    } //END of if my_start < my_end

  }//END of parallel region
  
   
}

//read ALL context groups' data and write them out
void SparseDB::rwAllCtxGroup(const std::vector<uint32_t>& my_ctxs, 
                             const std::vector<pms_profile_info_t>& prof_info, 
                             const std::vector<uint64_t>& ctx_off, 
                             const int threads, 
                             const std::vector<std::vector<PMS_CtxIdIdxPair>>& all_prof_ctx_pairs,
                             const util::File& fh,
                             const util::File& ofh,
                             const int rank)//TEMP
{
  //For each ctx group (< memory limit) this rank is in charge of, read and write
  std::vector<uint32_t> ctx_ids;
  size_t cur_size = 0;
  int cur_cnt = 0;
  uint64_t size_limit = std::min<std::size_t>(1024*1024*100, ctx_off[CTX_VEC_IDX(my_ctxs.back()) + 1] - ctx_off[CTX_VEC_IDX(my_ctxs.front())]);

  for(uint i =0; i<my_ctxs.size(); i++){
    uint32_t ctx_id = my_ctxs[i];
    size_t cur_ctx_size = ctx_off[CTX_VEC_IDX(ctx_id) + 1] - ctx_off[CTX_VEC_IDX(ctx_id)];

    if((cur_size + cur_ctx_size) <= size_limit){
      ctx_ids.emplace_back(ctx_id);
      cur_size += cur_ctx_size;
      cur_cnt++;
    }else{
      rwOneCtxGroup(ctx_ids, prof_info, ctx_off, threads, all_prof_ctx_pairs, fh, ofh);

      ctx_ids.clear();
      ctx_ids.emplace_back(ctx_id);
      cur_size = cur_ctx_size;
      cur_cnt = 1;
    }   

    // final ctx group
    if((i == my_ctxs.size() - 1) && (ctx_ids.size() != 0)) 
      rwOneCtxGroup(ctx_ids, prof_info, ctx_off, threads, all_prof_ctx_pairs, fh, ofh);
    
  }

}


void SparseDB::writeCCTMajor(const std::vector<uint64_t>& ctx_nzval_cnts, 
                             std::vector<std::set<uint16_t>>& ctx_nzmids,
                             const int world_rank, 
                             const int world_size, 
                             const int threads)
{
  //Prepare a union ctx_nzmids, only rank 0's ctx_nzmids is global
  unionMids(ctx_nzmids,world_rank,world_size, threads);

  //Get context global final offsets for cct.db
  auto ctx_off = getCtxOffset(ctx_nzval_cnts, ctx_nzmids, threads, world_rank);
  auto my_ctxs = getMyCtxs(ctx_off, world_size, world_rank);
  updateCtxOffset(threads, ctx_off);

  //Prepare files to read and write, get the list of profiles
  util::File profile_major_f(dir / "profile.db", false);
  util::File cct_major_f(dir / "cct.db", true);
  
  if(world_rank == 0){
    auto cct_major_fi = cct_major_f.open(true);
    // Write hdr
    writeCMSHdr(cct_major_fi);
    // Write ctx info section
    writeCtxInfoSec(ctx_nzmids, ctx_off, cct_major_fi);
  }

  //get the list of prof_info
  auto prof_info_list = profInfoList(threads, profile_major_f);

  //get the ctx_id & ctx_idx pairs for all profiles
  auto all_prof_ctx_pairs = getProfileCtxIdIdxPairs(profile_major_f, threads, prof_info_list);
  
  rwAllCtxGroup(my_ctxs, prof_info_list, ctx_off, threads, all_prof_ctx_pairs, profile_major_f, cct_major_f, world_rank); //TEMP: world_rank
}


//***************************************************************************
// general - YUMENG
//***************************************************************************

void SparseDB::merge(int threads, bool debug) {
  int world_rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  ctxcnt = mpi::bcast(ctxcnt, 0);

  {
    util::log::debug msg{false};  // Switch to true for CTX id printouts
    msg << "CTXs (" << world_rank << ":" << sparseInputs.size() << "): "
        << ctxcnt;
  }

  std::vector<uint64_t> ctx_nzval_cnts (ctxcnt,0);
  std::set<uint16_t> empty;
  std::vector<std::set<uint16_t>> ctx_nzmids(ctxcnt,empty);
  keepTemps = debug;
  writeProfileMajor(threads,world_rank,world_size, ctx_nzval_cnts, ctx_nzmids);
  writeCCTMajor(ctx_nzval_cnts,ctx_nzmids, world_rank, world_size, threads);

}


//local exscan over a vector of T, value after exscan will be stored in the original vector
template <typename T>
void SparseDB::exscan(std::vector<T>& data, int threads) {
  int n = data.size();
  int rounds = ceil(std::log2(n));
  std::vector<T> tmp (n);

  for(int i = 0; i<rounds; i++){
    #pragma omp parallel for num_threads(threads)
    for(int j = 0; j < n; j++){
      int p = (int)pow(2.0,i);
      tmp.at(j) = (j<p) ?  data.at(j) : data.at(j) + data.at(j-p);
    }
    if(i<rounds-1) data = tmp;
  }

  if(n>0) data[0] = 0;
  #pragma omp parallel for num_threads(threads)
  for(int i = 1; i < n; i++){
    data[i] = tmp[i-1];
  }
}


//binary search over a vector of T, unlike std::binary_search, which only returns true/false, 
//this returns the idx of found one, SPARSE_ERR as NOT FOUND
template <typename T, typename MemberT>
int SparseDB::struct_member_binary_search(const std::vector<T>& datas, const T target, const MemberT target_type, const int length) {
  int m;
  int L = 0;
  int R = length - 1;
  while(L <= R){
    m = (L + R) / 2;

    auto target_val = target.*target_type;
    auto comp_val   = datas[m].*target_type;

    if(comp_val < target_val){
      L = m + 1;
    }else if(comp_val > target_val){
      R = m - 1;
    }else{ //find match
      return m;
    }
  }
  //return SPARSE_NOT_FOUND; 
  return (R == -1) ? R : (-2 - R); //make it negative to differentiate it from found
}


template<class A, class B>
MPI_Datatype SparseDB::createPairType(MPI_Datatype aty, MPI_Datatype bty) {
  using realtype = std::pair<A, B>;
  std::array<int, 2> cnts = {1, 1};
  std::array<MPI_Datatype, 2> types = {aty, bty};
  std::array<MPI_Aint, 2> offsets = {offsetof(realtype, first), offsetof(realtype, second)};
  MPI_Datatype outtype;
  MPI_Type_create_struct(2, cnts.data(), offsets.data(), types.data(), &outtype);
  MPI_Type_commit(&outtype);
  return outtype;
}
  



//use for MPI error 
void SparseDB::exitMPIError(int error_code, std::string info)
{
  char estring[MPI_MAX_ERROR_STRING];
  int len;
  MPI_Error_string(error_code, estring, &len);
  //util::log::fatal() << info << ": " << estring;
  std::cerr << "FATAL: " << info << ": " << estring << "\n";
  MPI_Abort(MPI_COMM_WORLD, error_code);
}

//use for regular error
void SparseDB::exitError(std::string info)
{
  //TODO: consider how to terminate mpi processes gracefully
  //util::log::fatal() << info << "\n";
  std::cerr << "FATAL: " << info << "\n";
  MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
}


