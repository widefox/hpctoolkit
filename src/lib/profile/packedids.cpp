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
// Copyright ((c)) 2019-2020, Rice University
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

#include "packedids.hpp"

#include "util/log.hpp"

using namespace hpctoolkit;

// Helpers for packing various things.
static void pack(std::vector<std::uint8_t>& out, const std::string& s) noexcept {
  out.reserve(out.size() + s.size() + 1);  // Allocate the space early
  for(auto c: s) {
    if(c == '\0') c = '?';
    out.push_back(c);
  }
  out.push_back('\0');
}
static void pack(std::vector<std::uint8_t>& out, const std::uint64_t v) noexcept {
  // Little-endian order. Just in case the compiler can optimize it away.
  for(int shift = 0x00; shift < 0x40; shift += 0x08)
    out.push_back((v >> shift) & 0xff);
}

IdPacker::IdPacker() : stripcnt(0), buffersize(0) {};

Context& IdPacker::Classifier::context(Context& c, Scope& s) {
  std::vector<std::reference_wrapper<Context>> v;
  auto& cc = ClassificationTransformer::context(c, s, v);

  // We also need the ID of the final child Context. So we just emit it
  // and let the anti-recursion keep us from spinning out of control.
  v.emplace_back(sink.context(cc, s));

  // Check that we haven't handled this particular Context already
  auto& strip = v.back().get().userdata[shared.udOnce];
  if(strip.once.test_and_set(std::memory_order_acquire)) return cc;
  shared.stripcnt.fetch_add(1, std::memory_order_relaxed);

  // Nab a pseudo-random buffer to fill with our data
  auto hash = std::hash<Context*>{}(&c) ^ std::hash<Scope>{}(s);
  static_assert(std::numeric_limits<decltype(hash)>::radix == 2, "Non-binary architecture?");
  unsigned char idx = hash & 0xff;
  for(int i = 8; i < std::numeric_limits<decltype(hash)>::digits; i += 8)
    idx ^= (hash >> i) & 0xff;

  auto& buffer = shared.stripbuffers[idx].second;
  std::unique_lock<std::mutex> lock(shared.stripbuffers[idx].first);
  auto oldsz = buffer.size();

  // Now we can write the entry for out friends to work with.
  // Format: [parent id] (Scope) [cnt] ([type] [child id])...
  auto cid = c.userdata[sink.identifier()];
  pack(buffer, (std::uint64_t)cid);
  if(s.type() == Scope::Type::point) {
    // Format: [module id] [offset]
    auto mo = s.point_data();
    pack(buffer, (std::uint64_t)mo.first.userdata[sink.identifier()]);
    pack(buffer, (std::uint64_t)mo.second);
  } else if(s.type() == Scope::Type::unknown) {
    // Format: [magic]
    pack(buffer, (std::uint64_t)0xF0F1F2F3ULL << 32);
  } else
    util::log::fatal{} << "PackedIds can't handle non-point Contexts!";
  pack(buffer, (std::uint64_t)v.size());
  for(Context& ct: v) {
    switch(ct.scope().type()) {
    case Scope::Type::global:
      util::log::fatal{} << "Global Contexts shouldn't come out of expansion!";
      break;
    case Scope::Type::unknown:
    case Scope::Type::point:
      buffer.emplace_back(0);
      break;
    case Scope::Type::function:
    case Scope::Type::inlined_function:
      buffer.emplace_back(1);
      break;
    case Scope::Type::loop:
      buffer.emplace_back(2);
      break;
    }
    pack(buffer, (std::uint64_t)ct.userdata[sink.identifier()]);
  }

  shared.buffersize.fetch_add(buffer.size() - oldsz, std::memory_order_relaxed);

  return cc;
}

IdPacker::Sink::Sink(IdPacker& s) : shared(s) {};

void IdPacker::Sink::notifyPipeline() noexcept {
  shared.udOnce = src.structs().context.add<ctxonce>(std::ref(*this));
}

void IdPacker::Sink::notifyWavefront(DataClass ds) {
  if(ds.hasReferences() && ds.hasContexts()) {  // This is it!
    std::vector<uint8_t> ct;
    // Format: [global id] [mod cnt] (modules) [map cnt] (map entries...)
    pack(ct, (std::uint64_t)src.contexts().userdata[src.identifier()]);

    std::vector<std::string> mods;
    for(const Module& m: src.modules().iterate()) {
      const auto& id = m.userdata[src.identifier()];
      if(mods.size() <= id) mods.resize(id+1, "");
      mods.at(id) = m.path().string();
    }
    pack(ct, (std::uint64_t)mods.size());
    for(auto& s: mods) pack(ct, std::move(s));

    pack(ct, (std::uint64_t)shared.stripcnt.load(std::memory_order_relaxed));
    ct.reserve(ct.size() + shared.buffersize.load(std::memory_order_relaxed));
    for(const auto& ls: shared.stripbuffers)
      ct.insert(ct.end(), ls.second.begin(), ls.second.end());

    // Format: ... [met cnt] ([id] [p id] [ex id] [inc id] [name])...
    pack(ct, (std::uint64_t)src.metrics().size());
    for(auto& m: src.metrics().citerate()) {
      pack(ct, (std::uint64_t)m().userdata[src.identifier()]);
      const auto& ids = m().userdata[src.mscopeIdentifiers()];
      const auto& sc = m().scopes();
      pack(ct, (std::uint64_t)(sc.has(MetricScope::point) ? ids.point : -1));
      pack(ct, (std::uint64_t)(sc.has(MetricScope::function) ? ids.function : -1));
      pack(ct, (std::uint64_t)(sc.has(MetricScope::execution) ? ids.execution : -1));
      pack(ct, m().name());
    }

    notifyPacked(std::move(ct));
  }
}

// Helpers for unpacking various things
template<class T> static T unpack(std::vector<uint8_t>::const_iterator&) noexcept;
template<>
std::string unpack<std::string>(std::vector<uint8_t>::const_iterator& it) noexcept {
  std::string out;
  for(; *it != '\0'; ++it) out += *it;
  ++it;  // First location after the string
  return out;
}
template<>
std::uint64_t unpack<std::uint64_t>(std::vector<uint8_t>::const_iterator& it) noexcept {
  // Little-endian order. Same as in sinks/packed.cpp.
  std::uint64_t out = 0;
  for(int shift = 0x00; shift < 0x40; shift += 0x08) {
    out |= ((std::uint64_t)*it) << shift;
    ++it;
  }
  return out;
}

IdUnpacker::IdUnpacker(std::vector<uint8_t>&& c) : ctxtree(std::move(c)) {
  auto it = ctxtree.cbegin();
  globalid = ::unpack<std::uint64_t>(it);
}

void IdUnpacker::unpack(ProfilePipeline::Source& sink) {
  auto it = ctxtree.cbegin();
  ::unpack<std::uint64_t>(it);  // Skip over the global id

  exmod = &sink.module("/nonexistent/exmod");
  exfile = &sink.file("/nonexistent/exfile");
  exfunc.reset(new Function(*exmod));

  auto cnt = ::unpack<std::uint64_t>(it);
  for(std::size_t i = 0; i < cnt; i++)
    modmap.emplace_back(sink.module(::unpack<std::string>(it)));

  cnt = ::unpack<std::uint64_t>(it);
  for(std::size_t i = 0; i < cnt; i++) {
    Scope s;
    // Format: [parent id] (Scope) [cnt] [children ids]...
    unsigned int parent = ::unpack<std::uint64_t>(it);
    auto next = ::unpack<std::uint64_t>(it);
    if(next == (0xF0F1F2F3ULL << 32)) {
      // Format: [magic]
      s = {};  // Unknown Scope
    } else {
      // Format: [module id] [offset]
      auto off = ::unpack<std::uint64_t>(it);
      s = {modmap.at(next), off};
    }
    std::size_t cnt = ::unpack<std::uint64_t>(it);
    auto& scopes = exmap[parent][s];
    for(std::size_t x = 0; x < cnt; x++) {
      auto ty = *it;
      it++;
      auto id = ::unpack<std::uint64_t>(it);
      switch(ty) {
      case 0:  // unknown or point -> point
        scopes.emplace_back(*exmod, id);
        break;
      case 1:  // function or inlined_function -> inlined_function
        scopes.emplace_back(*exfunc, *exfile, id);
        break;
      case 2:  // loop -> loop
        scopes.emplace_back(Scope::loop, *exfile, id);
        break;
      default:
        util::log::fatal{} << "Unrecognized packed Scope type " << ty;
      }
    }
  }

  cnt = ::unpack<std::uint64_t>(it);
  for(std::size_t i = 0; i < cnt; i++) {
    auto id = ::unpack<std::uint64_t>(it);
    Metric::ScopedIdentifiers ids;
    ids.point = ::unpack<std::uint64_t>(it);
    ids.function = ::unpack<std::uint64_t>(it);
    ids.execution = ::unpack<std::uint64_t>(it);
    auto name = ::unpack<std::string>(it);
    metmap.insert({std::move(name), {id, ids}});
  }

  ctxtree.clear();
}

Context& IdUnpacker::Expander::context(Context& c, Scope& s) {
  util::call_once(shared.once, [this]{ shared.unpack(sink); });
  Context* cp = &c;
  bool first = true;
  for(const auto& next: shared.exmap.at(c.userdata[sink.identifier()]).at(s)) {
    if(!first) cp = &sink.context(*cp, s);
    s = next;
    first = false;
  }
  return *cp;
}

void IdUnpacker::Finalizer::context(const Context& c, unsigned int& id) {
  switch(c.scope().type()) {
  case Scope::Type::global:
    id = shared.globalid;
    return;
  case Scope::Type::point: {
    auto mo = c.scope().point_data();
    if(&mo.first != shared.exmod)
      util::log::fatal{} << "Point scope with real Module in IdUnpacker!";
    id = mo.second;
    return;
  }
  case Scope::Type::inlined_function:
    if(&c.scope().function_data() != shared.exfunc.get())
      util::log::fatal{} << "inlined_function scope with real Function in IdUnpacker!";
    // fallthrough
  case Scope::Type::loop: {
    auto fl = c.scope().line_data();
    if(&fl.first != shared.exfile)
      util::log::fatal{} << "inlined_function scope with real File in IdUnpacker!";
    id = fl.second;
    return;
  }
  default:
    util::log::fatal{} << "Unrecognized Scope in IdUnpacker!";
  }
}

void IdUnpacker::Finalizer::metric(const Metric& m, unsigned int& id) {
  util::call_once(shared.once, [this]{ shared.unpack(sink); });
  auto it = shared.metmap.find(m.name());
  if(it == shared.metmap.end())
    util::log::fatal{} << "Unrecognized metric in IdUnpacker: " << m.name();
  id = it->second.first;
}

void IdUnpacker::Finalizer::metric(const Metric& m, Metric::ScopedIdentifiers& ids) {
  util::call_once(shared.once, [this]{ shared.unpack(sink); });
  auto it = shared.metmap.find(m.name());
  if(it == shared.metmap.end())
    util::log::fatal{} << "Unrecognized metric in IdUnpacker: " << m.name();
  ids = it->second.second;
}
