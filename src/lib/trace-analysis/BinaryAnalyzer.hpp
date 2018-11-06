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
// Copyright ((c)) 2002-2017, Rice University
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

/* 
 * File:   BinaryAnalyzer.hpp
 * Author: Lai Wei <lai.wei@rice.edu>
 *
 * Created on March 1, 2018, 11:40 PM
 * 
 * Analyzes execution binaries to derive control flows among call sites and loops.
 */

#ifndef BINARYANALYZER_HPP
#define BINARYANALYZER_HPP

#include <string>
using std::string;

#include <unordered_map>
using std::unordered_map;

#include "TraceAnalysisCommon.hpp"
#include "data/TCT-CFG.hpp"

namespace TraceAnalysis {
    
  class BinaryAnalyzer {
  public:
    BinaryAnalyzer();
    virtual ~BinaryAnalyzer();

    bool parse(const string& filename);

    // Return a pointer to CFGFunc associated with vma.
    // NULL if no such CFGFunc exist.
    CFGFunc* findFunc(VMA vma);
    
    // Return a pointer to CFGLoop associated with vma.
    // NULL if no such CFGFunc exist.
    CFGLoop* findLoop(VMA vma);
  private:
    unordered_map<VMA, CFGFunc*> CFGFuncMap;
    unordered_map<VMA, CFGLoop*> CFGLoopMap;
  };

  extern BinaryAnalyzer binaryAnalyzer;
}

#endif /* BINARYANALYZER_HPP */
