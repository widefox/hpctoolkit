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

#ifndef HPCTOOLKIT_PROFILE_ACCUMULATORS_H
#define HPCTOOLKIT_PROFILE_ACCUMULATORS_H

#include <atomic>
#include <bitset>
#include "stdshim/optional.hpp"

namespace hpctoolkit {

/// Every Metric can have values at multiple Scopes pertaining to the subtree
/// rooted at a particular Context with Metric data.
enum MetricScope {
  /// Encapsulates the current Context, and no other nodes. This references
  /// exactly where the data arose, and is the smallest MetricScope.
  point,

  /// Encapsulates the current Context and any decendants not connected by a
  /// function-type Scope. This represents the cost of a function outside of
  /// any child function calls.
  /// Called "exclusive" in the Viewer.
  function,

  /// Encapsulates the current Context and all decendants. This represents
  /// the entire execution spawned by a single source code construct, and is
  /// the largest MetricScope.
  /// Called "inclusive" in the Viewer.
  execution,
};

/// Bitset-like object used as a set of Scope values.
class MetricScopeSet final : private std::bitset<3> {
private:
  using base = std::bitset<3>;
  MetricScopeSet(const base& b) : base(b) {};
public:
  MetricScopeSet() = default;
  MetricScopeSet(MetricScope s) : base(1<<s) {};

  bool has(MetricScope s) const noexcept { return base::operator[](s); }

  MetricScopeSet operator|(const MetricScopeSet& o) { return (base)*this | (base)o; }
  MetricScopeSet operator+(const MetricScopeSet& o) { return (base)*this | (base)o; }
  MetricScopeSet operator&(const MetricScopeSet& o) { return (base)*this & (base)o; }
  MetricScopeSet& operator|=(const MetricScopeSet& o) { base::operator|=(o); return *this; }
  MetricScopeSet& operator+=(const MetricScopeSet& o) { base::operator|=(o); return *this; }
  MetricScopeSet& operator&=(const MetricScopeSet& o) { base::operator&=(o); return *this; }

  using base::count;
};

/// Accumulator structure for the data implicitly bound to a Thread and Context.
class MetricAccumulator final {
public:
  MetricAccumulator() : point(0), function(0), execution(0) {};

  MetricAccumulator(const MetricAccumulator&) = delete;
  MetricAccumulator& operator=(const MetricAccumulator&) = delete;
  MetricAccumulator(MetricAccumulator&&) = default;
  MetricAccumulator& operator=(MetricAccumulator&&) = delete;

  /// Add some value to this Accumulator. Only point-Scope is allowed.
  // MT: Internally Synchronized
  void add(double) noexcept;

  /// Get the Thread-local (:Sum) Statistic accumulation, for a particular Metric Scope.
  // MT: Safe (const), Unstable (before ThreadFinal wavefront)
  stdshim::optional<double> get(MetricScope) const noexcept;

private:
  void validate() const noexcept;

  friend class Metric;
  std::atomic<double> point;
  double function;
  double execution;
};

/// Accumulator structure for the Statistics implicitly bound to a Context.
class StatisticAccumulator final {
public:
  StatisticAccumulator() : point(0), function(0), execution(0) {};

  StatisticAccumulator(const StatisticAccumulator&) = delete;
  StatisticAccumulator& operator=(const StatisticAccumulator&) = delete;
  StatisticAccumulator(StatisticAccumulator&&) = default;
  StatisticAccumulator& operator=(StatisticAccumulator&&) = delete;

  /// Add some Statistic value to this Accumulator, for the given Scope.
  // MT: Internally Synchronized
  void add(MetricScope, double) noexcept;

  /// Get the (:Sum) Statistic accumulation, for a particular Scope.
  // MT: Safe (const), Unstable (before `metrics` wavefront)
  stdshim::optional<double> get(MetricScope) const noexcept;

private:
  void validate() const noexcept;

  friend class Metric;
  // Currently only for :Sum Statistics
  std::atomic<double> point;
  std::atomic<double> function;
  std::atomic<double> execution;
};

}

#endif  // HPCTOOLKIT_PROFILE_ACCUMULATORS_H
