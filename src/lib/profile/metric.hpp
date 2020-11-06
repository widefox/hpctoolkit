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

#ifndef HPCTOOLKIT_PROFILE_METRIC_H
#define HPCTOOLKIT_PROFILE_METRIC_H

#include "accumulators.hpp"
#include "attributes.hpp"

#include "util/atomic_unordered.hpp"
#include "util/locked_unordered.hpp"
#include "util/uniqable.hpp"
#include "util/ragged_vector.hpp"

#include <atomic>
#include <bitset>
#include <functional>
#include "stdshim/optional.hpp"
#include <variant>
#include <vector>

namespace hpctoolkit {

class Context;

class Metric;
class StatisticPartial;

/// A Statistic represents a combination of Metric data across all Threads,
/// on a per-Context basis. These are generated via three formulas:
///  - "accumulate": converts the thread-local value into an accumulation,
///  - "combine": combines two accumulations into a valid accumulation,
///  - "finalize": converts an accumulation into a presentable final value.
/// In total, this allows for condensed information regarding the entire
/// profiled execution, while still permitting inspection of individual Threads.
class Statistic final {
public:
  Statistic(const Statistic&) = delete;
  Statistic(Statistic&&) = default;
  Statistic& operator=(const Statistic&) = delete;
  Statistic& operator=(Statistic&&) = default;

  // Only a few combination formulas are permitted. This is the set.
  enum class combination_t { sum, min, max };

  // The other two formulas are best represented by C++ functions.
  using accumulate_t = std::function<double(double)>;
  using finalize_t = std::function<double(const std::vector<double>&)>;

  // Standard Statistics are hard-coded based on the following enumeration.
  enum class standard_t {
    sum,  // Sum of thread-local values
    mean,  // Average of non-zero thread-local values
  };

  /// Statistics are created by the associated Metric.
  Statistic() = delete;

  /// Get the additional suffix associated with this Statistic.
  /// E.g. "Sum" or "Avg".
  // MT: Safe (const)
  const std::string& suffix() const noexcept { return m_suffix; }

  /// Get whether the percentage should be shown for this Statistic.
  /// TODO: Figure out what property this indicates mathematically
  // MT: Safe (const)
  bool showPercent() const noexcept { return m_showPerc; }

  /// Type for formulas. Each element is either a string or the index of a Partial.
  /// If all such indices are replaced by variable names and the entire vector
  /// concatinated, the result is a C-like math formula.
  using formula_t = std::vector<std::variant<size_t, std::string>>;

  /// Get the formula used generate the final value for this Statistic.
  // MT: Safe (const)
  const formula_t& finalizeFormula() const noexcept { return m_formula; }

private:
  const std::string m_suffix;
  const bool m_showPerc;
  const formula_t m_formula;

  friend class Metric;
  Statistic(std::string, bool, formula_t);
};

/// A StatisticPartial is the "accumulate" and "combine" parts of a Statistic.
/// There may be multiple Partials used for a Statistic, and multiple Statistics
/// can share the same Partial.
class StatisticPartial final {
public:
  StatisticPartial(const StatisticPartial&) = delete;
  StatisticPartial(StatisticPartial&&) = default;
  StatisticPartial& operator=(const StatisticPartial&) = delete;
  StatisticPartial& operator=(StatisticPartial&&) = default;

  /// Get the combination function used for this Partial
  // MT: Safe (const)
  Statistic::combination_t combinator() const noexcept { return m_combin; }

private:
  const Statistic::accumulate_t m_accum;
  const Statistic::combination_t m_combin;
  const std::size_t m_idx;

  friend class Metric;
  friend class StatisticAccumulator;
  StatisticPartial() = default;
  StatisticPartial(Statistic::accumulate_t a, Statistic::combination_t c, std::size_t idx)
    : m_accum(std::move(a)), m_combin(std::move(c)), m_idx(idx) {};
};

// Just a simple metric class, nothing to see here
class Metric final {
public:
  using ud_t = util::ragged_vector<const Metric&>;

  /// Set of identifiers unique to each Metric Scope that a Metric may have.
  struct ScopedIdentifiers final {
    unsigned int point;
    unsigned int function;
    unsigned int execution;
    unsigned int get(MetricScope s) const noexcept;
  };

  /// Structure to be used for creating new Metrics. Encapsulates a number of
  /// smaller settings into a convienent structure.
  struct Settings final {
    std::string name;
    std::string description;

    bool operator==(const Settings& o) const noexcept {
      return name == o.name && description == o.description;
    }
  };

  Metric(ud_t::struct_t&, Settings);
  Metric(Metric&& m);

  const std::string& name() const { return u_settings().name; }
  const std::string& description() const { return u_settings().description; }

  mutable ud_t userdata;

  /// Get the set of Scopes that this Metric supports.
  MetricScopeSet scopes() const noexcept;

  /// List the StatisticPartials that are included in this Metric.
  // MT: Safe (const)
  const std::vector<StatisticPartial>& partials() const noexcept;

  /// List the Statistics that are included in this Metric.
  // MT: Safe (const)
  const std::vector<Statistic>& statistics() const noexcept;

  /// Obtain a pointer to the Statistic Accumulators for a particular Context.
  /// Returns `nullptr` if no Statistic data exists for the given Context.
  // MT: Safe (const), Unstable (before `metrics` wavefront)
  const StatisticAccumulator* getFor(const Context& c) const noexcept;

  /// Obtain a pointer to the Thread-local Accumulator for a particular Context.
  /// Returns `nullptr` if no metric data exists for the given Context.
  // MT: Safe (const), Unstable (before notifyThreadFinal)
  const MetricAccumulator* getFor(const Thread::Temporary&, const Context& c) const noexcept;

private:
  util::uniqable_key<Settings> u_settings;
  std::vector<StatisticPartial> m_partials;
  std::vector<Statistic> m_stats;

  friend class ProfilePipeline;
  // Finalize the MetricAccumulators for a Thread.
  // MT: Internally Synchronized
  static void finalize(Thread::Temporary& t) noexcept;

  friend class util::uniqued<Metric>;
  util::uniqable_key<Settings>& uniqable_key() { return u_settings; }
};

}

namespace std {
  using namespace hpctoolkit;
  template<> struct hash<Metric::Settings> {
    std::size_t operator()(const Metric::Settings&) const noexcept;
  };
}

#endif  // HPCTOOLKIT_PROFILE_METRIC_H
