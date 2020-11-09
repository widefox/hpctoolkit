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

#include "metric.hpp"

#include "context.hpp"
#include "attributes.hpp"

#include <stack>
#include <thread>
#include <ostream>

using namespace hpctoolkit;

static double atomic_add(std::atomic<double>& a, const double v) noexcept {
  double old = a.load(std::memory_order_relaxed);
  while(!a.compare_exchange_weak(old, old+v, std::memory_order_relaxed));
  return old;
}

static double atomic_op(std::atomic<double>& a, const double v, Statistic::combination_t op) noexcept {
  double old = a.load(std::memory_order_relaxed);
  switch(op) {
  case Statistic::combination_t::sum:
    while(!a.compare_exchange_weak(old, old+v, std::memory_order_relaxed));
    break;
  case Statistic::combination_t::min:
    while((v < old || old == 0) && !a.compare_exchange_weak(old, v, std::memory_order_relaxed));
    break;
  case Statistic::combination_t::max:
    while((v > old || old == 0) && !a.compare_exchange_weak(old, v, std::memory_order_relaxed));
    break;
  }
  return old;
}

unsigned int Metric::ScopedIdentifiers::get(MetricScope s) const noexcept {
  switch(s) {
  case MetricScope::point: return point;
  case MetricScope::function: return function;
  case MetricScope::execution: return execution;
  }
  util::log::fatal{} << "Invalid Metric::scope value!";
  std::abort();  // unreachable
}

Statistic::Statistic(std::string suff, bool showp, formula_t form)
  : m_suffix(std::move(suff)), m_showPerc(showp), m_formula(std::move(form)) {};

Metric::Metric(Metric&& m)
  : userdata(std::move(m.userdata), std::cref(*this)),
    u_settings(std::move(m.u_settings)),
    m_partials(std::move(m.m_partials)),
    m_stats(std::move(m.m_stats)) {};

Metric::Metric(ud_t::struct_t& rs, Settings s)
  : userdata(rs, std::cref(*this)), u_settings(std::move(s)) {
  size_t cntIdx = -1;
  if(s.mean || s.stddev || s.cfvar) {
    cntIdx = m_partials.size();
    m_partials.push_back({[](double x) -> double { return x == 0 ? 0 : 1; },
                          Statistic::combination_t::sum, cntIdx});
  }
  size_t xIdx = -1;
  if(s.sum || s.mean || s.stddev || s.cfvar) {
    xIdx = m_partials.size();
    m_partials.push_back({[](double x) -> double { return x; },
                          Statistic::combination_t::sum, xIdx});
  }
  size_t x2Idx = -1;
  if(s.stddev || s.cfvar) {
    x2Idx = m_partials.size();
    m_partials.push_back({[](double x) -> double { return x * x; },
                          Statistic::combination_t::sum, x2Idx});
  }
  size_t minIdx = -1;
  if(s.min) {
    minIdx = m_partials.size();
    m_partials.push_back({[](double x) -> double { return x; },
                          Statistic::combination_t::min, minIdx});
  }
  size_t maxIdx = -1;
  if(s.max) {
    maxIdx = m_partials.size();
    m_partials.push_back({[](double x) -> double { return x; },
                          Statistic::combination_t::max, maxIdx});
  }

  if(s.sum)
    m_stats.push_back({"Sum", true, {(Statistic::formula_t::value_type)xIdx} });
  if(s.mean)
    m_stats.push_back({"Mean", false, {xIdx, "/", cntIdx} });
  if(s.stddev)
    m_stats.push_back({"StdDev", false,
      {"sqrt((", x2Idx, "/", cntIdx, ") - pow(", xIdx, "/", cntIdx, ", 2))"} });
  if(s.cfvar)
    m_stats.push_back({"CfVar", false,
      {"sqrt((", x2Idx, "/", cntIdx, ") - pow(", xIdx, "/", cntIdx, ", 2)) / (", xIdx, "/", cntIdx, ")"} });
  if(s.min)
    m_stats.push_back({"Min", false, {(Statistic::formula_t::value_type)minIdx} });
  if(s.max)
    m_stats.push_back({"Max", false, {(Statistic::formula_t::value_type)maxIdx} });
}

MetricScopeSet Metric::scopes() const noexcept {
  // For now, its always point/function/execution
  return MetricScopeSet(MetricScope::point) +
    MetricScopeSet(MetricScope::function) + MetricScopeSet(MetricScope::execution);
}

const std::vector<StatisticPartial>& Metric::partials() const noexcept {
  return m_partials;
}
const std::vector<Statistic>& Metric::statistics() const noexcept {
  return m_stats;
}

StatisticAccumulator::StatisticAccumulator(const Metric& m)
  : partials(m.partials().size()) {};

void StatisticAccumulator::PartialRef::add(MetricScope s, double v) noexcept {
  if(v == 0) util::log::warning{} << "Adding a 0-metric value!";
  switch(s) {
  case MetricScope::point: atomic_op(partial.point, v, statpart.combinator()); return;
  case MetricScope::function: atomic_op(partial.function, v, statpart.combinator()); return;
  case MetricScope::execution: atomic_op(partial.execution, v, statpart.combinator()); return;
  }
  util::log::fatal{} << "Invalid MetricScope!";
}

StatisticAccumulator::PartialCRef StatisticAccumulator::get(const StatisticPartial& p) const noexcept {
  return {partials[p.m_idx], p};
}
StatisticAccumulator::PartialRef StatisticAccumulator::get(const StatisticPartial& p) noexcept {
  return {partials[p.m_idx], p};
}

void MetricAccumulator::add(double v) noexcept {
  if(v == 0) util::log::warning{} << "Adding a 0-metric value!";
  atomic_add(point, v);
}

static stdshim::optional<double> opt0(double d) {
  return d == 0 ? stdshim::optional<double>{} : d;
}

stdshim::optional<double> StatisticAccumulator::PartialRef::get(MetricScope s) const noexcept {
  partial.validate();
  switch(s) {
  case MetricScope::point: return opt0(partial.point.load(std::memory_order_relaxed));
  case MetricScope::function: return opt0(partial.function.load(std::memory_order_relaxed));
  case MetricScope::execution: return opt0(partial.execution.load(std::memory_order_relaxed));
  };
  util::log::fatal{} << "Invalid MetricScope value!";
  std::abort();  // unreachable
}
stdshim::optional<double> StatisticAccumulator::PartialCRef::get(MetricScope s) const noexcept {
  partial.validate();
  switch(s) {
  case MetricScope::point: return opt0(partial.point.load(std::memory_order_relaxed));
  case MetricScope::function: return opt0(partial.function.load(std::memory_order_relaxed));
  case MetricScope::execution: return opt0(partial.execution.load(std::memory_order_relaxed));
  };
  util::log::fatal{} << "Invalid MetricScope value!";
  std::abort();  // unreachable
}

void StatisticAccumulator::Partial::validate() const noexcept {
  if(point.load(std::memory_order_relaxed) != 0) return;
  if(function.load(std::memory_order_relaxed) != 0) return;
  if(execution.load(std::memory_order_relaxed) != 0) return;
  util::log::warning{} << "Returning a Statistic accumulator with no value!";
}

const StatisticAccumulator* Metric::getFor(const Context& c) const noexcept {
  return c.data.find(this);
}

stdshim::optional<double> MetricAccumulator::get(MetricScope s) const noexcept {
  validate();
  switch(s) {
  case MetricScope::point: return opt0(point.load(std::memory_order_relaxed));
  case MetricScope::function: return opt0(function);
  case MetricScope::execution: return opt0(execution);
  }
  util::log::fatal{} << "Invalid MetricScope value!";
  std::abort();  // unreachable
}

void MetricAccumulator::validate() const noexcept {
  if(point.load(std::memory_order_relaxed) != 0) return;
  if(function != 0) return;
  if(execution != 0) return;
  util::log::warning{} << "Returning a Metric accumulator with no value!";
}

const MetricAccumulator* Metric::getFor(const Thread::Temporary& t, const Context& c) const noexcept {
  auto* cd = t.data.find(&c);
  if(cd == nullptr) return nullptr;
  return cd->find(this);
}

static bool pullsFunction(const Context& parent, const Context& child) {
  switch(child.scope().type()) {
  // Function-type Scopes, and unknown (which could be a function)
  case Scope::Type::function:
  case Scope::Type::inlined_function:
  case Scope::Type::unknown:
    return false;
  case Scope::Type::point:
  case Scope::Type::loop:
    switch(parent.scope().type()) {
    // Function-type scopes, and unknown (which could be a function)
    case Scope::Type::function:
    case Scope::Type::inlined_function:
    case Scope::Type::loop:
    case Scope::Type::unknown:
      return true;
    case Scope::Type::point:
    case Scope::Type::global:
      return false;
    }
    break;
  case Scope::Type::global:
    util::log::fatal{} << "Operation invalid for the global Context!";
    break;
  }
  std::abort();  // unreachable
}

void Metric::finalize(Thread::Temporary& t) noexcept {
  // For each Context we need to know what its children are. But we only care
  // about ones that have decendants with actual data. So we construct a
  // temporary subtree with all the bits.
  const Context* global = nullptr;
  std::unordered_map<const Context*, std::unordered_set<const Context*>> children;
  {
    std::vector<const Context*> newContexts;
    newContexts.reserve(t.data.size());
    for(const auto& cx: t.data.citerate()) newContexts.emplace_back(cx.first);
    while(!newContexts.empty()) {
      decltype(newContexts) next;
      next.reserve(newContexts.size());
      for(const Context* cp: newContexts) {
        if(!cp->direct_parent()) {
          if(global != nullptr) util::log::fatal{} << "Multiple root contexts???";
          global = cp;
          continue;
        }
        auto x = children.emplace(cp->direct_parent(),
                                  std::unordered_set<const Context*>{});
        if(x.second) next.push_back(cp->direct_parent());
        x.first->second.emplace(cp);
      }
      next.shrink_to_fit();
      newContexts = std::move(next);
    }
  }
  if(global == nullptr) return;  // Apparently there's nothing to propagate

  // Now that the critical subtree is built, recursively propagate up.
  using md_t = util::locked_unordered_map<const Metric*, MetricAccumulator>;
  struct frame_t {
    frame_t(const Context& c) : ctx(c) {};
    frame_t(const Context& c, std::unordered_set<const Context*>& v)
      : ctx(c), here(v.cbegin()), end(v.cend()) {};
    const Context& ctx;
    std::unordered_set<const Context*>::const_iterator here;
    std::unordered_set<const Context*>::const_iterator end;
    std::vector<std::pair<std::reference_wrapper<const Context>,
                          std::reference_wrapper<const md_t>>> submds;
  };
  std::stack<frame_t, std::vector<frame_t>> stack;

  // Post-order in-memory tree traversal
  stack.emplace(*global, children.at(global));
  while(!stack.empty()) {
    if(stack.top().here != stack.top().end) {
      // This frame still has children to handle
      const Context* c = *stack.top().here;
      auto ccit = children.find(c);
      ++stack.top().here;
      if(ccit == children.end()) stack.emplace(*c);
      else stack.emplace(*c, ccit->second);
      continue;  // We'll come back eventually
    }

    const Context& c = stack.top().ctx;
    md_t& data = t.data[&c];
    // Handle the internal propagation first, so we don't get mixed up.
    for(auto& mx: data.iterate()) {
      mx.second.execution = mx.second.function
                          = mx.second.point.load(std::memory_order_relaxed);
    }

    // Go through our children and sum into our bits
    for(std::size_t i = 0; i < stack.top().submds.size(); i++) {
      const Context& cc = stack.top().submds[i].first;
      const md_t& ccmd = stack.top().submds[i].second;
      const bool pullfunc = pullsFunction(c, cc);
      for(const auto& mx: ccmd.citerate()) {
        auto& accum = data[mx.first];
        if(pullfunc) accum.function += mx.second.function;
        accum.execution += mx.second.execution;
      }
    }

    // Now that our bits are stable, accumulate back into the Statistics
    auto& cdata = const_cast<Context&>(c).data;
    for(const auto& mx: data.citerate()) {
      auto& accum = cdata.emplace(std::piecewise_construct,
        std::forward_as_tuple(mx.first), std::forward_as_tuple(*mx.first)).first;
      for(size_t i = 0; i < mx.first->partials().size(); i++) {
        auto& partial = mx.first->partials()[i];
        auto& atomics = accum.partials[i];
        atomic_op(atomics.point, partial.m_accum(mx.second.point.load(std::memory_order_relaxed)), partial.combinator());
        atomic_op(atomics.function, partial.m_accum(mx.second.function), partial.combinator());
        atomic_op(atomics.execution, partial.m_accum(mx.second.execution), partial.combinator());
      }
    }

    stack.pop();
    if(!stack.empty()) stack.top().submds.emplace_back(c, data);
  }
}

std::size_t std::hash<Metric::Settings>::operator()(const Metric::Settings &s) const noexcept {
  const auto h1 = std::hash<std::string>{}(s.name);
  const auto h2 = std::hash<std::string>{}(s.description);
  return h1 ^ ((h2 << 1) | (h2 >> (-1 + 8 * sizeof h2)));
}
