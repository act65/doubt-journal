---
layout: paper-part
slug: cost-of-waiting
part: tutorial
title: "Tutorial · The Hidden Cost of Waiting for Accurate Predictions"
---

## Problem

The two questions the paper asks:

1. Ranking: How does the ranking loss change as the planner collects more data, but some individuals fail out of the population?
2. Allocation: For a given instance of this problem, what is the optimal time to allocate resources? When is early intervention justified?

<!-- what is the optimal time?
does it need to be;
- a single time for all,
- a time for each individual to get all the help
- can it be spaced out over time?

 -->

$$
W_t = \sum_{i \in A_t} u_t(p_i) · \mathbb{1} {r_t(i) \leq B}
$$

### The assumptions

Explicit Assumptions:

* The problem is set in a finite time horizon T. (Page 2)
* Each individual i has a fixed failure probability p_i per time step if not intervened upon. (Page 2)
* Failure events for different individuals are independent. (Page 2)
* Individuals who fail are removed from the active pool. (Page 2)
* The planner observes a binary signal o_i^t ~ Ber(p̃_i) from each active individual i at each time step t. (Page 2)
* The signal probability p̃_i is an increasing function of the failure probability p_i. (Page 3)
* The planner has a fixed budget B for resources, and each unit of resource has a fixed cost. (Page 3)
* The utility u^t(p) from intervention is non-increasing in time t and non-decreasing in failure probability p. (Page 3)
* The utility function u^t(p) is concave in p, and u^t(0) = 0. (Page 3)
* For the "fully effective treatment" example (Example 2.1), intervention prevents any future failure. (Page 3)
* For Theorem 3.1, the inverse of p̃(·) is O(1)-Lipschitz. (Page 4)
* Assumption 4.2 specifies a particular observation model: p̃ = 1 - (1-p)^γ, with γ > 1. (Page 6)
* For Theorem 4.3, the initial distribution of failure probabilities is G-decaying. (Page 6)

Implicit Assumptions:

* The planner's objective is to maximize the total expected utility from interventions. (Page 1, 3)
* Resources allocated can prevent or mitigate the "undesirable outcomes."
* The model parameters (like p_i, p̃_i, utility functions) are known or can be accurately estimated by the planner.
* The number of individuals (N) is often assumed to be large, especially for results "in the limit of many individuals." (Page 7)
* Time progresses in discrete steps.
* The population is closed (no new individuals enter during the T steps, though it shrinks due to failures).
* Observation of signals is costless and occurs for all active individuals.
* Intervention is instantaneous once decided.
* Primarily, there are no spillover effects from interventions on one individual to another (unless specified otherwise). (Page 3)
