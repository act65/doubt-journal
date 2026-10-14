---
layout: paper-part
slug: inference-via-interpolation
part: thoughts
title: "Additional thoughts · Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference"
---

### Picking the plan length by likelihood

Since $n$ is a measurement rather than a knob, measure it. Composing $m$ links (using
$\alpha/(1-\alpha) = c$ to tidy the variance):

$$
p(\psi_g \mid \psi_0,\, m \text{ links}) \;=\; \mathcal{N}\!\left(\alpha^{m/2} O^m \psi_0,\;
\sigma_m^2 I\right), \qquad \sigma_m^2 = c\left(1 - \alpha^m\right)
$$

Sanity check at both ends: $\sigma_m^2 \to c$ as $m \to \infty$, i.e. the chain forgets $\psi_0$
and relaxes to the marginal from Assumption 1. So evaluate that log-density at $\psi_g$ for
$m = 1, 2, 3, \dots$, take $m^*$, and use $n = m^* - 1$ waypoints. As a by-product you get the
distance to the goal in real steps, $\approx m^* H \pm \sqrt{m^*}\sqrt\gamma/(1-\gamma)$, and a
sharper reachability test than the one-shot score — if even the best $m$ has a terrible
likelihood, $g$ is unreachable at any horizon.

The clean statement of the whole horizon story: *the representation measures distance to the goal
in units of the effective horizon, and that measurement is exactly the number of subgoals.* Which
is elegant, and also exactly why the plan cannot be refined below $H$.

### The implicit policy

$p_{t+}$ is carrying more than its notation admits. One correction first: it *is* conditional on
the start state — $p_{t+}(x_{t+} \mid x_0)$ has $x_0$ right there. What has been marginalised is
the lag (visibly, in Eq. 1) and **the policy** (invisibly, nowhere).

The setting is stated as a Markov process over states, with no actions and no policy anywhere. But
an MDP plus a behaviour policy $\beta$ *is* a Markov process — that is exactly how you get one.
So the transition kernel behind $p_t(x_t \mid x_0)$ is $\int p(x_t \mid x_{t-1}, a)\beta(a \mid
x_{t-1})\,da$, and every object in the paper is silently $\beta$-conditioned: $p_{t+}^\beta$,
$\psi^\beta$, $A^\beta$. The occupancy measure is a property of the data-collecting policy, not of
the environment.

What that implies:

- **The critic is an on-policy density ratio.** $\lVert A\psi(s) - \psi(g)\rVert^2$ measures
  reachability *under $\beta$*, not reachability under an optimal policy. If $\beta$ is random
  exploration, the induced geometry is the random walk's occupancy, not the environment's shortest
  path. Two states one wall apart but reachable by a long corridor will read as far apart or close
  together depending entirely on how often $\beta$ walked the corridor.
- **Waypoints predict rather than prescribe.** $p(\psi_w \mid \psi_0, \psi_g)$ answers "given that
  $\beta$ went from $s$ to $g$, where did it typically pass through?" It is a posterior over
  someone else's behaviour conditioned on a successful outcome — a plan only in the
  planning-as-inference sense.
- **Which imports the classic optimism bias.** Conditioning passive dynamics on a desired outcome
  assumes the lucky transitions happened. If reaching $g$ requires a favourable coin flip, the
  posterior conditions on it having come up favourably, and the inferred waypoint sits on the lucky
  branch. Harmless in deterministic dynamics; wrong in stochastic ones, in exactly the way the
  control-as-inference literature documents. Nothing in the construction defends against it.
- **Acting on the plans breaks the model.** If you execute these subgoals with a policy that is not
  $\beta$, the occupancy shifts and the learned $\psi, A$ no longer describe your data. Iterating
  plan → act → retrain is off-distribution in the standard offline-RL way.

The genuine upside, which is easy to miss: **no goal appears in training at all.** The goal enters
only at inference time, as the conditioning variable $\psi_g$. Train the representation once,
unconditionally, then condition on whatever goal you like. That is the real payoff of learning a
density ratio instead of a policy, and it is what makes the closed forms worth having.

The fix for the policy problem is known — action-conditioned representations $\psi(x, a)$, which is
what contrastive RL (Eysenbach et al., 2022) actually does, making the critic a genuine
goal-conditioned $Q$-function you can maximise over actions. This paper strips actions out to get
the clean Gaussian chain. Some of the tractability is bought by removing the control problem.

### What "inference via interpolation" does and doesn't claim

The title is doing some work it hasn't earned, in three separable ways:

1. **The direction is backwards.** "Inference *via* interpolation" reads as though interpolation
   is the method by which inference is performed. The logic runs the other way — the inference is
   Gaussian conditioning, which *reduces to* interpolation in a limit. "Inference as
   interpolation" would be accurate.
2. **Interpolation is a corollary, not the result.** It requires $A$ orthogonal and $c \to \infty$
   — which, per the consistency condition above, is also the only regime where the two assumptions
   are mutually compatible. Outside it you are back to a tridiagonal solve.
3. **The inference is over representations.** Every posterior is over $\psi$, in a learned, lossy,
   non-invertible space, on a chain whose time axis has been integrated away. "Planning and
   inference" invites a reader to expect posteriors over states at times, and neither is available.

### There are two clocks and nothing ties them

Per link, two unrelated things happen. The **lag clock** advances $H = \gamma/(1-\gamma)$ steps on
average, set by how positives were sampled. The **forgetting clock** contracts the posterior mean
by $\sqrt\alpha$, set by the regularisation budget $c$. These are independent hyperparameters — but
if both assumptions hold they cannot be. Chaining the resolvent reading
$A \approx (1-\gamma)(I - \gamma M)^{-1}$ with the consistency condition $AA^\top = \frac{c+1}{c}I$
says the singular values of the resolvent must all equal $\sqrt{(c+1)/c}$, which **pins $c$ as a
function of $\gamma$ and the true dynamics $M$**.

Sanity check at the degenerate end: static dynamics $M = I$ give $A = I$, requiring
$\sqrt{(c+1)/c} = 1$, i.e. $c \to \infty$. Correct — nothing moves, no contraction, so the variance
budget must be unbounded.

That yields a cheap empirical test of the entire setup: compare the forgetting rate implied by the
chosen $c$ against the mixing rate implied by $\gamma$ and the data. Disagreement should show up
as plans that collapse toward the origin (over-contraction) or never converge on the goal
(under-contraction).

### Multimodality is not representable

The posterior is Gaussian by construction and therefore unimodal by construction. Two corridors
around an obstacle put the mean in the wall. There is a partial defence — $\psi$ is nonlinear and
non-injective, so a unimodal blob in latent space can pull back to disconnected sets of states —
but the mean can still land where no state lives. Sampling instead of averaging mitigates it; the
model has no way to *represent* two distinct plans.

This connects back to Assumption 1 in a way worth noting: interpolation is only safe if every point
between $\psi_0$ and $\psi_g$ corresponds to some real state, which is precisely what a
full-support isotropic Gaussian marginal asserts. **Assumption 1's job in this paper is to license
interpolation.** If the true pushforward is atomic or supported on a low-dimensional manifold,
straight-line interpolation walks off the data and retrieval returns garbage. The assumption most
likely to be false is also the one holding up the headline result.

### Open questions

- [ ] Check the consistency condition $AA^\top = \frac{c+1}{c}I$ against their appendix — is it
      stated anywhere, and do the singular values of a trained $A$ actually cluster there?
- [ ] Verify the $m$-link composition against Lemma 3 before trusting the plan-length recipe.
- [ ] Does the uniformity term really maximise Shannon entropy, or only a Rényi-2 surrogate whose
      maximiser under a covariance constraint is *not* Gaussian? How much does the difference
      matter at realistic $k$?
- [ ] What does Assumption 1 degrade into for genuinely discrete state spaces — is there a
      smoothed-atoms version of Lemma 1 with quantifiable error?
- [ ] Empirically: does the negative-binomial lag spread show up as a characteristic failure mode
      (roughly 1 in 20 subgoals unreachable), and does resampling from the posterior fix it?
- [ ] An $A(\Delta t)$ variant trades homogeneity for a nameable horizon. Is the resulting system
      still tridiagonal in a useful way, and does it recover controllable plan durations?
- [ ] How badly does the optimism bias actually bite? Construct a stochastic gridworld where
      reaching $g$ needs a low-probability transition, and check whether the inferred waypoints sit
      on the lucky branch.
- [ ] How much does the learned geometry depend on $\beta$? Train on random exploration vs. an
      expert and compare the induced $\lVert A\psi(s) - \psi(g)\rVert$ against true shortest-path
      distance.
- [ ] Is there a version of Lemma 2 conditioned on a *set* of acceptable goals rather than a single
      $\psi_g$ — a Gaussian mixture on the endpoint would stay conjugate, so this looks cheap.
