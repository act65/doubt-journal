---
layout: paper-part
slug: inference-via-interpolation
part: tutorial
title: "Tutorial · Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference"
---

## Problem

Probabilistic inference over time series is expensive in high dimensions. Given where you are now
and where you want to end up, questions like *what states lie between them?* or *how likely is
that endpoint at all?* normally need a generative model of observations plus search over it —
rollouts, beam search, sampling. Both parts scale badly, and the generative model has to spend
capacity modelling detail that has nothing to do with where the trajectory goes.

The paper's claim is that if you learn representations with one particular contrastive objective,
these inference questions stop needing search. They acquire closed-form answers that cost a
$k \times k$ matrix inverse — and in a limiting case, no inverse at all, just linear interpolation
between the encoded start and the encoded goal.

The setting is a Markov process with states $x_0, x_1, x_2, \dots$. The central object is the
$\gamma$-discounted state occupancy measure (their Eq. 1),

$$
p_{t+}(x_{t+} = x \mid x_0) \;=\; (1-\gamma) \sum_{t=0}^{\infty} \gamma^{t}\, p_t(x_t = x \mid x_0),
$$

which is what "a future state" means throughout — not the next state, and not the state at any
fixed offset, but a state at a geometrically-distributed random lag. Positive pairs for the
contrastive loss are drawn from it; negatives come from the marginal $p(x)$.

**Read "$t+$" as a single symbol, not as "$t$ plus something".** It is a *label* meaning "at some
future time", and $x_{t+}$ is one random variable — the state at a random future time. There is no
free $t$: the $t$ on the right-hand side is a summation index, bound by the sum and gone by the
time you reach the left-hand side. So $p_{t+}(x_{t+} = x \mid x_0)$ has exactly two arguments, $x$
and $x_0$, and the time offset has already been integrated away. The notation is genuinely
misleading — it looks indexed by a time that isn't there — and much of the horizon awkwardness
below is the consequence of that integration being invisible in the symbols.

Note also what this object is *not* conditioned on. There are no actions anywhere in the paper, so
the transition kernel behind $p_t(x_t \mid x_0)$ is that of a Markov process with a **fixed
behaviour policy already baked in**. Everything that follows is implicitly conditioned on that
policy; see [the implicit policy](#the-implicit-policy) below.

The two assumptions the whole paper rests on are stated in §3.1 with almost no justification, and
they are the reason this entry exists. Assumption 2 is a known theorem being imported, not an
assumption in the usual sense. Assumption 1 is a genuine assumption and is doing more work than
the paper admits.

## Overview

### Notation

| paper | here | what it is |
|---|---|---|
| $\psi(x)$ | the encoder | maps a state to $\mathbb{R}^k$; the only network in the construction |
| $\phi(x) \triangleq A\psi(x)$ | "the other tower" | not a second encoder — the same $\psi$ with a learned $k \times k$ matrix in front |
| $x_{t+}$ | a future state | drawn from the discounted occupancy, *not* at a fixed offset |
| $c$ | the variance budget | the regularisation constant in $\tfrac{1}{k}\mathbb{E}\lVert\psi\rVert^2 \le c$ |
| — | $\alpha \triangleq c/(c+1)$ | the per-link contraction factor; every result below is cleaner in terms of it |
| — | $H \triangleq \gamma/(1-\gamma)$ | effective horizon: mean number of real time steps per "link" |
| — | link | one application of the occupancy kernel — one edge of the chain |
| — | waypoint | an *intermediate* node; $n$ waypoints means $m = n+1$ links |

### The objective, written out

There is one encoder $\psi$ and one matrix $A$, with $\phi(x) \triangleq A\psi(x)$. Two choices
distinguish this from a standard contrastive setup, and both are load-bearing:

**The critic is a Gaussian kernel, not an inner product.**

$$
f(x, y) \;=\; -\tfrac{1}{2}\lVert A\psi(x) - \psi(y) \rVert_2^2
$$

The $\tfrac12\lVert\cdot\rVert^2$ is not cosmetic — it is the log of an unnormalised Gaussian
density, which is the entire reason a Gaussian falls out later. Expanding,
$f = \langle A\psi(x), \psi(y)\rangle - \tfrac12\lVert A\psi(x)\rVert^2 - \tfrac12\lVert\psi(y)\rVert^2$,
so it *is* the usual bilinear critic — it just keeps the norm terms instead of normalising them
away, and the retained $\lVert\psi(y)\rVert^2$ is what multiplies against the Gaussian prior in
Lemma 1.

**The regulariser is a variance budget, not a unit sphere** (their Eq. 3):

$$
\tfrac{1}{k}\,\mathbb{E}_{p(x)}\!\left[\lVert\psi(x)\rVert_2^2\right] \;\le\; c
$$

Every vision paper normalises to $\lVert\psi\rVert = 1$, and max-entropy on a sphere gives uniform
on a sphere. Swapping to a trace budget is what makes the max-entropy answer a Gaussian instead.
The choice is reverse-engineered to make the marginal Gaussian.

The objective itself is symmetrised InfoNCE, maximised over $\psi$ and $A$:

$$
\mathbb{E}\!\left[\, f(x, x^+) \;-\; \log \sum_{k} e^{f(x,\, y_k)} \right],
\qquad x^+ \sim p_{t+}(\cdot \mid x),\;\; y^- \sim p(x)
$$

The denominator is a **sum of exponentials over the candidates**, and the negatives are drawn from
the state marginal. In a batch implementation they are the other rows' positives, which is
legitimate because a positive marginalised over random anchors is distributed as $p(x)$.

### Why InfoNCE recovers a density ratio (Assumption 2)

Assumption 2 says

$$
e^{-\frac{1}{2}\lVert A\psi(x_0) - \psi(x)\rVert^2} \;=\; \frac{p_{t+}(x_{t+} = x \mid x_0)}{p(x)\, C}
$$

(Again, $t+$ is a label, not a free index — the lag was integrated out in Eq. 1. Both sides are
functions of $x$ and $x_0$ only.)

This looks like a conclusion. It is — this is the classical InfoNCE optimum, and the way to stop
being confused by it is to see that **InfoNCE is a classification problem whose Bayes-optimal
answer is a softmax over log density ratios.**

Set up the game. Draw $x \sim p(x)$, draw a hidden index $I \sim \mathrm{Unif}\{1,\dots,B\}$, then
draw

$$
y_I \sim p(y \mid x) \quad\text{(the positive, at a secret slot)},
\qquad y_j \sim p(y) \;\; \text{for } j \ne I \quad\text{(the negatives)}
$$

Hand over $(x, y_1, \dots, y_B)$ and ask: **which slot holds the positive?** This has an exact
Bayes-optimal answer, and deriving it is worth doing slowly, because the answer *is* the softmax
that InfoNCE fits.

Write the joint density of the index and all $B$ candidates. Given $x$, the index is uniform, slot
$i$ was drawn from the conditional, and every other slot was drawn from the marginal:

$$
P(I = i,\, y_{1:B} \mid x) \;=\; \underbrace{\tfrac{1}{B}}_{P(I=i)} \cdot
\underbrace{p(y_i \mid x)}_{\text{the positive}} \cdot
\underbrace{\textstyle\prod_{j \ne i} p(y_j)}_{\text{the negatives}}
$$

Bayes' rule for the posterior over $i$ is just this, normalised over $i$ (the $\tfrac1B$ cancels
immediately, being the same for every term):

$$
P(I = i \mid x, y_{1:B}) \;=\;
\frac{p(y_i \mid x) \prod_{j \ne i} p(y_j)}{\sum_{k} p(y_k \mid x) \prod_{j \ne k} p(y_j)}
$$

Now the trick. Every term in that expression — numerator and every summand of the denominator —
contains $B$ factors, one per slot: the conditional at its own slot and the marginal everywhere
else. So multiply top and bottom by $1 = \big(\prod_{\text{all } j} p(y_j)\big)^{-1}$, which turns
each term's $\prod_{j \ne i} p(y_j)$ into $1/p(y_i)$:

$$
\frac{p(y_i \mid x) \prod_{j \ne i} p(y_j)}{\prod_{\text{all } j} p(y_j)}
\;=\; \frac{p(y_i \mid x)}{p(y_i)} \;=\; r(x, y_i)
$$

**Every nuisance marginal cancels except the one at the slot being scored.** Applying that to
numerator and denominator alike:

$$
P(I = i \mid x, y_{1:B}) \;=\; \frac{r(x, y_i)}{\sum_k r(x, y_k)},
\qquad r(x,y) \triangleq \frac{p(y \mid x)}{p(y)}
$$

Exactly, by cancellation — no approximation anywhere. The point to hold onto: **the Bayes-optimal
classifier for this game is a softmax whose logits are log density ratios.** That is not a
modelling choice, it is forced by the generative process, which was rigged so that the marginals
divide out. So when you fit a softmax to this game, you are fitting logits to $\log r$ — and the
density ratio in Assumption 2 is what the optimum has to be.

Now posit a softmax model with a learned critic as logits,
$q_f(I = i \mid \cdot) = e^{f(x, y_i)} / \sum_k e^{f(x, y_k)}$, and maximise the expected
log-likelihood of the true index. By exchangeability that objective *is* InfoNCE, and decomposing
it gives

$$
\mathbb{E}[\log q_f] \;=\; -\,\mathbb{E}\!\left[H\big(P(I \mid \cdot)\big)\right]
\;-\; \mathbb{E}\!\left[\mathrm{KL}\big(P(I \mid \cdot)\;\Vert\; q_f(I \mid \cdot)\big)\right]
$$

The entropy term does not depend on $f$. So **maximising InfoNCE is minimising a forward KL from
the Bayes posterior to your softmax model**, batch by batch — unique optimum at $q_f = P$.
Matching for every batch forces $e^{f(x,y_i)} = \lambda(\cdot)\, r(x, y_i)$; taking $B = 2$ and
varying the two slots gives $f(x,y) - \log r(x,y)$ constant in $y$, hence

$$
f^*(x, y) \;=\; \log \frac{p(y \mid x)}{p(y)} \;+\; c(x)
$$

This holds for **any $B \ge 2$** at the population optimum — the ratio argument only ever used two
slots. What large $B$ buys is a different thing: the *value* of the objective is capped at
$\log B$, which is why InfoNCE is a bad mutual-information estimator on small batches. The
optimal critic is the log ratio regardless.

The **symmetrisation** is what kills $c(x)$. Normalising over the other axis of the $B \times B$
logit matrix gives $f^*(x,y) = \log\frac{p(x\mid y)}{p(x)} + c'(y)$, and since the log ratio is
the same symmetric PMI either way, $c(x) = c'(y)$ for all pairs — so both are one global constant,
the $C$ in Assumption 2.

So what is actually being assumed? Three things, only one of which is interesting:

1. The global optimum is reached.
2. The norm constraint (Eq. 3) is inactive — otherwise you are at a *constrained* optimum, not
   this one.
3. **Realisability.** The critic can only express
   $\langle A\psi(x), \psi(y)\rangle + a(x) + b(y)$ — a rank-$k$ bilinear form plus row and column
   offsets. Assumption 2 asserts that the true log density ratio *has that structure*. This is a
   low-rank assumption on the occupancy measure's PMI, and it is the same assumption as low-rank
   MDPs and successor features. It is the entire content.

### Where the Gaussian marginal comes from (Assumption 1)

Assumption 1 says the pushforward of $p(x)$ through $\psi$ is $\mathcal{N}(0, cI)$. The
justification chain is:

- the denominator of InfoNCE is a *uniformity* pressure — it pushes representations of random
  states apart, i.e. it maximises the entropy of the marginal;
- among densities on $\mathbb{R}^k$ with $\mathbb{E}\lVert\psi\rVert^2 \le kc$, the Shannon
  max-entropy density is $\mathcal{N}(0, cI)$ — zero-mean because the budget penalises the mean,
  isotropic because only the trace is constrained and $\log\det\Sigma$ at fixed trace is maximised
  at equal eigenvalues.

The second step is standard information theory. The first is where the doubt belongs, for three
reasons.

**The constraint gives much less than the assumption takes.** Eq. 3 bounds the *trace* of the
second moment matrix. It says nothing about the mean, nothing about isotropy, nothing about shape,
and it is an *inequality*. Everything from "trace bound" to "isotropic Gaussian density" is
supplied by the max-entropy argument, not by the regulariser. It is worth separating two claims
that are easy to conflate:

| | statement | what supplies it |
|---|---|---|
| (a) moment condition | $\mathbb{E}[\psi] = 0$, $\mathbb{E}[\psi\psi^\top] = cI$ | roughly, the regulariser |
| (b) distributional | the pushforward $\psi_\# p$ *is* $\mathcal{N}(0, cI)$ | Assumption 1 |

The proofs need (b). Lemma 1 multiplies $p(\psi)$ against a Gaussian likelihood and completes the
square; moment matching gives nothing there, since infinitely many densities have mean $0$ and
covariance $cI$ and almost none of them are conjugate to a Gaussian.

**The mechanism and the theorem are not the same objective.** The uniformity term is driving down
$\mathbb{E}_{\psi,\psi'}\big[e^{-\frac12\lVert\psi - \psi'\rVert^2}\big]$ — a Gaussian-kernel
information potential, which is a Rényi-2-flavoured quantity ($\approx \int p^2$ for a narrow
kernel), not Shannon entropy. Maximum-Rényi-$\alpha$ distributions under a covariance constraint
are not Gaussian for $\alpha \ne 1$. So the pressure that plausibly does the work and the
max-entropy result that yields the Gaussian are not obviously the same thing.

**It is literally false for discrete data.** If $p(x)$ is a finite state space, or just a finite
dataset, then $\psi_\# p$ is a sum of atoms and is never a continuous density. The defensible
reading is "the atoms are arranged so that anything you smooth or integrate looks Gaussian", which
is fine as an idealisation but means the Lemmas describe a limiting object, not the $\psi$ you
trained. The paper's limitations section concedes exactly this and no more: *"it remains open how
errors in those approximations translate into errors in our analysis."*

### The two assumptions are the two halves of Bayes

This is the calculation that shows why both are needed and what each one does. Write the
conditional as marginal times ratio:

$$
p(\psi_{t+} = \psi \mid \psi_0) \;=\; p(\psi)\cdot\frac{p(\psi \mid \psi_0)}{p(\psi)}
$$

Because Assumption 2's right-hand side depends on $x$ only through $\psi(x)$, the ratio survives
the pushforward unchanged. So

$$
p(\psi \mid \psi_0) \;\propto\; \underbrace{\mathcal{N}(\psi;\, 0,\, cI)}_{\text{Assumption 1 — the prior}}
\;\cdot\; \underbrace{\exp\!\left(-\tfrac{1}{2}\lVert \psi - A\psi_0 \rVert^2\right)}_{\text{Assumption 2 — the likelihood}}
$$

This is the cleanest way to hold the whole paper in mind — the two assumptions are not two
unrelated technical conditions, they are **one conjugate Bayesian update**, and each supplies
exactly one factor:

| assumption | what it really says | role in Bayes |
|---|---|---|
| 1 | the marginal $p(\psi)$ is $\mathcal{N}(0, cI)$ | **the prior** |
| 2 | the critic is the log density ratio, and the critic is a squared distance | **the likelihood**, Gaussian in $\psi$ |

Gaussian prior $\times$ Gaussian likelihood $=$ Gaussian posterior. Every design choice in §3 —
the squared-distance critic instead of an inner product, the variance budget instead of a unit
sphere — exists to make those two factors conjugate. Once you see that, the rest of the paper is
bookkeeping.

A product of two Gaussians in $\psi$. Precision $(1/c + 1)I$, so completing the square gives
Lemma 1:

$$
p(\psi_{t+} \mid \psi_0) \;=\; \mathcal{N}\!\left(\alpha A \psi_0,\; \alpha I\right),
\qquad \alpha = \frac{c}{c+1}
$$

**Assumption 2 supplies the likelihood, Assumption 1 supplies the prior.** Drop Assumption 1 and
you have a Gaussian-shaped likelihood against an unknown marginal, and nothing follows. Everything
downstream — Lemmas 2 and 3, waypoint planning, the tridiagonal precision matrix — is then
textbook linear-Gaussian smoothing. The contribution is the setup, not the inference.

### A consistency condition, not stated in the paper

The two assumptions constrain each other, and the constraint is checkable. Marginalising Lemma 1
must reproduce Assumption 1:

$$
\alpha^2 A (cI) A^\top + \alpha I \;\overset{!}{=}\; cI
\qquad\Longrightarrow\qquad
A A^\top = \frac{c+1}{c}\,I
$$

So **$A$ must be an orthogonal matrix scaled by $\sqrt{(c+1)/c} = 1/\sqrt{\alpha}$**; any other
$A$ makes the two assumptions mutually inconsistent. Note $\sqrt{(c+1)/c} \to 1$ as $c \to \infty$,
which explains why the paper's clean interpolation result is stated for orthogonal $A$ and
$c \to \infty$ — that is the regime where its own assumptions are compatible. Combined with
$\alpha A$ having singular values $\sqrt{\alpha} < 1$, the chain contracts at rate $\sqrt{\alpha}$
per link.

_(Derived here, not read off the paper — worth checking against their appendix.)_

### $A$ is a resolvent, not a transition operator

The obvious parametrisation would be $A(\Delta t)$ or $A^{\Delta t}$, making it a true temporal
step operator. It isn't, because the positive is never at a fixed offset — the lag is a geometric
random variable that was marginalised out *inside the training objective*. A single $A$ is fitted
to the whole mixture.

If representations evolved linearly, $\psi_{t+1} \approx M\psi_t$, then

$$
A \;\approx\; (1-\gamma)\sum_{t \ge 0} \gamma^t M^t \;=\; (1-\gamma)(I - \gamma M)^{-1}
$$

the Neumann series — i.e. **$A$ is the successor representation operator, not the transition
operator**. Two consequences:

- **$A$ is not a semigroup.** $A^2$ is not "$A$ at twice the lag". Chaining $m$ links convolves
  $m$ geometric lags into a negative binomial, so $m$ links ahead is *negative-binomially-many
  steps* ahead. You cannot ask this model where you will be in exactly ten steps.
- **But the chain is homogeneous by construction**, which is exactly what the tridiagonal solve
  needs. Every link is the same random horizon with the same matrix. An $A(\Delta t)$
  parametrisation would also give a tridiagonal system, but you would have to commit to a schedule
  of offsets and jointly optimise over it. Homogeneity was bought by giving up the ability to name
  a horizon.

### Which inference questions this actually answers

"Inference" here means **probabilistic inference** — computing marginals and conditionals in a
graphical model — not statistical inference, and not a network forward pass. Once Lemma 1 has
established that the representations form a Gauss-Markov chain, the supported queries are the
standard ones for a linear-Gaussian state space model:

| query | standard name | where |
|---|---|---|
| $p(\psi_{t+} \mid \psi_0)$ | prediction | Lemma 1 |
| $p(\psi_w \mid \psi_0, \psi_{t+})$ | smoothing, one latent | Lemma 2 |
| $p(\psi_{w_1}, \dots, \psi_{w_n} \mid \psi_0, \psi_g)$ | smoothing, $n$ latents | Lemma 3 |

**Planning is the smoothing query.** That is the planning-as-inference tradition — Attias (2003),
Toussaint & Storkey (2006), Levine's (2018) control-as-inference survey — where "find a path from
$s$ to $g$" is recast as "condition the chain on both endpoints and read off the posterior over
what happened in between".

Three limits on what that covers, worth being precise about:

- **The queries are over $\psi$, not $x$.** A posterior over representations is not a posterior
  over states. You cannot ask "how likely is the robot to be in the kitchen"; you get a Gaussian
  over a lossy code, and getting back to states needs retrieval.
- **Only endpoint conditioning.** There is no filtering from partial observations, no conditioning
  on a region or a set of acceptable goals, no conditioning on a reward. The chain is conditioned
  on two fully-observed endpoint states and nothing else.
- **Interpolation is the special case, not the mechanism.** The general answer is a tridiagonal
  solve; the lerp appears only when $A$ is orthogonal and $c \to \infty$.

### Planning: the precision matrix is a discrete Laplacian

Given $\psi_0 = \psi(s)$ and $\psi_g = \psi(g)$, condition the chain on both ends. For a single
waypoint,

$$
p(\psi_w \mid \psi_0, \psi_g) \;\propto\;
\exp\!\left(-\tfrac{\lVert \psi_w - \alpha A\psi_0\rVert^2}{2\alpha}\right)
\exp\!\left(-\tfrac{\lVert \psi_g - \alpha A\psi_w\rVert^2}{2\alpha}\right)
$$

Collecting quadratic and linear terms in $\psi_w$ gives their Lemma 2:

$$
\Lambda = \tfrac{1}{\alpha}I + \alpha A^\top A,
\qquad \mu = \Lambda^{-1}\left(A\psi_0 + A^\top \psi_g\right)
$$

In words: **$A\psi_0$ pushes the start forward, $A^\top\psi_g$ pulls the goal backward, and the
waypoint is the precision-weighted compromise.** (Their Lemma 2 calls this matrix $\Sigma$ but
then uses $\Sigma^{-1}$ in the mean — it is a precision, not a covariance. Reading it as $\Lambda$
makes it consistent.)

For $n$ waypoints the joint posterior is Gaussian with block-tridiagonal precision — diagonal
blocks $\tfrac1\alpha I + \alpha A^\top A$, off-diagonal blocks $-A^\top$, and a right-hand side
that is zero everywhere except $A\psi_0$ at the first node and $A^\top\psi_g$ at the last. That is
a block Thomas solve, or equivalently an RTS smoother, because this *is* a linear-Gaussian state
space model.

The punchline is what that matrix becomes in the limit. Apply the consistency condition
$A^\top A = \tfrac1\alpha I$ and let $c \to \infty$ (so $\alpha \to 1$), absorbing the rotation:

$$
\Lambda \;=\;
\begin{pmatrix}
2 & -1 & & \\
-1 & 2 & -1 & \\
& -1 & 2 & -1 \\
& & -1 & 2
\end{pmatrix}
$$

the **1-D discrete Laplacian with Dirichlet boundary conditions** $\psi_0$ and $A^\top\psi_g$. Its
solution is the discrete harmonic extension, and in one dimension harmonic means linear:

$$
\mu_i \;=\; (1 - \lambda_i)\,\psi_0 \;+\; \lambda_i\, A^\top \psi_g,
\qquad \lambda_i = \frac{i}{n+1}
$$

Planning collapses to lerping between two encoded states. That is the title.

### What one "link" costs in real time

The chain's only clock is the link, and a link is worth a random number of real steps. The weights
$(1-\gamma)\gamma^t$ in Eq. 1 are a Geometric distribution on $\{0, 1, 2, \dots\}$, so

$$
\mathbb{E}[T] = \frac{\gamma}{1-\gamma} = H, \qquad
\mathrm{SD}[T] = \frac{\sqrt\gamma}{1-\gamma} \approx H, \qquad
P(T = 0) = 1-\gamma
$$

Two things fall out of that immediately. There is an **atom at $t = 0$** — with probability
$1-\gamma$ the "future state" is the current state, so the learned distance has a floor it can
never go below. And the **spread equals the mean**: one link is not $H$ steps, it is a draw from a
skewed distribution running from $0$ to several hundred.

Composition is clean in operator form. With $K = (1-\gamma)\sum_t \gamma^t P^t$,

$$
K^2 = (1-\gamma)^2\sum_{s \ge 0}(s+1)\gamma^s P^s
$$

and $(s+1)(1-\gamma)^2\gamma^s$ is the negative binomial pmf with $r = 2$. In general $m$ links
give $T \sim \mathrm{NegBinom}(r = m)$ with mean $mH$ and standard deviation
$\sqrt{m}\,\sqrt\gamma/(1-\gamma)$. At $\gamma = 0.99$ (so $H = 99$):

| waypoints $n$ | links $m$ | lag to goal, steps |
|---|---|---|
| 0 | 1 | $99 \pm 100$ |
| 3 | 4 | $396 \pm 199$ |
| 4 | 5 | $495 \pm 222$ |
| 9 | 10 | $990 \pm 315$ |

**This is the structural fact that matters.** Inverting $\mathbb{E}[\text{lag}] = (n+1)H$:

$$
n \;\approx\; \frac{D}{H} - 1
$$

for a goal genuinely $D$ steps away. So $n$ is not a resolution knob — it is a *distance
measurement*. Asking for more waypoints does not subdivide the same journey more finely, it
asserts a longer journey and conditions a chain that no longer matches your $s$ and $g$.

Consequences worth keeping:

- **Subgoal spacing is fixed at $H$ by $\gamma$, at training time.** If the low-level policy is
  reliable over ten steps and $H = 99$, no number of waypoints fixes that; only retraining with a
  smaller $\gamma$ does. $\gamma$ is the hierarchy-granularity hyperparameter, and it also sets the
  range over which the critic is informative — two jobs, one knob.
- **Even spacing in $\psi$-space is not even spacing in time.** The lerp is uniform in *links*, but
  each link has $\mathrm{SD} \approx$ its own mean. At $\gamma = 0.99$, $P(T \le 10) \approx 10\%$
  (a near no-op subgoal) and $P(T > 300) \approx 5\%$ (a hop the policy cannot make). Over a
  ten-link plan, expect at least one of each.

The partial defence is that a goal-reaching policy cares about achievability, not timing, and a
state one occupancy draw ahead is by construction "reachable within about a horizon" — which is
the policy's assumed competence. The smearing bites only when timing matters, or when the tail
hands the policy something unreachable and the plan stalls.

### Getting from representations back to actions

$\mu_i$ is a vector in $\mathbb{R}^k$, not a state, and $\psi$ is deliberately non-invertible — it
is trained to be a sufficient statistic for the discounted future and to destroy everything else.
So either:

1. **Retrieve** — nearest neighbour $\arg\min_x \lVert \psi(x) - \mu_i \rVert$ over the dataset,
   yielding a real subgoal state. Better: sample $\psi_w \sim \mathcal{N}(\mu_i, \Sigma_i)$ a few
   times and retrieve each, ranking by the reachability score below. Taking the mean throws away
   the posterior covariance, which is the model's own uncertainty estimate.
2. **Feed it straight in**, if the low-level policy is already conditioned on goal
   *representations*, $\pi(a \mid \psi(s), \psi_{\text{goal}})$. Then nothing is decoded.

Route 1 undercuts the headline slightly: the $O(k^3)$ solve is followed by an $O(N)$ lookup over a
state buffer. Search was moved, not removed — though a flat similarity lookup with no dynamics
rollout is a genuine improvement over search over action sequences.

Separately, Assumption 2 hands you a free scoring function before you plan anything:

$$
\tfrac{1}{2}\lVert A\psi(s) - \psi(g)\rVert^2 \;=\; -\log\frac{p_{t+}(g \mid s)}{p(g)} + \text{const}
$$

**Squared distance in representation space is negative log reachability**, normalised by how common
$g$ is. This is the contrastive-RL critic — a $Q$-function. Use it to rank goals and to reject
plans, because if this number is large the waypoints are confidently interpolating a path that
does not exist.

Note finally that **actions appear nowhere in the paper**. $p_{t+}$ is the occupancy under whatever
policy collected the data, so a "plan" is a prediction of what the behaviour policy tends to do
between $s$ and $g$, not an optimal path. And "planning" here means subgoal generation only; the
subgoal-to-action problem is delegated to an assumed short-horizon goal-conditioned policy.
