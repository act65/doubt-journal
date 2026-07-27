import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From a coin to an Epistemic Neural Network

    A step-by-step tutorial for the question:

    > *In "Epistemic Neural Networks" (Osband et al. 2021, arXiv:2107.08924),
    > eqns 1 & 2 define a **joint** distribution over the outputs
    > $\{y_1,\dots,y_\tau\}$ of $\tau$ inputs $\{x_1,\dots,x_\tau\}$.
    > Why? And — since we only ever observe single labels — how is the
    > joint ever **learned**?*

    The plan:

    1. **The coin.** Marginal vs joint for a Beta–Bernoulli coin. Different
       priors with the **same marginal** but **different joints**; more flips
       ($\tau=2,3,4$) sharpen the separation; the joint collapses toward a
       product as data arrives.
    2. **The sampling view.** "Sample one hypothesis $z$, then predict
       conditionally-independently" — this *is* eqn 2, with $z=\theta$.
    3. **The network.** Replace the scalar $\theta$ with an *uncertain
       function*. Build an Epistemic NN in JAX, train it on single labels
       only, and watch the joint structure emerge — collapsed where there is
       data, correlated where there is not.

    The load-bearing identity to keep in your head:

    $$\underbrace{\mathrm{Cov}(y_i,y_j)}_{\text{joint correlation}}
      \;=\; \underbrace{\mathrm{Var}_{\text{posterior}}(\theta)}_{\text{epistemic uncertainty}}.$$

    The correlation in the joint *is* the epistemic uncertainty (derived below).
    A **companion notebook**, `variance_vs_information_eu.py`, then shows how the
    *information* (mutual-information / BALD) EU is derived from this
    variance-flavoured EU by a change of metric — the argument of `papers/ave`
    §2.2. That thread is deliberately kept out of this notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # this is a tiny CPU demo; skip GPU init

    from functools import partial

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    return jax, jnp, np, optax, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 — The coin: marginal vs joint

    A coin has an unknown bias $\theta = P(\text{heads})$. We put a prior
    on $\theta$, observe single flips, and ask about **two future flips**.

    Conditional on $\theta$, flips are independent, so for a single $\theta$:

    $$P(y_1,y_2\mid\theta) = \theta^{y_1+y_2}(1-\theta)^{2-y_1-y_2}.$$

    The *predictive* joint integrates the shared, uncertain $\theta$ out —
    exactly the structure of eqn 2 ($\int p_z(z)\prod_t\dots$):

    $$P(y_1,y_2) = \int_0^1 P(y_1,y_2\mid\theta)\,p(\theta)\,d\theta.$$

    For $\theta\sim\text{Beta}(a,b)$ every entry is a moment of $\theta$:

    - $P(H,H)=\mathbb{E}[\theta^2]$
    - $P(H,T)=P(T,H)=\mathbb{E}[\theta(1-\theta)]$
    - $P(T,T)=\mathbb{E}[(1-\theta)^2]$

    and the off-diagonal covariance of the two outcomes works out to
    $\mathrm{Cov}(y_1,y_2)=\mathbb{E}[\theta^2]-\mathbb{E}[\theta]^2=\mathrm{Var}(\theta)$.
    """)
    return


@app.cell
def _(np):
    def beta_stats(a, b):
        """Predictive stats for two future flips under theta ~ Beta(a, b)."""
        m1 = a / (a + b)                              # E[theta]   = marginal P(H)
        m2 = a * (a + 1) / ((a + b) * (a + b + 1))    # E[theta^2] = P(H,H)
        joint = np.array(
            [
                [1 - 2 * m1 + m2, m1 - m2],  # (T,T), (T,H)
                [m1 - m2, m2],               # (H,T), (H,H)
            ]
        )
        return {
            "marginal_H": m1,
            "P_HH": m2,
            "cov": m2 - m1 ** 2,   # == Var(theta)
            "joint": joint,        # rows = y1 in {T,H}, cols = y2 in {T,H}
        }

    # Three models that ALL have marginal P(H) = 0.5 but different joints.
    model_names = ["known fair\n(theta=0.5)", "uniform prior\nBeta(1,1)", "two-point prior\ntheta in {0,1}"]

    # known fair coin: theta fixed at 0.5 -> independent flips
    _fair = {"marginal_H": 0.5, "P_HH": 0.25, "cov": 0.0,
             "joint": np.array([[0.25, 0.25], [0.25, 0.25]])}
    # uniform prior Beta(1,1): E[theta^2] = 1/3
    _uniform = beta_stats(1.0, 1.0)
    # two-point prior theta in {0,1} each w.p. 1/2: E[theta]=1/2, E[theta^2]=1/2
    _twopoint = {"marginal_H": 0.5, "P_HH": 0.5, "cov": 0.25,
                 "joint": np.array([[0.5, 0.0], [0.0, 0.5]])}

    joint_tables = [_fair, _uniform, _twopoint]
    return beta_stats, joint_tables, model_names


@app.cell
def _(joint_tables, model_names, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.4))
    _ticks = ["T", "H"]
    for _ax, _m, _name in zip(_axes, joint_tables, model_names):
        _J = _m["joint"]
        _ax.imshow(_J, cmap="Blues", vmin=0, vmax=0.5)
        for _i in range(2):
            for _j in range(2):
                _ax.text(_j, _i, f"{_J[_i, _j]:.3f}", ha="center", va="center",
                         color="black", fontsize=11)
        _ax.set_xticks([0, 1], _ticks)
        _ax.set_yticks([0, 1], _ticks)
        _ax.set_xlabel("$y_2$")
        _ax.set_ylabel("$y_1$")
        _ax.set_title(f"{_name}\nP(H)={_m['marginal_H']:.2f}  cov={_m['cov']:.3f}",
                      fontsize=9)
    _fig.suptitle("Same marginal P(H)=0.5, three different joints", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Read the three panels.** Every model says $P(H)=0.5$ — a marginal
    prediction cannot tell them apart. The **joints** are completely different:

    - *Known fair coin* (pure **aleatoric**): flips independent, joint is a
      product, covariance $0$.
    - *Uniform prior*: covariance $1/12\approx0.083$ — mild correlation from
      not-yet-knowing $\theta$.
    - *Two-point prior* ("all-heads or all-tails"): covariance $1/4$ — one
      flip tells you the other. Pure **epistemic**.

    This is the entire reason eqn 2 exists: the joint exposes the
    aleatoric/epistemic split that the marginal hides.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1½ — More flips, more priors

    Two flips already separated the coins. With $\tau$ flips the separation
    grows and a richer structure appears. Because flips are exchangeable given
    $\theta$, the joint depends only on the **count** $S_\tau=\sum_t y_t$, and

    $$P(S_\tau=k)=\binom{\tau}{k}\,\mathbb{E}_\theta[\theta^k(1-\theta)^{\tau-k}]
    \qquad(\text{a Beta–Binomial when the prior is Beta}).$$

    Epistemic uncertainty shows up as **over-dispersion**: the count spreads
    toward the extremes relative to the i.i.d. Binomial that a *known-*$\theta$
    coin would give. We compare five priors that **all have marginal
    $P(H)=0.5$** but increasing EU:

    $$\underbrace{\text{fair }(\theta=0.5)}_{\mathrm{Var}=0}
    < \text{Beta}(8,8) < \text{uniform }\text{Beta}(1,1)
    < \text{Jeffreys }\text{Beta}(\tfrac12,\tfrac12)
    < \underbrace{\text{two-point }\{0,1\}}_{\mathrm{Var}=1/4}.$$
    """)
    return


@app.cell
def _(np):
    from math import comb, exp, lgamma

    def _betaln(a, b):
        return lgamma(a) + lgamma(b) - lgamma(a + b)

    def count_dist(kind, params, tau):
        """P(S_tau = k) for k=0..tau, given a prior on theta. Marginal is 0.5 for all."""
        ks = range(tau + 1)
        if kind == "point":                        # theta fixed -> Binomial
            th = params
            return np.array([comb(tau, k) * th ** k * (1 - th) ** (tau - k) for k in ks])
        if kind == "beta":                         # Beta-Binomial
            a, b = params
            return np.array([comb(tau, k) * exp(_betaln(a + k, b + tau - k) - _betaln(a, b)) for k in ks])
        if kind == "twopoint":                     # theta in {0,1}, each 1/2
            p = np.zeros(tau + 1)
            p[0] = p[-1] = 0.5
            return p
        raise ValueError(kind)

    def var_theta(kind, params):
        if kind == "point":
            return 0.0
        if kind == "beta":
            a, b = params
            return a * b / ((a + b) ** 2 * (a + b + 1))
        if kind == "twopoint":
            return 0.25

    flip_priors = [
        ("fair theta=0.5", "point", 0.5),
        ("Beta(8,8)", "beta", (8.0, 8.0)),
        ("uniform Beta(1,1)", "beta", (1.0, 1.0)),
        ("Jeffreys Beta(.5,.5)", "beta", (0.5, 0.5)),
        ("two-point {0,1}", "twopoint", None),
    ]
    return count_dist, flip_priors, var_theta


@app.cell
def _(count_dist, flip_priors, plt, var_theta):
    _taus = [2, 3, 4]
    _fig, _axes = plt.subplots(len(_taus), len(flip_priors), figsize=(12, 6), sharey=True)
    for _ri, _tau in enumerate(_taus):
        for _ci, (_name, _kind, _params) in enumerate(flip_priors):
            _ax = _axes[_ri, _ci]
            _ax.bar(range(_tau + 1), count_dist(_kind, _params, _tau), color="C0", alpha=0.85)
            _ax.set_ylim(0, 1)
            _ax.set_xticks(range(_tau + 1))
            if _ri == 0:
                _ax.set_title(f"{_name}\nVar(theta)={var_theta(_kind, _params):.3f}", fontsize=8)
            if _ci == 0:
                _ax.set_ylabel(f"tau={_tau}\nP(S=k)", fontsize=9)
    _fig.suptitle("Count distribution P(#heads): identical marginal 0.5, EU = over-dispersion toward the extremes",
                  fontsize=11)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np, plt, var_theta):
    # Covariance matrix of tau=4 flips. Diagonal = Var(y_i) = mu(1-mu) = 0.25 (total);
    # every off-diagonal = Var(theta) (pure epistemic). Shown for 3 priors.
    _tau, _mu = 4, 0.5
    _sel = [("fair theta=0.5", "point", 0.5),
            ("uniform Beta(1,1)", "beta", (1.0, 1.0)),
            ("two-point {0,1}", "twopoint", None)]
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for _ax, (_name, _kind, _params) in zip(_axes, _sel):
        _v = var_theta(_kind, _params)
        _Sigma = np.full((_tau, _tau), _v)
        np.fill_diagonal(_Sigma, _mu * (1 - _mu))
        _ax.imshow(_Sigma, cmap="Blues", vmin=0, vmax=0.25)
        for _i in range(_tau):
            for _j in range(_tau):
                _ax.text(_j, _i, f"{_Sigma[_i, _j]:.2f}", ha="center", va="center", fontsize=7)
        _ax.set_xticks(range(_tau))
        _ax.set_yticks(range(_tau))
        _ax.set_xlabel("flip j")
        _ax.set_ylabel("flip i")
        _ax.set_title(f"{_name}\noff-diag = Var(theta) = {_v:.3f}", fontsize=8)
    _fig.suptitle(r"Cov$(y_i,y_j)$, $\tau=4$: diagonal = total per-flip (0.25), off-diagonal = epistemic Var($\theta$)",
                  fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(count_dist, flip_priors, np, plt):
    _taus = np.arange(1, 9)
    _fig, _ax = plt.subplots(figsize=(6.6, 4.0))
    for _name, _kind, _params in flip_priors:
        _pall = [count_dist(_kind, _params, int(_t))[-1] for _t in _taus]  # P(S=tau) = all heads
        _ax.semilogy(_taus, _pall, "-o", ms=3, label=_name)
    _ax.set_xlabel(r"$\tau$  (number of flips in the joint)")
    _ax.set_ylabel(r"P(all $\tau$ heads) = E$[\theta^\tau]$")
    _ax.set_title("The joint over more flips separates the priors — the marginals never could")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the extra flips reveal.** All five priors are indistinguishable on a
    single flip (marginal $0.5$). But:

    - **Count distribution** (grid): fans toward the extremes as EU grows — the
      fair coin is a tight Binomial, the two-point prior is all-or-nothing.
    - **Covariance matrix**: the diagonal is always the total per-flip variance
      $0.25$, while *every* off-diagonal is exactly $\mathrm{Var}(\theta)$ — zero
      for the fair coin (independent), $0.25$ for the two-point coin (rank-one,
      perfectly correlated).
    - $P(\text{all heads})=\mathbb{E}[\theta^\tau]$: separates the priors *more*
      the longer the run.

    This is why **joint predictions over many inputs**, not marginals, carry the
    epistemic information. Next: where does that clean $\mathrm{Cov}=\mathrm{Var}(\theta)$
    actually come from?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Aside — where does $\mathrm{Cov}(y_i,y_j)=\mathrm{Var}(\theta)$ come from?

    Two future flips $y_i,y_j$. Only two facts are needed:

    1. **Conditional independence:** given $\theta$, the flips are independent.
    2. **The conditional mean is $\theta$:** $\mathbb{E}[y_i\mid\theta]=\theta$.

    Direct computation for $i\neq j$:

    $$\mathbb{E}[y_i y_j]=\mathbb{E}_\theta\big[\mathbb{E}[y_i y_j\mid\theta]\big]
    \overset{(1)}{=}\mathbb{E}_\theta\big[\mathbb{E}[y_i\mid\theta]\,\mathbb{E}[y_j\mid\theta]\big]
    \overset{(2)}{=}\mathbb{E}[\theta^2],$$

    while $\mathbb{E}[y_i]=\mathbb{E}_\theta[\theta]$, so

    $$\mathrm{Cov}(y_i,y_j)=\mathbb{E}[\theta^2]-\mathbb{E}[\theta]^2=\mathrm{Var}(\theta).$$

    The one-line version is the **law of total covariance**:

    $$\mathrm{Cov}(y_i,y_j)=\underbrace{\mathbb{E}_\theta[\mathrm{Cov}(y_i,y_j\mid\theta)]}_{=\,0\ \text{(cond. indep.)}}
    +\underbrace{\mathrm{Cov}\!\big(\mathbb{E}[y_i\mid\theta],\mathbb{E}[y_j\mid\theta]\big)}_{=\,\mathrm{Var}(\theta)}.$$

    Both ingredients are load-bearing: conditional independence kills the
    aleatoric term, leaving *only* the variance of the shared latent.

    #### Why the off-diagonal is special: it isolates EU

    The **diagonal** ($i=j$, a single flip) obeys the law of total *variance*:

    $$\mathrm{Var}(y_i)=\underbrace{\mathbb{E}_\theta[\mathrm{Var}(y_i\mid\theta)]}_{\text{aleatoric}=\,\mathbb{E}[\theta(1-\theta)]}
    +\underbrace{\mathrm{Var}_\theta(\mathbb{E}[y_i\mid\theta])}_{\text{epistemic}=\,\mathrm{Var}(\theta)}.$$

    A single observation **mixes** AU and EU. The **off-diagonal** drops the
    aleatoric term entirely — noise is conditionally independent across flips and
    washes out — leaving *pure* epistemic $\mathrm{Var}(\theta)$. That is exactly
    the diagonal-vs-off-diagonal structure you just saw in the covariance matrix.

    #### The honest generalization (what Part 3 measures)

    At two *different* inputs $x_a,x_b$ the identity becomes a **cross-covariance**:

    $$\mathrm{Cov}(y_a,y_b)=\mathrm{Cov}_z\big(\underbrace{\mathbb{E}[y\mid x_a,z]}_{p_a(z)},\ \underbrace{\mathbb{E}[y\mid x_b,z]}_{p_b(z)}\big),$$

    which is exactly the `Cov_z(p_a, p_b)` computed in the neural-net joint cell.
    The scalar coin is just the case $x_a=x_b$ (same conditional mean at both).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Now watch the joint get *learned* from single flips

    We never observe a joint label. We flip a **genuinely fair** coin
    ($\theta_\text{true}=0.5$), feed single outcomes to a Beta(1,1) model, and
    track the predictive joint's covariance = the posterior variance of $\theta$.
    The three markers are "before data / after 5 / after 10k".
    """)
    return


@app.cell
def _(beta_stats, np, plt):
    _rng = np.random.default_rng(0)
    _n_max = 10_000
    _flips = _rng.integers(0, 2, size=_n_max)      # fair coin, theta_true = 0.5
    _cum_heads = np.concatenate([[0], np.cumsum(_flips)])
    _ns = np.unique(np.concatenate([
        np.array([0, 5]),
        np.round(np.logspace(0, np.log10(_n_max), 40)).astype(int),
    ]))
    _cov, _marg = [], []
    for _n in _ns:
        _h = _cum_heads[_n]
        _s = beta_stats(1.0 + _h, 1.0 + (_n - _h))
        _cov.append(_s["cov"])
        _marg.append(_s["marginal_H"])
    _cov, _marg = np.array(_cov), np.array(_marg)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    _a1.loglog(_ns + 1, _cov, "-o", ms=3, color="C3")
    for _nmark, _lab in [(0, "n=0"), (5, "n=5"), (10_000, "n=10k")]:
        _sm = beta_stats(1.0 + _cum_heads[_nmark], 1.0 + (_nmark - _cum_heads[_nmark]))
        _a1.plot(_nmark + 1, _sm["cov"], "s", ms=9, color="k")
        _a1.annotate(f"{_lab}\ncov={_sm['cov']:.4f}", (_nmark + 1, _sm["cov"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    _a1.set_xlabel("# single flips observed")
    _a1.set_ylabel("joint covariance  =  Var(theta)")
    _a1.set_title("Epistemic uncertainty collapses ~ 1/n")

    _a2.semilogx(_ns + 1, _marg, "-o", ms=3, color="C0")
    _a2.axhline(0.5, ls="--", color="grey")
    _a2.set_ylim(0, 1)
    _a2.set_xlabel("# single flips observed")
    _a2.set_ylabel("marginal P(H)")
    _a2.set_title("Marginal stays ~0.5 (aleatoric never leaves)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The answer to "how is the joint learned?"** It isn't observed — it is
    *derived* from a prior + single-outcome data, exactly as a Bayesian
    posterior. The correlation (left plot) is the posterior variance of the
    shared latent $\theta$, and it shrinks $\sim 1/n$ while the marginal
    (right plot) stays pinned at the aleatoric $0.5$. No joint label ever
    entered the computation.

    ## Part 2 — The sampling view (this is literally eqn 2)

    We do not need the analytic integral. Eqn 2 says: **sample one
    hypothesis $z\sim p_z$, predict conditionally-independently, average.**
    For the coin, $z=\theta$:

    $$P(y_1,y_2)\approx\frac1S\sum_{s=1}^S
      \prod_{t=1}^2 \text{Bernoulli}(y_t\mid\theta^{(s)}),
      \qquad \theta^{(s)}\sim p(\theta).$$

    The averaging over $z$ is what couples $y_1,y_2$: a *mixture of
    products is not a product*.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    # Posterior after 5 heads, 0 tails -> Beta(6, 1).
    _a, _b = 6.0, 1.0
    _key = jax.random.PRNGKey(1)
    _S = 200_000
    _theta = jax.random.beta(_key, _a, _b, shape=(_S,))          # sample z = theta
    _kf = jax.random.split(_key, 2)
    # conditional-independent flips given theta
    _y1 = (jax.random.uniform(_kf[0], (_S,)) < _theta).astype(jnp.float32)
    _y2 = (jax.random.uniform(_kf[1], (_S,)) < _theta).astype(jnp.float32)

    _mc_HH = float(jnp.mean(_y1 * _y2))
    _mc_cov = float(jnp.mean(_y1 * _y2) - jnp.mean(_y1) * jnp.mean(_y2))
    _an_HH = _a * (_a + 1) / ((_a + _b) * (_a + _b + 1))
    _an_cov = _an_HH - (_a / (_a + _b)) ** 2

    mo.md(
        f"""
        **Beta(6, 1) posterior (5 heads, 0 tails), {_S:,} samples of $z=\\theta$:**

        | quantity | Monte-Carlo (sample $z$, predict indep.) | analytic $\\int p(z)\\dots$ |
        |---|---|---|
        | $P(H,H)$ | {_mc_HH:.4f} | {_an_HH:.4f} |
        | $\\mathrm{{Cov}}(y_1,y_2)$ | {_mc_cov:.4f} | {_an_cov:.4f} |

        They match. The "integrate out a shared latent" recipe of eqn 2 is just
        "sample a hypothesis, predict, average" — nothing more exotic. Now we
        make $z$ index a **function** instead of a scalar.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 — From coin to Epistemic Neural Network

    Generalise the coin. The unknown is no longer a scalar $\theta$ but
    **which function** generated the labels. The epistemic index $z$ now
    indexes that uncertain function, and eqn 2 becomes
    $\hat P(y_{1:\tau})=\int p_z(z)\prod_t \mathrm{softmax}(f_\theta(x_t,z))_{y_t}\,dz$.

    An ENN is any net whose output depends on such an index $z$. The cleanest
    instantiation — and the one the paper reduces to when $z$ is a *particle*
    index $z\in\{1,\dots,K\}$ — is an **ensemble of randomized-prior
    functions** (Osband et al. 2018). Member $k$ is

    $$f_k(x) = \underbrace{t_k(x)}_{\text{trainable}} + \beta\,\underbrace{p_k(x)}_{\text{fixed random prior}},$$

    with a *different frozen random* prior $p_k$ per member. This is the exact
    neural analogue of the coin, and it makes the learning mechanism visible:

    - **Where there is data**, each $t_k$ adjusts to fit the label *despite*
      its own prior $\beta p_k$ → all members agree → uncertainty collapses.
      This is $\theta$ getting pinned down.
    - **Where there is no data**, nothing constrains $t_k$; the frozen priors
      $p_k$ differ, so the members *fan out* → epistemic spread. This is the
      prior's shared-across-inputs structure surviving.

    (The paper's headline *epinet* shares one body across indices for
    parameter efficiency — $f(x,z)=\mu(x)+g(\mathrm{sg}[\phi(x)],z)+\beta\sigma^P(x,z)$;
    same idea, the ensemble just makes it transparent.)

    ### A toy that mirrors the coin exactly

    1-D inputs, **two dense clusters of 50/50 labels** with an empty gap:

    - $x\in[-7,-4]$ and $x\in[4,7]$: label $\sim\text{Bernoulli}(0.5)$ —
      **aleatoric** regions (the "fair coin"). Marginal $\to 0.5$.
    - $x\in(-4,4)$: **no data at all** — an **epistemic** region (the
      "unknown coin"). Marginal also $\approx0.5$.

    The marginal is $\approx0.5$ *everywhere* — useless for locating
    ignorance. Only the **spread across $z$**, and the **joint**, tell the
    two kinds of region apart.
    """)
    return


@app.cell
def _(jnp, np):
    _rng = np.random.default_rng(3)
    _r = lambda lo, hi, n: _rng.uniform(lo, hi, n)

    _xl, _xr = _r(-7.0, -4.0, 120), _r(4.0, 7.0, 120)   # two dense 50/50 clusters
    _yl = (np.arange(120) % 2).astype(float)            # exactly balanced -> marginal 0.5
    _yr = (np.arange(120) % 2).astype(float)

    X_train = jnp.asarray(np.concatenate([_xl, _xr]), dtype=jnp.float32)
    y_train = jnp.asarray(np.concatenate([_yl, _yr]), dtype=jnp.float32)
    x_grid = jnp.linspace(-9.0, 9.0, 400, dtype=jnp.float32)

    # test pairs for the joint
    pair_alea = (-6.0, -5.0)    # both inside a well-observed 50/50 cluster
    pair_epi = (-1.0, -0.4)     # both inside the unobserved gap, close together
    return X_train, pair_alea, pair_epi, x_grid, y_train


@app.cell
def _(X_train, np, pair_alea, pair_epi, plt, y_train):
    _fig, _ax = plt.subplots(figsize=(10, 2.6))
    _x, _y = np.asarray(X_train), np.asarray(y_train)
    _ax.scatter(_x, _y + np.random.default_rng(0).normal(0, 0.02, _y.shape),
                s=12, alpha=0.4, color="C0")
    _ax.axvspan(-4.0, 4.0, color="grey", alpha=0.08)
    _ax.text(0.0, 0.5, "no data\n(epistemic gap)", ha="center", va="center",
             color="grey", fontsize=9)
    for _p, _c, _lab in [(pair_alea, "C2", "aleatoric pair"), (pair_epi, "C3", "epistemic pair")]:
        _ax.axvspan(_p[0] - 0.1, _p[1] + 0.1, color=_c, alpha=0.25)
        _ax.text(np.mean(_p), 1.18, _lab, ha="center", color=_c, fontsize=9)
    _ax.set_xlim(-9, 9)
    _ax.set_ylim(-0.3, 1.35)
    _ax.set_xlabel("x")
    _ax.set_ylabel("label")
    _ax.set_title("Two dense 50/50 clusters (aleatoric) + empty gap (epistemic)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(jax, jnp, np):
    # ---- ensemble of randomized-prior functions (an ENN with particle index z) ----
    n_members = 40
    beta = 3.0             # prior scale
    _sizes = [1, 32, 32, 1]

    def init_mlp(key, sizes, scale=1.0):
        keys = jax.random.split(key, len(sizes) - 1)
        params = []
        for i in range(len(sizes) - 1):
            W = jax.random.normal(keys[i], (sizes[i], sizes[i + 1])) * scale / np.sqrt(sizes[i])
            params.append((W, jnp.zeros(sizes[i + 1])))
        return params

    def mlp(layers, x):
        h = x
        n = len(layers)
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < n - 1:          # tanh hidden, linear output
                h = jnp.tanh(h)
        return h

    def member_logit(t_params, p_params, x):
        """One member: trainable net + beta * frozen prior (long lengthscale)."""
        xv = jnp.atleast_1d(x)
        return (mlp(t_params, xv) + beta * mlp(p_params, xv / 3.0))[0]

    # stacked params: leading axis = member index z
    _kt = jax.random.split(jax.random.PRNGKey(0), n_members)
    _kp = jax.random.split(jax.random.PRNGKey(1), n_members)
    tparams_init = jax.vmap(lambda k: init_mlp(k, _sizes, scale=1.0))(_kt)
    pparams = jax.vmap(lambda k: init_mlp(k, _sizes, scale=2.0))(_kp)   # frozen priors
    return member_logit, pparams, tparams_init


@app.cell
def _(X_train, jax, jnp, member_logit, optax, pparams, tparams_init, y_train):
    def _member_loss(t_params, p_params):
        L = jax.vmap(lambda x: member_logit(t_params, p_params, x))(X_train)
        p = jnp.clip(jax.nn.sigmoid(L), 1e-6, 1 - 1e-6)
        return jnp.mean(-(y_train * jnp.log(p) + (1 - y_train) * jnp.log(1 - p)))

    def total_loss(tparams):
        # each member fits the SAME data but keeps its OWN frozen prior
        return jnp.mean(jax.vmap(_member_loss)(tparams, pparams))

    _opt = optax.adamw(3e-3, weight_decay=1e-4)
    _opt_state = _opt.init(tparams_init)

    @jax.jit
    def _step(tparams, opt_state):
        loss, grads = jax.value_and_grad(total_loss)(tparams)
        updates, opt_state = _opt.update(grads, opt_state, tparams)
        return optax.apply_updates(tparams, updates), opt_state, loss

    _t = tparams_init
    _loss = None
    for _i in range(3000):
        _t, _opt_state, _loss = _step(_t, _opt_state)

    tparams_trained = _t
    print(f"final training BCE: {float(_loss):.4f}  (50/50 floor = ln2 ~ 0.693)")
    return (tparams_trained,)


@app.cell
def _(jax, member_logit, np, pparams, tparams_trained, x_grid):
    # each member's probability curve over the grid; leading axis = member z
    _logits = jax.vmap(lambda tp, pp: jax.vmap(lambda x: member_logit(tp, pp, x))(x_grid))(
        tparams_trained, pparams
    )
    probs = np.asarray(jax.nn.sigmoid(_logits))   # (n_members, n_grid)
    grid_np = np.asarray(x_grid)
    marg = probs.mean(0)
    std = probs.std(0)
    return grid_np, marg, probs, std


@app.cell
def _(X_train, grid_np, marg, np, plt, probs, std, y_train):
    _fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for _k in range(probs.shape[0]):
        _a1.plot(grid_np, probs[_k], color="C3", alpha=0.12, lw=0.8)
    _a1.scatter(X_train, np.asarray(y_train), s=8, color="k", alpha=0.3, zorder=5)
    _a1.axvspan(-4.0, 4.0, color="grey", alpha=0.08)
    _a1.set_ylabel("member  sigmoid(f_k(x))")
    _a1.set_title("Members agree on data (pinned to 0.5), fan out in the epistemic gap")

    _a2.plot(grid_np, marg, color="C0", lw=2, label="marginal  mean_k p")
    _a2.fill_between(grid_np, marg - std, marg + std, color="C0", alpha=0.2,
                     label="+/- epistemic std_k p")
    _a2.axhline(0.5, ls="--", color="grey", lw=1)
    _a2.axvspan(-4.0, 4.0, color="grey", alpha=0.08)
    _a2.text(0.0, 0.05, "epistemic gap:\nmarg~0.5, HIGH spread", ha="center", fontsize=8, color="C3")
    _a2.text(-5.5, 0.05, "aleatoric:\nmarg~0.5, LOW spread", ha="center", fontsize=8, color="C2")
    _a2.set_ylabel("probability")
    _a2.set_xlabel("x")
    _a2.set_ylim(-0.05, 1.05)
    _a2.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **This is the coin lesson, reincarnated in a neural net.** The marginal
    (bottom, blue) is $\approx0.5$ *everywhere* — it cannot tell you where
    the model is ignorant. But the **spread across members** does:

    - **Aleatoric** (the 50/50 clusters): the data *pins every member* to
      $p=0.5$, so the curves collapse together — low spread.
    - **Epistemic** (the empty gap): no data, so each member's frozen prior
      survives and they *fan out* — high spread.

    Same marginal, different epistemic spread — exactly the coins of Part 1.
    Now measure the **joint**.
    """)
    return


@app.cell
def _(
    jax,
    member_logit,
    mo,
    np,
    pair_alea,
    pair_epi,
    pparams,
    tparams_trained,
):
    def _p_at(x):
        # per-member prediction at x; the ensemble is our sample over z
        L = jax.vmap(lambda tp, pp: member_logit(tp, pp, x))(tparams_trained, pparams)
        return np.asarray(jax.nn.sigmoid(L))

    def _joint(pair):
        pa, pb = _p_at(pair[0]), _p_at(pair[1])
        # outcomes are independent GIVEN z, so Cov(y_a,y_b) = Cov_z(p_a,p_b)
        # -- the exact analogue of Cov = Var(theta) from the coin.
        return pa.mean(), pb.mean(), float(np.cov(pa, pb)[0, 1]), float(np.corrcoef(pa, pb)[0, 1])

    _ma, _mb, _cov_a, _corr_a = _joint(pair_alea)
    _ea, _eb, _cov_e, _corr_e = _joint(pair_epi)

    mo.md(
        f"""
        ### The joint, measured

        For each pair we read every member's prediction and compute the outcome
        covariance. Because outcomes are independent **given the index $z$**,
        $\\mathrm{{Cov}}(y_a,y_b)=\\mathrm{{Cov}}_z(p_a,p_b)$ — the exact analogue of
        $\\mathrm{{Cov}}=\\mathrm{{Var}}(\\theta)$ from the coin.

        | pair | $x_a,x_b$ | marg $p_a$ | marg $p_b$ | **joint cov** | corr |
        |---|---|---|---|---|---|
        | aleatoric | {pair_alea} | {_ma:.2f} | {_mb:.2f} | **{_cov_a:+.4f}** | {_corr_a:+.2f} |
        | epistemic | {pair_epi} | {_ea:.2f} | {_eb:.2f} | **{_cov_e:+.4f}** | {_corr_e:+.2f} |

        Same marginals ($\\approx0.5$), opposite joints. The **aleatoric** pair is
        ~independent (joint cov $\\approx0$, a *product* — like the fair coin); the
        **epistemic** pair is strongly coupled (joint cov two-to-three orders of
        magnitude larger, corr near $+1$ — like the two-point coin). The network
        learned this coupling from **single labels only** — no joint label ever
        existed.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap — the two questions, answered

    **Why a joint over $\{y_1,\dots,y_\tau\}$?** Because the marginal cannot
    separate aleatoric from epistemic uncertainty (Part 1: many priors,
    identical marginal, different joints; Part 3: marginal $\approx0.5$
    everywhere yet the spread and the joint pinpoint ignorance). The
    correlation between predictions at different inputs *is* the epistemic
    uncertainty: $\mathrm{Cov}(y_i,y_j)=\mathrm{Var}(\theta)$, generalised to
    $\mathrm{Cov}_z(p_a,p_b)$. Every downstream use that motivates the paper —
    Thompson sampling, active learning, "do I know this?" — is a question
    about *how observing one output changes belief about another*, i.e. the
    joint.

    **How is the joint learned if we only see single labels?** It is never
    observed; it is *derived* from a prior + finite data, exactly like a
    Bayesian posterior predictive. Eqn 2's $\int p_z(z)\prod_t$ is "sample a
    hypothesis, predict conditionally-independently, average" (Part 2). The
    ENN realises this: frozen random priors supply coherent cross-input
    correlation, and single-label training cancels it where data exists and
    leaves it where data is absent (Part 3).

    ### Where this meets the repo's AU/EU work

    This is a decomposition by **conditional-independence structure**:
    aleatoric $=$ the inner product (independence given $z$),
    epistemic $=$ the spread of $p_z$. The joint correlation lives in
    **observation space** ($y$-space, not $\theta$-space), so it is
    reparameterisation-invariant *by construction*. **How this variance-flavoured
    EU relates to the mutual-information EU of ave §2.2 — and why the invariance
    matters — is the subject of the companion notebook**
    `variance_vs_information_eu.py`. Worth a footnote in
    `papers/imperfect-disentanglement/`.

    *Run: `~/my_jax_env/bin/marimo edit tutorials/epistemic_nn_from_coins.py`*
    """)
    return


if __name__ == "__main__":
    app.run()
