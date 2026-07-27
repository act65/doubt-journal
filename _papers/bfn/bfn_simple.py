import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import random, vmap
    import numpy as np
    import matplotlib.pyplot as plt

    return jnp, np, plt, random, vmap


@app.cell
def _(mo):
    mo.md(r"""
    # Bayesian Flow Networks, from Bayesian fundamentals

    Everything here is one sentence:

    > **We keep a belief about the clean data $x$ — a posterior built from noisy observations of
    > $x$ under a naive prior — and a "brain" that corrects that naive posterior into a smart
    > estimate using the true data prior. Generation bootstraps: guess $x$, observe our own
    > guess, update the belief, repeat.**

    We'll drop the paper's notation ($\theta,\mu,\rho,\gamma,\beta,p_I,p_S,p_R,p_O,\Psi$) and use
    plain Bayesian pieces:

    | piece | what it is | paper name |
    |---|---|---|
    | `belief = (m, A)` | a noisy estimate `m` of $x$ with total precision `A` (so `m ~ N(x, 1/A)`) | input dist / $\theta$ |
    | `observe(x, a)` | a noisy measurement `y ~ N(x, 1/a)` | sender |
    | `update(belief, y, a)` | fold a measurement in (conjugate Bayes) | Bayesian update |
    | `denoise(m, A)` | evidence + **true prior** → smart estimate $\hat x$ | network + output dist |

    The only "learning" is inside `denoise`. For a known toy prior we can write the optimal
    `denoise` in closed form, so **no training is needed** and every line is transparent.
    """)
    return


@app.cell
def _(jnp, random):
    # ---- the true data prior: a mixture of 3 Gaussians (deliberately multimodal) ----
    w = jnp.array([0.4, 0.35, 0.25])                       # mixture weights
    mp = jnp.array([[-0.5, -0.4], [0.5, -0.3], [0.0, 0.55]])   # component means
    Sp = jnp.stack([0.02 * jnp.eye(2), 0.015 * jnp.eye(2), 0.025 * jnp.eye(2)])  # covariances
    K = 3
    Sp_inv = jnp.linalg.inv(Sp)
    global_mean = w @ mp                                   # E[x] under the prior

    def sample_data(key, n):
        _ck, _nk = random.split(key)
        _comp = random.choice(_ck, K, (n,), p=w)
        _L = jnp.linalg.cholesky(Sp)
        return mp[_comp] + jnp.einsum("nij,nj->ni", _L[_comp], random.normal(_nk, (n, 2)))

    return K, Sp, Sp_inv, global_mean, mp, sample_data, w


@app.cell
def _(np, plt, random, sample_data):
    _pts = np.array(sample_data(random.PRNGKey(0), 3000))
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _ax.scatter(_pts[:, 0], _pts[:, 1], s=4, alpha=0.3, c="C0")
    _ax.set_title("the data we want to generate  (3-mode mixture)")
    _ax.set_xlim(-1, 1); _ax.set_ylim(-1, 1); _ax.set_aspect("equal")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The belief, and two Bayesian primitives

    A **belief** is just a running noisy estimate of $x$ plus how sure we are:
    `belief = (m, A)` meaning $m \sim \mathcal N(x,\,1/A)$. Big `A` = confident.

    Two primitives, both pure Bayesian bookkeeping — no neural net, no prior baked in:

    - **`observe(x, a)`** returns a measurement $y \sim \mathcal N(x, 1/a)$ (accuracy `a`).
    - **`update((m,A), y, a)`** folds a measurement into the belief. Combining independent
      Gaussian evidence = **precision-weighted average**, precisions add:

    $$
    A' = A + a, \qquad m' = \frac{A\,m + a\,y}{A'}
    $$

    Start from *no evidence*: `A = 0`. (The very first `update` just sets `m = y`.)
    """)
    return


@app.cell
def _(random):
    def observe(key, x, a):
        return x + random.normal(key, x.shape) / (a ** 0.5)   # y ~ N(x, 1/a)

    def update(m, A, y, a):
        A_new = A + a
        m_new = (A * m + a * y) / A_new                        # precision-weighted average
        return m_new, A_new

    return observe, update


@app.cell
def _(jnp, np, observe, plt, random, update):
    # Demo: accumulate REAL measurements of a fixed x. Belief mean -> x, uncertainty shrinks.
    def _accumulate():
        _x = jnp.array([0.5, -0.3])
        _key = random.PRNGKey(4)
        _m = jnp.zeros(2); _A = 0.0
        _ms = []; _As = []
        for _ in range(15):
            _key, _k = random.split(_key)
            _y = observe(_k, _x, 3.0)               # each measurement has precision 3
            _m, _A = update(_m, _A, _y, 3.0)
            _ms.append(np.array(_m)); _As.append(float(_A))
        return _x, np.array(_ms), np.array(_As)

    _x, _ms, _As = _accumulate()
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 4))
    _ax[0].plot(_ms[:, 0], _ms[:, 1], "-o", ms=3, c="C0", label="belief mean m")
    _ax[0].scatter(*_x, c="k", marker="*", s=200, label="true x", zorder=5)
    for _i in range(0, 15, 3):                       # uncertainty circles radius 1/sqrt(A)
        _c = plt.Circle(_ms[_i], 1 / np.sqrt(_As[_i]), fill=False, color="C0", alpha=0.3)
        _ax[0].add_patch(_c)
    _ax[0].legend(fontsize=8); _ax[0].set_aspect("equal"); _ax[0].set_title("belief mean → x")
    _ax[1].plot(_As, "-o", ms=3, c="C1"); _ax[1].set_xlabel("# measurements")
    _ax[1].set_ylabel("total precision A"); _ax[1].set_title("confidence grows")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The brain: naive evidence + true prior → smart estimate

    The belief `(m, A)` is a posterior over $x$ under a **naive** assumption (each coordinate
    independent, no idea of the data's shape). That's mathematically exact for *accumulating
    measurements*, but it's a dumb summary — its "best guess" is just `m`.

    **`denoise(m, A)` is the correction.** It reads the belief as evidence
    $\mathcal N(x \mid m, 1/A)$ and combines it with the **true** data prior $p_{\text{data}}$,
    returning the posterior mean $\hat x = \mathbb E[x \mid m, A]$. For our mixture prior this is
    closed form (a responsibility-weighted blend of per-mode posteriors). Two limits:

    - `A = 0` (no evidence): $\hat x = \mathbb E_{p_{\text{data}}}[x]$ — the prior mean.
    - `A → ∞` (sharp evidence): $\hat x \to m$ — trust the measurement.

    In between, it **pulls the naive estimate toward the nearest mode** — knowledge the belief
    alone doesn't have. This single function is the network + output distribution.
    """)
    return


@app.cell
def _(K, Sp, Sp_inv, global_mean, jnp, mp, w):
    def denoise(m, A, eps=1e-8):
        # posterior mean of x under the true mixture prior, given evidence  m ~ N(x, 1/A)
        def _log_gauss(x, mean, cov):
            _d = x - mean
            _L = jnp.linalg.cholesky(cov)
            _s = jnp.linalg.solve(_L, _d)
            return -0.5 * (2 * jnp.log(2 * jnp.pi) + 2 * jnp.sum(jnp.log(jnp.diag(_L))) + _s @ _s)

        v = 1.0 / jnp.maximum(A, eps)                     # measurement variance
        # responsibility: how much each mode "explains" the estimate m  (stable softmax)
        logr = jnp.stack([jnp.log(w[k]) + _log_gauss(m, mp[k], Sp[k] + v * jnp.eye(2))
                          for k in range(K)])
        r = jnp.exp(logr - jnp.max(logr)); r = r / jnp.sum(r)
        # per-mode Gaussian posterior mean, then blend by responsibility
        means = jnp.stack([jnp.linalg.inv(Sp_inv[k] + A * jnp.eye(2))
                           @ (Sp_inv[k] @ mp[k] + A * m) for k in range(K)])
        xhat = jnp.sum(r[:, None] * means, axis=0)
        return jnp.where(A < eps, global_mean, xhat)      # no evidence -> prior mean

    return (denoise,)


@app.cell
def _(denoise, mp, np, plt, random):
    # Demo: for a true x near mode 0, sample noisy estimates m at growing precision A,
    # and show how denoise corrects m (grey) into xhat (green), pulling toward the mode.
    def _brain_demo():
        _x = mp[0]
        _As = [0.5, 3.0, 30.0]
        _key = random.PRNGKey(2)
        _rows = []
        for _A in _As:
            _key, _k = random.split(_key)
            _m = _x + random.normal(_k, (2,)) / np.sqrt(_A)
            _rows.append((_A, np.array(_m), np.array(denoise(_m, _A))))
        return _rows

    _rows = _brain_demo()
    _fig, _ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for _i, (_A, _m, _xh) in enumerate(_rows):
        _a = _ax[_i]
        _a.scatter(np.array(mp)[:, 0], np.array(mp)[:, 1], c="k", marker="+", s=120, label="modes")
        _a.scatter(*_m, c="0.5", s=60, label="m (naive estimate)")
        _a.scatter(*_xh, c="C2", marker="P", s=90, label="x̂ = denoise(m,A)")
        _a.annotate("", xy=_xh, xytext=_m, arrowprops=dict(arrowstyle="->", color="C2"))
        _a.set_title(f"A = {_A}"); _a.set_xlim(-1, 1); _a.set_ylim(-1, 1); _a.set_aspect("equal")
        if _i == 0: _a.legend(fontsize=7, loc="upper left")
    _fig.suptitle("the brain corrects the naive estimate using the true prior")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Schedule: how fast to accumulate evidence

    During generation we add measurements over `n` steps, growing total precision from `0` to
    some large `P_end`. Any smooth increase works; we grow it **exponentially** (equal *ratio*
    per step), which spreads the "work" evenly:
    $$A(s) = (1 + P_{\text{end}})^{\,s} - 1,\qquad s \in [0,1].$$
    Step `i` adds `a_i = A(i/n) − A((i-1)/n)`. Early steps add tiny precision (very noisy
    measurements that just *seed* which mode we drift toward); late steps add lots (measurements
    that pin the sample down). *(This is the paper's schedule with `P_end = σ₁⁻²`.)*
    """)
    return


@app.cell
def _(np, plt):
    P_end = 1e4
    def total_precision(s):
        return (1.0 + P_end) ** s - 1.0

    _s = np.linspace(0, 1, 200)
    _fig, _ax = plt.subplots(figsize=(5, 2.6))
    _ax.plot(_s, [total_precision(v) for v in _s], c="C1")
    _ax.set_yscale("log"); _ax.set_xlabel("progress s"); _ax.set_ylabel("total precision A(s)")
    _ax.set_title("evidence accumulated by progress s")
    _fig
    return (total_precision,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Generation = bootstrap

    No real $x$ exists, so we let the brain stand in for it. Each step:

    1. **guess**: `xhat = denoise(m, A)` — best estimate of the clean data from the belief;
    2. **observe our own guess**: `y = observe(xhat, a_i)` — a noisy measurement of `xhat`;
    3. **update**: fold `y` into the belief.

    The belief sharpens, the guesses stabilise onto a self-consistent sample. Randomness in the
    early (noisy) measurements is what makes different runs land in different modes.
    """)
    return


@app.cell
def _(denoise, jnp, observe, random, total_precision, update, vmap):
    def generate(key, n=100):
        m = jnp.zeros(2); A = 0.0
        for i in range(1, n + 1):
            a_i = total_precision(i / n) - total_precision((i - 1) / n)
            xhat = denoise(m, A)                  # 1. guess
            key, k = random.split(key)
            y = observe(k, xhat, a_i)             # 2. observe the guess
            m, A = update(m, A, y, a_i)           # 3. update belief
        return denoise(m, A)

    def generate_trace(key, n=40):
        m = jnp.zeros(2); A = 0.0
        ms = [m]; xs = []; ys = []
        for i in range(1, n + 1):
            a_i = total_precision(i / n) - total_precision((i - 1) / n)
            xhat = denoise(m, A)
            key, k = random.split(key)
            y = observe(k, xhat, a_i)
            m, A = update(m, A, y, a_i)
            xs.append(xhat); ys.append(y); ms.append(m)
        return jnp.stack(ms), jnp.stack(xs), jnp.stack(ys)

    def generate_many(seed=0, count=4000, n=100):
        return vmap(lambda k: generate(k, n))(random.split(random.PRNGKey(seed), count))

    return generate_many, generate_trace


@app.cell
def _(generate_trace, np, plt, random, sample_data):
    _bg = np.array(sample_data(random.PRNGKey(9), 2000))
    _ms, _xs, _ys = generate_trace(random.PRNGKey(7), n=40)
    _ms, _xs, _ys = np.array(_ms), np.array(_xs), np.array(_ys)
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(_bg[:, 0], _bg[:, 1], s=4, alpha=0.12, c="0.6")
    _ax.plot(_ms[:, 0], _ms[:, 1], "-o", c="C0", ms=3, lw=1, label="belief mean m")
    _ax.scatter(_xs[:, 0], _xs[:, 1], c="C2", marker="P", s=30, label="guesses x̂")
    _ax.scatter(_ys[:, 0], _ys[:, 1], c="C3", marker="x", s=20, alpha=0.5, label="measurements y")
    _ax.scatter(*_ms[0], c="k", marker="s", s=70, label="start (no evidence)", zorder=5)
    _ax.scatter(*_xs[-1], c="red", marker="*", s=260, label="final sample", zorder=6)
    _ax.set_xlim(-1, 1); _ax.set_ylim(-1, 1); _ax.set_aspect("equal")
    _ax.legend(fontsize=7, loc="upper left"); _ax.set_title("one generation run")
    _fig
    return


@app.cell
def _(generate_many, mp, np, plt, w):
    _samp = np.array(generate_many())
    _d = np.linalg.norm(_samp[:, None, :] - np.array(mp)[None], axis=-1)
    _assign = _d.argmin(1)
    _frac = np.array([np.mean(_assign == k) for k in range(3)])
    _fig, _ax = plt.subplots(figsize=(4.4, 4.4))
    _ax.scatter(_samp[:, 0], _samp[:, 1], s=4, alpha=0.25, c="C0")
    _ax.scatter(np.array(mp)[:, 0], np.array(mp)[:, 1], c="red", marker="+", s=140)
    _ax.set_xlim(-1, 1); _ax.set_ylim(-1, 1); _ax.set_aspect("equal")
    _ax.set_title("4000 generated samples")
    print("mode fractions (generated):", np.round(_frac, 3), " target weights:", np.array(w))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why this is the whole architecture

    Every BFN piece is here, renamed:

    - **belief `(m, A)`** = input distribution $\theta=\{\mu,\rho\}$ (we use the un-scaled mean
      `m`; the paper's $\mu=\gamma\,m$ just re-scales it, and $\rho = A+1$).
    - **`observe`** = sender; **`update`** = Bayesian update; the *schedule* = accuracies $\alpha_i$.
    - **`denoise`** = network $\Psi$ + output distribution: it maps the naive belief to a smart
      $\hat x$. We plugged in the closed-form optimal `denoise`; a real BFN **learns** it by the
      simplest possible objective — noise a data point to a random precision `A` and regress the
      clean point back:
      $$\min\ \mathbb E_{x\sim\text{data},\,A\sim\text{schedule},\,m\sim\mathcal N(x,1/A)}
      \big\|x - \texttt{denoise}(m, A)\big\|^2 .$$
      (That's it — the entire training loss. It's a denoiser, exactly as you suspected; the
      Bayesian scaffolding is what lets the *same* recipe extend to discrete data.)
    - **generation** = the bootstrap loop, sampling each measurement from the *receiver*
      (`observe(xhat, ·)`) because we have no true $x$.

    And the payoff you can see above: a purely Gaussian belief could never be multimodal, yet
    generation reproduces all three modes with the right weights — because `denoise` folds the
    true prior back in at every step. That division of labour (exact Bayesian accumulation +
    learned prior-aware correction) is the entire idea.
    """)
    return


if __name__ == "__main__":
    app.run()
