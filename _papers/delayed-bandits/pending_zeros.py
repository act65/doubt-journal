import marimo

__generated_with = "0.17.6"
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
    from jax import random, vmap, lax
    from functools import partial
    import numpy as np
    import matplotlib.pyplot as plt
    return jax, jnp, lax, np, partial, plt, random, vmap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pending zeros: the worst case vs a posterior

    Vernade et al., *Linear Bandits with Stochastic Delayed Feedback* (arXiv:1807.02089).

    | here | paper | meaning |
    |---|---|---|
    | `theta` | $\theta$ | unknown parameter, $\lVert\theta\rVert_2 \le 1$ |
    | `arms` | $\mathcal A$ | fixed action set, $K$ vectors in $\mathbb R^d$ |
    | `X[s]` | $X_s \sim \mathcal B(A_s^\top\theta)$ | the reward action $s$ will eventually give |
    | `D[s]` | $D_s \sim \mathcal D$ | its delay (geometric, mean `mu`) |
    | `age` | $t-s$ | rounds since action $s$ was played |
    | `m` | $m$ | window: not converted within $m$ rounds $\Rightarrow$ treated as $0$ forever |
    | `F(k)` | $P(D\le k)$ | delay CDF |
    | `tau_m` | $\tau_m = F(m)$ | fraction of conversions that land inside the window |
    | `G(k)` | $F(k)/F(m)$ | $P(D\le k \mid D\le m)$ |
    | `f(t)` | $f_{t,\delta}$ | self-normalised radius, eq. (4) |
    | `pending_term` | $\sum_{s=t-m}^{t-1}\lVert A_s\rVert_{V_t^{-1}}$ | the extra width in eq. (3) / (6) |

    _Notes: not written yet._
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One pending zero

    For an action played $k$ rounds ago that has not converted, with success probability $p$:
    the probability it still converts inside the window, $\dfrac{p\,(F(m)-F(k))}{1-p\,F(k)}$,
    against the value $1$ the worst-case bound in Theorem 1 assigns it.

    _Notes: not written yet._
    """)
    return


@app.cell
def _(jnp):
    def geom_cdf(k, mu):
        # D ~ Geometric(1/mu) on {1, 2, ...}:  P(D <= k) = 1 - (1 - 1/mu)^k
        return 1.0 - (1.0 - 1.0 / mu) ** jnp.maximum(k, 0)

    def p_still_converts(p, k, m, mu):
        Fk, Fm = geom_cdf(k, mu), geom_cdf(m, mu)
        return p * (Fm - Fk) / (1.0 - p * Fk)
    return geom_cdf, p_still_converts


@app.cell
def _(jnp, p_still_converts, plt):
    _m, _mu = 100, 100
    _k = jnp.arange(0, _m + 1)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.6))
    for _p in [0.1, 0.3, 0.6, 0.9]:
        _ax.plot(_k, p_still_converts(_p, _k, _m, _mu), label=f"p = {_p}")
    _ax.axhline(1.0, color="k", ls="--", lw=1, label="worst case (Theorem 1)")
    _ax.set_xlabel("age k of the pending zero")
    _ax.set_ylabel("P(converts within window | zero so far)")
    _ax.set_title(f"geometric delays, mean {_mu}, window m = {_m}")
    _ax.set_ylim(0, 1.05)
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The environment

    Fixed action set of $K$ unit vectors in the positive orthant and $\theta$ a positive unit
    vector, so $a^\top\theta\in[0,1]$. Rewards, delays and the uniforms that decide $X_s$ are
    drawn once per run and shared by every learner (common random numbers), so differences
    between learners are not sampling noise in the environment.

    Observation rule used throughout: at decision time $t$, action $s<t$ shows $Y_{s,t}=1$ iff
    $X_s=1$ and $D_s\le t-s$.
    """)
    return


@app.cell
def _(jnp, random):
    def make_env(key, d, K, T, mu):
        k1, k2, k3, k4 = random.split(key, 4)
        arms = jnp.abs(random.normal(k1, (K, d)))
        arms = arms / jnp.linalg.norm(arms, axis=1, keepdims=True)
        theta = jnp.abs(random.normal(k2, (d,)))
        theta = theta / jnp.linalg.norm(theta)
        U = random.uniform(k3, (T,))                          # X_t = 1{U_t < A_t^T theta}
        D = random.geometric(k4, 1.0 / mu, (T,))              # delays, support {1, 2, ...}
        return dict(arms=arms, theta=theta, U=U, D=D)
    return (make_env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Five learners

    All are UCB on a regularised least-squares estimate with radius $2f_{t,\delta}$; they differ
    only in how a pending zero enters the regression. The `scale on 2f` slider multiplies the
    $2f_{t,\delta}$ part of every learner's width (not `otf`'s pending term); $1$ is the paper's
    radius.

    | learner | regression target | pending zero | extra width |
    |---|---|---|---|
    | `otf` (paper, Alg. 1) | $\tau_m\theta$ | $0$, weight $1$ | $\sum_{s=t-m}^{t-1}\lVert A_s\rVert_{V_t^{-1}}$ |
    | `naive` | $\tau_m\theta$ | $0$, weight $1$ | none |
    | `oracle_G` | $\tau_m\theta$ | design scaled by $G(\text{age})$, $G$ known | none |
    | `learned_G` | $\tau_m\theta$ | design scaled by $\hat G(\text{age})$ | none |
    | `decay` (thoughts.md) | — | $\hat Y = e^{-\beta\,\text{age}\,(1-Y)}$ | none |

    `learned_G` estimates $G$ from the delays of conversions whose actions have matured
    (age $\ge m$): those are an uncensored sample of $D \mid D\le m$. Until the first one
    arrives it uses $\hat G=1$, i.e. it behaves like `naive`.

    `decay` uses the formula from `thoughts.md` as written (no window), with $\beta$ a free
    parameter.
    """)
    return


@app.cell
def _(geom_cdf, jax, jnp, lax, make_env, partial, random, vmap):
    LEARNERS = ["otf", "naive", "oracle_G", "learned_G", "decay"]

    def f_radius(t, d, lam, delta):
        # eq. (4)
        return jnp.sqrt(lam) + jnp.sqrt(2 * jnp.log(1 / delta)
                                        + d * jnp.log((d * lam + t) / (d * lam)))

    @partial(jax.jit, static_argnames=("learner", "T", "m"))
    def run(env, learner, T, m, mu, lam=1.0, delta=0.05, beta=None, scale=1.0):
        arms, theta, U, D = env["arms"], env["theta"], env["U"], env["D"]
        K, d = arms.shape
        beta = 1.0 / mu if beta is None else beta
        s_idx = jnp.arange(T)
        best = jnp.max(arms @ theta)

        def step(carry, t):
            A, X, hist = carry                                # hist[j] = #matured conversions with delay j
            age = t - s_idx
            played = s_idx < t
            seen = X * (D <= jnp.minimum(age, m))             # windowed observation  Y~_{s,t}
            # weight on each row of the design
            if learner == "oracle_G":
                w = geom_cdf(jnp.minimum(age, m), mu) / geom_cdf(m, mu)
            elif learner == "learned_G":
                tot = hist.sum()
                cdf = jnp.where(tot > 0, jnp.cumsum(hist) / jnp.maximum(tot, 1), 1.0)
                w = cdf[jnp.clip(jnp.minimum(age, m), 0, m)]
            else:
                w = jnp.ones(T)
            if learner == "decay":
                y = jnp.exp(-beta * age * (1.0 - X * (D <= age)))
            else:
                y = seen
            w = w * played
            V = lam * jnp.eye(d) + (A * (w ** 2)[:, None]).T @ A
            Vinv = jnp.linalg.inv(V)
            theta_hat = Vinv @ (A.T @ (w * y))
            f = f_radius(t + 1.0, d, lam, delta)
            row_norms = jnp.sqrt(jnp.einsum("si,ij,sj->s", A, Vinv, A))
            pending = played & (age <= m)
            pending_term = jnp.sum(jnp.where(pending, row_norms, 0.0))
            alpha = scale * 2 * f + (pending_term if learner == "otf" else 0.0)
            arm_norms = jnp.sqrt(jnp.einsum("ki,ij,kj->k", arms, Vinv, arms))
            a_idx = jnp.argmax(arms @ theta_hat + alpha * arm_norms)
            a = arms[a_idx]
            A = A.at[t].set(a)
            X = X.at[t].set((U[t] < a @ theta).astype(jnp.float64))
            # the action played m rounds ago has just matured
            s_mat = t - m
            ok = (s_mat >= 0) & (X[jnp.maximum(s_mat, 0)] > 0) & (D[jnp.maximum(s_mat, 0)] <= m)
            hist = hist.at[jnp.clip(D[jnp.maximum(s_mat, 0)], 0, m)].add(ok.astype(jnp.float64))
            out = dict(regret=best - a @ theta, f=f, pending_term=pending_term)
            return (A, X, hist), out

        init = (jnp.zeros((T, d)), jnp.zeros(T), jnp.zeros(m + 1))
        _, out = lax.scan(step, init, jnp.arange(T))
        return out

    def run_many(key, n_runs, learner, d, K, T, m, mu, **kw):
        envs = vmap(lambda k: make_env(k, d, K, T, mu))(random.split(key, n_runs))
        return vmap(lambda e: run(e, learner, T, m, mu, **kw))(envs)
    return LEARNERS, run_many


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Regret

    The three $(m,\mu)$ settings of the paper's Figure 1, with $d=5$, $K=10$. Mean cumulative
    regret over runs, shaded $\pm$ one standard error.

    _Notes: not written yet._
    """)
    return


@app.cell
def _(mo):
    T_slider = mo.ui.slider(500, 3000, step=500, value=2000, label="T")
    runs_slider = mo.ui.slider(5, 40, step=5, value=10, label="runs")
    scale_slider = mo.ui.slider(0.05, 1.0, step=0.05, value=1.0, label="scale on 2f")
    mo.hstack([T_slider, runs_slider, scale_slider])
    return T_slider, runs_slider, scale_slider


@app.cell
def _(LEARNERS, T_slider, jnp, random, run_many, runs_slider, scale_slider):
    SETTINGS = [(100, 100), (500, 100), (100, 500)]           # (m, mu)
    results = {}
    for _m, _mu in SETTINGS:
        for _name in LEARNERS:
            _out = run_many(random.PRNGKey(0), runs_slider.value, _name,
                            d=5, K=10, T=T_slider.value, m=_m, mu=_mu,
                            scale=scale_slider.value)
            results[(_m, _mu, _name)] = {k: jnp.asarray(v) for k, v in _out.items()}
    return SETTINGS, results


@app.cell
def _(LEARNERS, SETTINGS, geom_cdf, np, plt, results):
    _fig, _axes = plt.subplots(1, len(SETTINGS), figsize=(13, 3.8), sharey=False)
    for _ax, (_m, _mu) in zip(_axes, SETTINGS):
        for _name in LEARNERS:
            _r = np.cumsum(np.asarray(results[(_m, _mu, _name)]["regret"]), axis=1)
            _mean = _r.mean(0)
            _se = _r.std(0) / np.sqrt(_r.shape[0])
            _x = np.arange(1, _r.shape[1] + 1)
            _ax.plot(_x, _mean, label=_name)
            _ax.fill_between(_x, _mean - _se, _mean + _se, alpha=0.2)
        _tau = float(geom_cdf(_m, _mu))
        _ax.set_title(f"m = {_m}, mu = {_mu}  (tau_m = {_tau:.3f})")
        _ax.set_xlabel("t")
    _axes[0].set_ylabel("cumulative regret")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Where `otf`'s width comes from

    The two parts of `otf`'s confidence width $\alpha_t = 2f_{t,\delta} + \sum_{s=t-m}^{t-1}
    \lVert A_s\rVert_{V_t^{-1}}$ over time, mean over runs.

    _Notes: not written yet._
    """)
    return


@app.cell
def _(SETTINGS, np, plt, results):
    _fig, _axes = plt.subplots(1, len(SETTINGS), figsize=(13, 3.4))
    for _ax, (_m, _mu) in zip(_axes, SETTINGS):
        _o = results[(_m, _mu, "otf")]
        _x = np.arange(1, np.asarray(_o["f"]).shape[1] + 1)
        _ax.plot(_x, 2 * np.asarray(_o["f"]).mean(0), label="2 f_t")
        _ax.plot(_x, np.asarray(_o["pending_term"]).mean(0), label="pending term")
        _ax.set_yscale("log")
        _ax.set_title(f"m = {_m}, mu = {_mu}")
        _ax.set_xlabel("t")
    _axes[0].set_ylabel("width")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
