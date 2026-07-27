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
    # From a Gaussian to an Epistemic Neural Network

    A self-contained, bottom-up tutorial. We climb a **ladder**: each rung is one
    small idea with a picture, and by the top we have built an Epistemic Neural
    Network (Osband et al. 2021) from scratch and understand exactly what it is
    doing and why.

    The whole tutorial rests on one running example — a Gaussian measurement

    $$y = \underbrace{\mu}_{\text{unknown mean}} + \underbrace{\varepsilon}_{\varepsilon\sim\mathcal N(0,\sigma^2)\ \text{noise}} .$$

    Two kinds of uncertainty live here, and telling them apart is the whole game:

    - **Aleatoric** ($\sigma^2$): irreducible measurement noise. More data will
      *not* remove it.
    - **Epistemic** ($s^2 = \mathrm{Var}(\mu)$): our ignorance about $\mu$. More
      data *will* remove it.

    **The ladder**

    | rung | idea |
    |---|---|
    | 1 | one measurement can't separate the two |
    | 2 | *two* measurements can — the **joint** |
    | 3 | why: $\mathrm{Cov}(y_i,y_j)=\mathrm{Var}(\mu)$ |
    | 4 | aleatoric is *independent*, epistemic is *shared* |
    | 5 | *conditioning* on data shrinks the epistemic part |
    | 6 | the sampling recipe (this is eqn 2) |
    | 7 | from a *number* to a *function* |
    | 8 | a prior over functions (randomized priors) |
    | 9 | train: prior → posterior over functions |
    | 10 | read off aleatoric vs epistemic |
    | 11 | the joint, on the (ensemble) network |
    | — | *Part II — the real architecture* |
    | 12 | the catch: rungs 8–11 were K full networks |
    | 13 | the **epinet** — one shared body, z as an input |
    | 14 | sample many z from one base pass |
    | 15 | the joint, on the epinet |

    Rungs 1–11 build the simplest ENN (an ensemble); rungs 12–15 build the
    paper's efficient **epinet**. Needs only `jax`, `optax`, `numpy`,
    `matplotlib`.
    """)
    return


@app.cell(hide_code=True)
def _():
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # tiny CPU demo; skip GPU init

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import optax

    return jax, jnp, np, optax, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 1 — One measurement can't tell noise from ignorance

    Suppose a single measurement has predictive distribution $\mathcal N(0,2)$.
    Where did that variance $2$ come from? Two completely different worlds give
    the *identical* prediction:

    - **World A** — a **noisy** instrument ($\sigma^2=2$) measuring a **known**
      mean ($s^2=0$). All the spread is aleatoric.
    - **World B** — a **noiseless** instrument ($\sigma^2=0$) measuring an
      **unknown** mean with $\mathrm{Var}(\mu)=2$. All the spread is epistemic.

    A single observation is $\mathcal N(m,\sigma^2+s^2)$ — it sees only the
    **sum**. Below, the two predictive curves lie exactly on top of each other.
    """)
    return


@app.cell
def _(np, plt):
    _grid = np.linspace(-5, 5, 400)
    _pdf = np.exp(-_grid ** 2 / (2 * 2.0)) / np.sqrt(2 * np.pi * 2.0)   # N(0, 2)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_grid, _pdf, color="C2", lw=5, alpha=0.5, label="World A: sigma²=2, s²=0 (all aleatoric)")
    _ax.plot(_grid, _pdf, color="C3", lw=2, ls="--", label="World B: sigma²=0, s²=2 (all epistemic)")
    _ax.set_xlabel("y")
    _ax.set_ylabel("predictive density")
    _ax.set_title("Same single-measurement prediction N(0, 2) — the marginal is blind to the split")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 2 — Two measurements *can* (the joint)

    Take **two** measurements $y_1,y_2$ of the same quantity. Conditional on
    $\mu$ they are independent $\mathcal N(\mu,\sigma^2)$; averaging over the
    uncertain $\mu\sim\mathcal N(m,s^2)$ couples them:

    $$\begin{pmatrix}y_1\\y_2\end{pmatrix}\sim
    \mathcal N\!\left(\begin{pmatrix}m\\m\end{pmatrix},
    \begin{pmatrix}\sigma^2+s^2 & s^2\\ s^2 & \sigma^2+s^2\end{pmatrix}\right).$$

    Hold the marginal fixed (total variance $V=\sigma^2+s^2=2$) and slide the
    **epistemic fraction** $\rho=s^2/V$. The marginal never changes; the joint
    rotates from a **circle** (independent noise) to a **line** $y_1=y_2$ (a
    shared unknown — one measurement pins the other).
    """)
    return


@app.cell
def _(mo):
    rho = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.5,
                       label="epistemic fraction  rho = s² / (sigma²+s²)")
    rho
    return (rho,)


@app.cell
def _(np, plt, rho):
    from matplotlib.patches import Ellipse

    _V = 2.0
    _s2 = rho.value * _V          # epistemic
    _sig2 = (1 - rho.value) * _V  # aleatoric
    _rng = np.random.default_rng(0)
    _mu = np.sqrt(_s2) * _rng.standard_normal(1500)               # sample mu
    _y1 = _mu + np.sqrt(_sig2) * _rng.standard_normal(1500)       # add indep noise
    _y2 = _mu + np.sqrt(_sig2) * _rng.standard_normal(1500)

    _fig, _ax = plt.subplots(figsize=(4.6, 4.6))
    _ax.scatter(_y1, _y2, s=6, alpha=0.2, color="C0")
    _ax.plot([-5, 5], [-5, 5], ls="--", color="grey", lw=1)
    _Sigma = np.array([[_sig2 + _s2, _s2], [_s2, _sig2 + _s2]])
    _w, _v = np.linalg.eigh(_Sigma)
    _ang = np.degrees(np.arctan2(_v[1, -1], _v[0, -1]))
    _ax.add_patch(Ellipse((0, 0), 2 * 2.4 * np.sqrt(_w[-1]), 2 * 2.4 * np.sqrt(max(_w[0], 1e-9)),
                          angle=_ang, fill=False, color="C3", lw=2))
    _ax.set_xlim(-4.5, 4.5)
    _ax.set_ylim(-4.5, 4.5)
    _ax.set_aspect("equal")
    _ax.set_xlabel("$y_1$")
    _ax.set_ylabel("$y_2$")
    _corr = _s2 / _V
    _ax.set_title(f"marginal Var=2 (fixed).   cov = s² = {_s2:.2f},   corr = {_corr:.2f}", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Drag the slider. At $\rho=0$ (circle) knowing $y_1$ tells you nothing about
    $y_2$. At $\rho=1$ (line) knowing $y_1$ tells you $y_2$ *exactly* — there is
    no noise, only a shared unknown. **The off-diagonal covariance is the
    epistemic uncertainty**, and the marginal cannot see it.

    ## Rung 3 — Why the off-diagonal *is* the epistemic variance

    One line, exact for the Gaussian — the **law of total covariance** ($i\neq j$):

    $$\mathrm{Cov}(y_i,y_j)=\underbrace{\mathbb{E}_\mu[\mathrm{Cov}(y_i,y_j\mid\mu)]}_{=\,0\ \text{(indep. given }\mu)}
    +\underbrace{\mathrm{Cov}\!\big(\mathbb{E}[y_i\mid\mu],\mathbb{E}[y_j\mid\mu]\big)}_{=\,\mathrm{Cov}(\mu,\mu)=\mathrm{Var}(\mu)=s^2}.$$

    Conditional independence zeroes the noise term; the shared mean supplies the
    rest. The **diagonal** ($i=j$) keeps both — $\mathrm{Var}(y_i)=\sigma^2+s^2$
    (law of total variance). So: *diagonal = total, off-diagonal = pure
    epistemic.* Everything else in this notebook is a consequence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 4 — Aleatoric is *independent*, epistemic is *shared*

    The $\tau\times\tau$ joint covariance is $\Sigma=\sigma^2 I+s^2\,\mathbf{1}\mathbf{1}^\top$.
    Read its structure: aleatoric $\sigma^2$ sits on the **diagonal only** (each
    observation's noise is its own), while epistemic $s^2$ fills **every
    off-diagonal** (all observations share the one unknown $\mu$).

    Now average $\tau$ observations that share the same $\mu$. Their *independent*
    noises cancel at rate $1/\tau$; the *shared* offset is identical in every term
    and stays:

    $$\mathrm{Var}(\bar y_\tau)=\frac{\sigma^2}{\tau}+s^2\ \xrightarrow{\ \tau\to\infty\ }\ s^2 .$$

    So aleatoric is the part you can wash out by averaging many independent draws;
    epistemic is the **common mode** that averaging leaves behind — because every
    draw is wrong in the *same* direction.

    > ⚠️ **Averaging is not learning.** Here we average predictions while holding
    > our belief about $\mu$ *fixed* (its variance stays $s^2$). If instead we
    > **condition** on those observations — use them to *update* the posterior —
    > then the epistemic $s^2$ itself shrinks. That is Rung 5. Averaging draws
    > under a fixed belief ≠ conditioning on draws to sharpen the belief.
    """)
    return


@app.cell
def _(np, plt):
    _taus = np.arange(1, 51)
    _fig, _ax = plt.subplots(figsize=(6.6, 4.0))
    for _name, _sig2, _s2, _c in [("all aleatoric (s²=0)", 2.0, 0.0, "C2"),
                                  ("mixed (s²=1)", 1.0, 1.0, "C0"),
                                  ("epistemic-heavy (s²=1.6)", 0.4, 1.6, "C3")]:
        _ax.plot(_taus, _sig2 / _taus + _s2, "-o", ms=2, color=_c, label=_name)
        _ax.axhline(_s2, ls=":", color=_c, lw=0.8)
    _ax.set_xlabel("tau  (number of measurements averaged)")
    _ax.set_ylabel("Var of the average  =  sigma²/tau + s²")
    _ax.set_title("Independent (aleatoric) part averages away; shared (epistemic) part remains (dotted = s²)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 5 — *Conditioning* on data shrinks the epistemic part

    This is the other half of Rung 4: not averaging draws under a fixed belief,
    but **conditioning** on them to sharpen the belief. Where does $s^2$ come
    from? A posterior. Start from a prior
    $\mu\sim\mathcal N(0,s_0^2)$ and known noise $\sigma^2$; after $n$
    observations the posterior variance is

    $$s_n^2=\Big(\tfrac1{s_0^2}+\tfrac{n}{\sigma^2}\Big)^{-1}\ \sim\ \frac{\sigma^2}{n}.$$

    Slide $n$ below. On the left the posterior over $\mu$ tightens; on the right
    the **joint of two future measurements collapses from an ellipse back to a
    circle** — the epistemic coupling $s_n^2$ vanishes $\sim 1/n$, leaving only
    the aleatoric noise floor $\sigma^2$.
    """)
    return


@app.cell
def _(mo):
    n_slider = mo.ui.slider(start=0, stop=100, step=1, value=3, label="n  (observations seen)")
    n_slider
    return (n_slider,)


@app.cell
def _(n_slider, np, plt):
    _sigma2, _s0_2 = 1.0, 8.0
    _n = n_slider.value
    _sn2 = 1.0 / (1.0 / _s0_2 + _n / _sigma2)     # posterior Var(mu) = joint off-diagonal

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.8))

    _g = np.linspace(-6, 6, 400)
    _a1.plot(_g, np.exp(-_g ** 2 / (2 * _s0_2)) / np.sqrt(2 * np.pi * _s0_2),
             color="grey", ls="--", lw=1, label="prior")
    _a1.plot(_g, np.exp(-_g ** 2 / (2 * _sn2)) / np.sqrt(2 * np.pi * _sn2),
             color="C3", lw=2, label=f"posterior (n={_n})")
    _a1.set_xlabel("mu")
    _a1.set_ylabel("density")
    _a1.set_title(f"posterior over mu:  s_n² = {_sn2:.3f}")
    _a1.legend(fontsize=8)

    _rng = np.random.default_rng(0)
    _mu = np.sqrt(_sn2) * _rng.standard_normal(1500)
    _y1 = _mu + np.sqrt(_sigma2) * _rng.standard_normal(1500)
    _y2 = _mu + np.sqrt(_sigma2) * _rng.standard_normal(1500)
    _a2.scatter(_y1, _y2, s=6, alpha=0.2, color="C0")
    _a2.plot([-6, 6], [-6, 6], ls="--", color="grey", lw=1)
    _a2.set_xlim(-5, 5)
    _a2.set_ylim(-5, 5)
    _a2.set_aspect("equal")
    _a2.set_xlabel("$y_1$")
    _a2.set_ylabel("$y_2$")
    _a2.set_title(f"joint of 2 future obs:  cov = s_n² = {_sn2:.3f}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 6 — The sampling recipe (this is literally eqn 2)

    Every joint we have drawn is *defined* by an integral — marginalising the
    shared unknown $\mu$ out of the product of per-observation likelihoods:

    $$P(y_1,y_2)=\int \underbrace{\mathcal N(y_1\mid\mu,\sigma^2)\,\mathcal N(y_2\mid\mu,\sigma^2)}_{\text{indep. given }\mu}\ p(\mu)\,d\mu .$$

    For a Gaussian this integral has a closed form (it *is* the bivariate normal
    of Rung 2). For a neural network it will not. The trick that works either way
    is **Monte Carlo** — replace the integral by an average over samples of $\mu$:

    $$P(y_1,y_2)\approx\frac1S\sum_{s=1}^S
      \mathcal N(y_1\mid\mu^{(s)},\sigma^2)\,\mathcal N(y_2\mid\mu^{(s)},\sigma^2),
      \qquad \mu^{(s)}\sim\mathcal N(m_n,s_n^2).$$

    In words: **sample one hypothesis $\mu^{(s)}$, predict *both* points under it,
    repeat, and average.** The correlation appears because the *same* $\mu^{(s)}$
    is used for both points inside each sample — a mixture of products is not a
    product. This "sample a hypothesis, predict, average" is exactly eqn 2 of the
    ENN paper, with the epistemic index $z=\mu$. The check below (sampling vs. the
    closed form) matches $s_n^2$ and $\sigma^2+s_n^2$.
    """)
    return


@app.cell
def _(jax, jnp, mo):
    _sigma2, _s0_2, _n = 1.0, 8.0, 5
    _sn2 = 1.0 / (1.0 / _s0_2 + _n / _sigma2)
    _S = 400_000
    _km, _k1, _k2 = jax.random.split(jax.random.PRNGKey(2), 3)
    _mu = jnp.sqrt(_sn2) * jax.random.normal(_km, (_S,))              # sample z = mu (m_n = 0)
    _y1 = _mu + jnp.sqrt(_sigma2) * jax.random.normal(_k1, (_S,))
    _y2 = _mu + jnp.sqrt(_sigma2) * jax.random.normal(_k2, (_S,))
    _mc_cov = float(jnp.mean(_y1 * _y2) - jnp.mean(_y1) * jnp.mean(_y2))
    _mc_var = float(jnp.var(_y1))

    mo.md(
        f"""
        **n=5 posterior, $s_n^2={_sn2:.4f}$, {_S:,} samples of $z=\\mu$:**

        | quantity | sampling recipe | analytic |
        |---|---|---|
        | $\\mathrm{{Cov}}(y_1,y_2)$ (off-diagonal) | {_mc_cov:.4f} | $s_n^2$ = {_sn2:.4f} |
        | $\\mathrm{{Var}}(y_1)$ (diagonal) | {_mc_var:.4f} | $\\sigma^2+s_n^2$ = {_sigma2 + _sn2:.4f} |

        The recipe reproduces the joint exactly. Everything so far used a scalar
        $\\mu$. The only new idea in an ENN is: **let $z$ index a function instead
        of a number.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 7 — From a *number* to a *function*

    In regression the unknown is not one mean but a whole function
    $f(\cdot)$ — and at each input $x$ we are back to Rung 1: a prediction
    $y=f(x)+\varepsilon$. The epistemic index $z$ now selects **which function**,
    and eqn 2 reads $\;\hat P(y_{1:\tau})=\int p(z)\prod_t \mathcal N(y_t\mid f(x_t,z),\sigma^2)\,dz$.

    A toy: a smooth truth $f_\text{true}(x)=\sin(1.1x)$ observed with known noise
    $\sigma$ in **two clusters**, leaving an empty gap in the middle (and
    open-ended sides). We *want* a model that is confident on the clusters and
    uncertain in the gap — but only if it represents a distribution over
    functions.
    """)
    return


@app.cell
def _(jnp, np):
    def f_true(x):
        return np.sin(1.1 * np.asarray(x))

    _rng = np.random.default_rng(1)
    _xl = _rng.uniform(-5.0, -2.0, 60)
    _xr = _rng.uniform(1.5, 4.5, 60)
    sigma_noise = 0.15                       # known aleatoric noise

    X_train = jnp.asarray(np.concatenate([_xl, _xr]), dtype=jnp.float32)
    Y_train = jnp.asarray(f_true(np.concatenate([_xl, _xr])) + sigma_noise * _rng.standard_normal(120),
                          dtype=jnp.float32)
    x_grid = jnp.linspace(-8.0, 7.0, 300, dtype=jnp.float32)

    pair_data = (-3.5, -3.0)    # both inside the left cluster
    pair_gap = (-0.3, -0.1)     # both inside the empty gap, close together
    return X_train, Y_train, f_true, pair_data, pair_gap, sigma_noise, x_grid


@app.cell
def _(X_train, Y_train, f_true, np, plt, x_grid):
    _fig, _ax = plt.subplots(figsize=(10, 2.8))
    _g = np.asarray(x_grid)
    _ax.plot(_g, f_true(_g), color="k", lw=1.2, alpha=0.6, label="true f")
    _ax.scatter(np.asarray(X_train), np.asarray(Y_train), s=12, alpha=0.5, color="C0", label="data")
    _ax.axvspan(-2.0, 1.5, color="grey", alpha=0.08)
    _ax.text(-0.25, 1.4, "no data (epistemic gap)", ha="center", color="grey", fontsize=9)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_ylim(-2.0, 1.9)
    _ax.set_title("The toy: two clusters + an empty gap")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Our network toolkit (small MLP)

    Nothing fancy: a tiny multilayer perceptron in a few lines of JAX. We will
    reuse it three ways — as one net, as a fixed *prior* function, and as a
    trainable ensemble member.
    """)
    return


@app.cell
def _(jax, jnp, np):
    def init_mlp(key, sizes, scale=1.0):
        keys = jax.random.split(key, len(sizes) - 1)
        params = []
        for i in range(len(sizes) - 1):
            W = jax.random.normal(keys[i], (sizes[i], sizes[i + 1])) * scale / np.sqrt(sizes[i])
            params.append((W, jnp.zeros(sizes[i + 1])))
        return params

    def mlp(layers, x):
        h = jnp.atleast_1d(x)
        n = len(layers)
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < n - 1:
                h = jnp.tanh(h)
        return h[0]

    NET = [1, 64, 64, 1]
    return NET, init_mlp, mlp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rung 7a — one network is a point estimate: *no* uncertainty

    Fit a single MLP by least squares. It threads the data nicely — and then
    says something completely arbitrary in the gap **with total confidence**. A
    single network is a single function; it has no distribution over functions,
    so it cannot represent epistemic uncertainty at all. That is the thing we
    must fix.
    """)
    return


@app.cell
def _(NET, X_train, Y_train, init_mlp, jax, jnp, mlp, optax):
    def _loss(p):
        pred = jax.vmap(lambda x: mlp(p, x))(X_train)
        return jnp.mean((pred - Y_train) ** 2)

    _p = init_mlp(jax.random.PRNGKey(7), NET, scale=1.0)
    _opt = optax.adam(3e-3)
    _os = _opt.init(_p)

    @jax.jit
    def _step(p, os):
        loss, g = jax.value_and_grad(_loss)(p)
        u, os = _opt.update(g, os, p)
        return optax.apply_updates(p, u), os, loss

    for _i in range(2500):
        _p, _os, _ = _step(_p, _os)
    single_params = _p
    return (single_params,)


@app.cell
def _(X_train, Y_train, f_true, jax, mlp, np, plt, single_params, x_grid):
    _g = np.asarray(x_grid)
    _yhat = np.asarray(jax.vmap(lambda x: mlp(single_params, x))(x_grid))
    _fig, _ax = plt.subplots(figsize=(10, 2.8))
    _ax.plot(_g, f_true(_g), color="k", lw=1.2, alpha=0.4, label="true f")
    _ax.plot(_g, _yhat, color="C1", lw=2, label="single MLP fit")
    _ax.scatter(np.asarray(X_train), np.asarray(Y_train), s=12, alpha=0.5, color="C0")
    _ax.axvspan(-2.0, 1.5, color="grey", alpha=0.08)
    _ax.text(-0.25, 1.4, "confident, and arbitrary", ha="center", color="C1", fontsize=9)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_ylim(-2.2, 1.9)
    _ax.set_title("One network = one function = zero epistemic uncertainty")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 8 — A prior over functions (randomized priors)

    First, **notation** — every object here is a neural network (a *function* of
    $x$). There are $K$ members, indexed
    $k=1,\dots,K$, and each is built from two little MLPs:

    - $t_k(x)$ — the **trainable** network: an MLP with its own weights that we
      *will* optimise on the data. ("$t$" = trainable.)
    - $g_k(x)$ — the **prior** network: an MLP with *random* weights that we
      **freeze** and never train — one fixed, arbitrary function.
    - $\beta$ — a scalar knob for how loudly the prior speaks.
    - the member itself is their sum:
    $$f_k(x)\;=\;\underbrace{t_k(x)}_{\text{trainable MLP}}\;+\;\beta\,\underbrace{g_k(x)}_{\text{frozen random MLP}} .$$
    - and the epistemic index $z$ of Rung 6 is simply the member id: $z=k$.

    Now the idea. In the scalar case we carried a distribution over the unknown
    *number* $\mu$; the functional analogue is a distribution over *functions*.
    We get $K$ samples from it for free — the frozen priors $\beta\,g_k(x)$.
    Plotted below they are diverse everywhere: before any data, every function is
    plausible. Training will pin each member near its own $g_k$ only where the
    data pulls it off.
    """)
    return


@app.cell
def _(NET, init_mlp, jax, jnp, mlp):
    n_members = 40
    beta = 2.0                 # prior scale

    def member_pred(t_params, g_params, x):
        """trainable net + beta * fixed prior (evaluated at a longer lengthscale x/2)."""
        return mlp(t_params, x) + beta * mlp(g_params, jnp.atleast_1d(x) / 2.0)

    _kt = jax.random.split(jax.random.PRNGKey(0), n_members)
    _kp = jax.random.split(jax.random.PRNGKey(1), n_members)
    tparams_init = jax.vmap(lambda k: init_mlp(k, NET, scale=1.0))(_kt)
    gparams = jax.vmap(lambda k: init_mlp(k, NET, scale=2.0))(_kp)   # frozen prior functions
    return beta, gparams, member_pred, tparams_init


@app.cell
def _(beta, gparams, jax, mlp, np, plt, x_grid):
    # the K fixed prior functions beta * g_k(x) -- our sample of the functional prior
    _g = np.asarray(x_grid)
    _priors = np.asarray(
        jax.vmap(lambda gp: jax.vmap(lambda x: beta * mlp(gp, x / 2.0))(x_grid))(gparams)
    )
    _fig, _ax = plt.subplots(figsize=(10, 2.8))
    for _k in range(_priors.shape[0]):
        _ax.plot(_g, _priors[_k], color="C4", alpha=0.3, lw=0.8)
    _ax.set_xlabel("x")
    _ax.set_ylabel(r"$\beta\,g_k(x)$")
    _ax.set_title("40 fixed random prior functions = samples from a prior over functions (no data yet)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 9 — Train: prior over functions → posterior over functions

    Train every member by least squares (Gaussian likelihood, known $\sigma$),
    **each keeping its own frozen prior**. The mechanism, exactly parallel to the
    scalar posterior:

    - **On the data**, member $k$ adjusts $t_k$ to fit the targets *despite* its
      prior $\beta g_k$ — so all members are pulled onto the same curve → they
      **agree** → epistemic spread collapses. (This is $\mu$ getting pinned.)
    - **In the gap**, nothing constrains $t_k$; each member still carries its own
      $\beta g_k$, so they **stay spread**. (This is the leftover prior variance.)

    Below: the same ensemble **before** training (left; still the prior, diverse
    everywhere) and **after** (right; a posterior over functions — pinned on the
    clusters, fanned in the gap).
    """)
    return


@app.cell
def _(X_train, Y_train, gparams, jax, jnp, member_pred, optax, tparams_init):
    def _member_loss(t_params, g_params):
        pred = jax.vmap(lambda x: member_pred(t_params, g_params, x))(X_train)
        return jnp.mean((pred - Y_train) ** 2)          # MSE = Gaussian NLL, known sigma

    def total_loss(tparams):
        return jnp.mean(jax.vmap(_member_loss)(tparams, gparams))

    _opt = optax.adamw(3e-3, weight_decay=1e-5)
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
    print(f"final training MSE: {float(_loss):.4f}  (aleatoric floor sigma^2 = 0.0225)")
    return (tparams_trained,)


@app.cell
def _(
    X_train,
    Y_train,
    gparams,
    jax,
    member_pred,
    np,
    plt,
    tparams_init,
    tparams_trained,
    x_grid,
):
    _g = np.asarray(x_grid)

    def _curves(tp):
        return np.asarray(jax.vmap(lambda t, g: jax.vmap(lambda x: member_pred(t, g, x))(x_grid))(tp, gparams))

    _before, _after = _curves(tparams_init), _curves(tparams_trained)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for _ax, _C, _title in [(_a1, _before, "BEFORE training: prior over functions"),
                            (_a2, _after, "AFTER training: posterior over functions")]:
        for _k in range(_C.shape[0]):
            _ax.plot(_g, _C[_k], color="C3", alpha=0.12, lw=0.8)
        _ax.scatter(np.asarray(X_train), np.asarray(Y_train), s=8, color="C0", alpha=0.5, zorder=5)
        _ax.axvspan(-2.0, 1.5, color="grey", alpha=0.08)
        _ax.set_ylim(-3, 3)
        _ax.set_xlabel("x")
        _ax.set_title(_title, fontsize=10)
    _a1.set_ylabel("member f_k(x)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That right-hand panel is the whole point: a **posterior over functions**,
    obtained from single noisy observations, that is confident where it has seen
    data and honestly uncertain where it has not. This *is* a valid Epistemic
    Neural Network — the paper is explicit that an ensemble is an ENN, with the
    index $z$ simply selecting a member ($z\in\{1,\dots,K\}$). It is just the
    *expensive* kind (K full networks). In **Part II** we build the paper's
    actual architecture — the **epinet** — which gets the same behaviour from a
    single shared body.

    ## Rung 10 — Read off aleatoric vs epistemic

    At each $x$ the members give a little Gaussian over $f(x)$: their **mean** is
    the prediction, their **spread** is the epistemic $s^2(x)$, and on top sits
    the constant aleatoric $\sigma$. Drawn as two bands:
    """)
    return


@app.cell
def _(gparams, jax, member_pred, np, tparams_trained, x_grid):
    _curves = np.asarray(
        jax.vmap(lambda tp, gp: jax.vmap(lambda x: member_pred(tp, gp, x))(x_grid))(tparams_trained, gparams)
    )
    curves = _curves
    grid_np = np.asarray(x_grid)
    gmean = curves.mean(0)
    gstd = curves.std(0)          # epistemic std across members = sqrt(s²(x))
    return gmean, grid_np, gstd


@app.cell
def _(X_train, Y_train, f_true, gmean, grid_np, gstd, np, plt, sigma_noise):
    _fig, _ax = plt.subplots(figsize=(10, 3.6))
    _ax.plot(grid_np, f_true(grid_np), color="k", lw=1.0, alpha=0.4, label="true f")
    _ax.plot(grid_np, gmean, color="C0", lw=2, label="predictive mean")
    _ax.fill_between(grid_np, gmean - gstd, gmean + gstd, color="C3", alpha=0.3,
                     label="epistemic ± s(x)")
    _ax.fill_between(grid_np, gmean - sigma_noise, gmean + sigma_noise, color="C2", alpha=0.35,
                     label="aleatoric ± sigma")
    _ax.scatter(np.asarray(X_train), np.asarray(Y_train), s=10, color="C0", alpha=0.4, zorder=5)
    _ax.axvspan(-2.0, 1.5, color="grey", alpha=0.06)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_ylim(-2.6, 2.2)
    _ax.set_title("Aleatoric band is constant; epistemic band collapses on data, balloons in the gap")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare to Rung 1: on the clusters we are in **World A** (only aleatoric left),
    in the gap we are in **World B** (epistemic dominates) — and a marginal
    (predictive mean alone) could never tell you which. One last check: the
    **joint**.

    ## Rung 11 — The joint, on the network

    Pick two inputs, read every member's prediction, and compute the covariance
    across members. Because observations are independent **given $z$**,
    $\mathrm{Cov}(y_a,y_b)=\mathrm{Cov}_z(f_a,f_b)$ — the exact analogue of
    $\mathrm{Cov}=\mathrm{Var}(\mu)$ from Rung 3.
    """)
    return


@app.cell
def _(gparams, jax, member_pred, mo, np, pair_data, pair_gap, tparams_trained):
    def _f_at(x):
        return np.asarray(jax.vmap(lambda tp, gp: member_pred(tp, gp, x))(tparams_trained, gparams))

    def _joint(pair):
        fa, fb = _f_at(pair[0]), _f_at(pair[1])
        return float(np.cov(fa, fb)[0, 1]), float(np.corrcoef(fa, fb)[0, 1])

    _cov_d, _corr_d = _joint(pair_data)
    _cov_g, _corr_g = _joint(pair_gap)

    mo.md(
        f"""
        | pair | $x_a,x_b$ | **joint cov** | corr |
        |---|---|---|---|
        | inside a cluster | {pair_data} | **{_cov_d:+.4f}** | {_corr_d:+.2f} |
        | inside the gap | {pair_gap} | **{_cov_g:+.4f}** | {_corr_g:+.2f} |

        The **cluster** pair is ~independent (joint cov $\\approx0$ — the *circle*
        of Rung 2, pure aleatoric); the **gap** pair is strongly coupled (orders
        of magnitude larger, corr near $+1$ — the *line* of Rung 2, pure
        epistemic). The network learned this coupling from **single noisy
        observations only** — no joint target ever existed. That is exactly what
        makes joints, not marginals, the right object for exploration and active
        learning.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Part II — The epinet: the same idea, one shared body

    ## Rung 12 — The catch: rungs 8–11 were *K* full networks

    So yes — the ensemble is just $K$ separate random-prior functions, made cheap
    only in the sense that they share nothing. The paper's actual architecture,
    the **epinet**, keeps **one** base network and bolts on a small add-on that
    takes the index $z$ as an *input* (paper eqn 4):

    $$f_\theta(x,z)=\underbrace{\mu_\zeta(x)}_{\text{base net (shared)}}
      \;+\;\underbrace{\sigma_\eta(\,\mathrm{sg}[\phi_\zeta(x)],\,z\,)}_{\text{epinet (small add-on)}},
      \qquad z\sim\mathcal N(0,I_{D_z}).$$

    - $\mu_\zeta(x)$ — the ordinary base network (an MLP $x\to\mathbb R$).
    - $\phi_\zeta(x)$ — **features** of the base (a hidden layer) handed to the
      epinet; $\mathrm{sg}[\cdot]$ is a **stop-gradient** so the epinet reads the
      features but cannot reshape them.
    - $z\in\mathbb R^{D_z},\ z\sim\mathcal N(0,I)$ — a **continuous** index; sample
      as many as you like.

    ### Writing out $\sigma_\eta$ (the part I hand-waved)

    The epinet is itself a sum of a **learnable** net and a **fixed prior** net,
    $\sigma_\eta=\sigma^L_\eta+\sigma^P$:

    $$\sigma^L_\eta(\tilde\phi,z)=\frac{1}{\sqrt{D_z}}\,\mathrm{mlp}_\eta([\tilde\phi,z])\cdot z,
      \qquad \tilde\phi=\mathrm{sg}[\phi_\zeta(x)]$$

    — the MLP maps the concatenation $[\tilde\phi,z]$ to a $D_z$-vector, then we
    dot it with $z$ (paper eqn 6, output dim $C=1$ here). It is initialised near
    $0$.

    $$\sigma^P(x,z)=\frac{1}{\sqrt{D_z}}\sum_{i=1}^{D_z} z_i\,h_i(x)=\frac{z\cdot h(x)}{\sqrt{D_z}},$$

    where $h=(h_1,\dots,h_{D_z})$ are $D_z$ **fixed random functions** of the
    input. Look at what $\sigma^P$ is: a Gaussian-weighted combination of $D_z$
    frozen random functions — **literally a randomized-prior ensemble (Rung 8),
    now serving as the epinet's prior term.** Altogether,

    $$\boxed{\,f_\theta(x,z)=\mu_\zeta(x)+\sigma^L_\eta(\mathrm{sg}[\phi_\zeta(x)],z)+\beta\,\sigma^P(x,z)\,.}$$

    **The new choices it forces on you** (you asked): *what to feed as features* —
    we use the base's last hidden layer for $\sigma^L$ but the **raw input $x$**
    for the prior $\sigma^P$ (the paper allows $\phi$ to be "even just the input");
    that decoupling is what keeps the priors diverse in the gap, where $\sigma^L$
    — seeing only $\mathrm{sg}[\phi]$ — cannot follow to cancel them. Plus the
    *index dimension* $D_z$ and the *prior scale* $\beta$. The paper stresses these
    are "essential."

    **Why it's worth it.** The base runs **once**; every extra sample
    $f_\theta(x,z)$ is a cheap epinet pass. Bolt a tiny epinet onto a huge
    *pretrained* network and joint predictions come almost for free — an ensemble
    never can. (Formally, the epinet is an ENN that *cannot* be written as an
    ensemble/BNN with the same base — paper Thm 3.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 13 — Build and train the epinet

    Same toy, same MSE loss, but now **one** base plus a shared epinet, trained by
    sampling a fresh batch of indices $z\sim\mathcal N(0,I)$ each step (the paper's
    Algorithm 1). The base learns the mean; the epinet learns to make the sum
    $\mu+\sigma_\eta$ fit the data *for every $z$*, which forces the learnable
    $\sigma^L$ to cancel the prior $\sigma^P$ where there is data and leaves it be
    where there is not.
    """)
    return


@app.cell
def _(X_train, Y_train, jax, jnp, np, optax):
    Dz, F, beta_epi = 16, 16, 2.0   # index dim, feature dim, prior scale

    def einit(key, sizes, scale=1.0, last=None):
        ks = jax.random.split(key, len(sizes) - 1)
        out = []
        for i in range(len(sizes) - 1):
            s = last if (last is not None and i == len(sizes) - 2) else scale / np.sqrt(sizes[i])
            out.append((jax.random.normal(ks[i], (sizes[i], sizes[i + 1])) * s, jnp.zeros(sizes[i + 1])))
        return out

    def emlp(layers, x, final_tanh=False):
        h = x
        n = len(layers)
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < n - 1 or final_tanh:
                h = jnp.tanh(h)
        return h

    def epi_base(params, x):                  # the SHARED base -> features phi (F,)
        return emlp(params["base"], jnp.atleast_1d(x), final_tanh=True)

    def epi_head(params, frozen, x, phi, z):  # cheap per-z head
        mu = (phi @ params["mu"][0] + params["mu"][1])[0]                              # base prediction
        # learnable epinet  sigma^L = mlp([sg[phi], z]) . z   (paper eqn 6, C=1)
        sigL = jnp.dot(emlp(params["epiL"], jnp.concatenate([jax.lax.stop_gradient(phi), z])), z) / jnp.sqrt(Dz)
        # fixed prior  sigma^P = z . h(x) = sum_i z_i h_i(x): Dz frozen random functions of the RAW input
        sigP = jnp.dot(z, emlp(frozen["hprior"], jnp.atleast_1d(x) / 2.0, final_tanh=True)) / jnp.sqrt(Dz)
        return mu + sigL + beta_epi * sigP

    def epi_f(params, frozen, x, z):
        return epi_head(params, frozen, x, epi_base(params, x), z)

    _k = jax.random.split(jax.random.PRNGKey(0), 4)
    epi_params = {"base": einit(_k[0], [1, 64, 64, F]),
                  "mu": (jax.random.normal(_k[1], (F, 1)) * 0.1, jnp.zeros(1)),
                  "epiL": einit(_k[2], [F + Dz, 64, Dz], last=0.01)}   # learnable, init ~0
    epi_frozen = {"hprior": einit(_k[3], [1, 32, Dz], scale=1.5)}     # Dz fixed random functions of x

    def _loss(params, zb):
        def per_z(z):
            return jnp.mean((jax.vmap(lambda x: epi_f(params, epi_frozen, x, z))(X_train) - Y_train) ** 2)
        return jnp.mean(jax.vmap(per_z)(zb))

    _opt = optax.adam(3e-3)
    _os = _opt.init(epi_params)

    @jax.jit
    def _step(params, os, key):
        zb = jax.random.normal(key, (24, Dz))          # fresh index batch each step
        loss, g = jax.value_and_grad(_loss)(params, zb)
        u, os = _opt.update(g, os, params)
        return optax.apply_updates(params, u), os, loss

    _p, _key, _l = epi_params, jax.random.PRNGKey(7), None
    for _i in range(6000):
        _key, _sk = jax.random.split(_key)
        _p, _os, _l = _step(_p, _os, _sk)
    epi_trained = _p
    print(f"epinet final MSE: {float(_l):.4f}  (aleatoric floor sigma^2 = 0.0225)")
    return Dz, epi_base, epi_f, epi_frozen, epi_head, epi_trained


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rung 14 — Sample many z from *one* base pass

    Compute the base features on the grid **once**; then each epistemic sample is
    just a cheap epinet evaluation at a new $z$. The bands come out with the right
    shape — tight on the data, wider in the gap — from a single shared network.
    """)
    return


@app.cell
def _(
    Dz,
    X_train,
    Y_train,
    epi_base,
    epi_frozen,
    epi_head,
    epi_trained,
    f_true,
    jax,
    np,
    plt,
    sigma_noise,
    x_grid,
):
    _Z = jax.random.normal(jax.random.PRNGKey(123), (300, Dz))
    _phi = jax.vmap(lambda x: epi_base(epi_trained, x))(x_grid)        # ONE base pass over the grid
    _curves = np.asarray(
        jax.vmap(lambda z: jax.vmap(lambda x, p: epi_head(epi_trained, epi_frozen, x, p, z))(x_grid, _phi))(_Z)
    )                                                                  # (n_z, n_grid) via cheap heads
    _mean, _std = _curves.mean(0), _curves.std(0)
    _g = np.asarray(x_grid)

    _fig, _ax = plt.subplots(figsize=(10, 3.4))
    _ax.plot(_g, f_true(_g), color="k", lw=1.0, alpha=0.4, label="true f")
    _ax.plot(_g, _mean, color="C0", lw=2, label="predictive mean")
    _ax.fill_between(_g, _mean - _std, _mean + _std, color="C3", alpha=0.3, label="epistemic ± std_z")
    _ax.fill_between(_g, _mean - sigma_noise, _mean + sigma_noise, color="C2", alpha=0.35, label="aleatoric ± sigma")
    _ax.scatter(np.asarray(X_train), np.asarray(Y_train), s=10, color="C0", alpha=0.4, zorder=5)
    _ax.axvspan(-2.0, 1.5, color="grey", alpha=0.06)
    _ax.set_xlabel("x")
    _ax.set_ylabel("y")
    _ax.set_ylim(-2.6, 2.2)
    _ax.set_title("Epinet: one shared base + a small z-add-on -> tight on data, wider in the gap")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The band is tight on the data and balloons in the gap — the shape we want,
    from **one** shared base network. Getting here needed the design choices from
    Rung 12 (prior on the *raw input* so it stays diverse in the gap; $D_z$ and
    $\beta$ tuned): the paper is right that the epinet's "design, initialization
    and scaling ... is essential." It still sits in a sweet spot — push $\beta$ or
    the training too far and the learnable part cancels the prior *everywhere*,
    collapsing the gap. That delicacy is the price of the shared body; the payoff
    is that this scales to a large pretrained model where 40 full copies would
    not.
    """)
    return


@app.cell
def _(Dz, epi_f, epi_frozen, epi_trained, jax, mo, np, pair_data, pair_gap):
    _Z = jax.random.normal(jax.random.PRNGKey(7), (400, Dz))

    def _f_at(x):
        return np.asarray(jax.vmap(lambda z: epi_f(epi_trained, epi_frozen, x, z))(_Z))

    def _joint(pair):
        fa, fb = _f_at(pair[0]), _f_at(pair[1])
        return float(np.cov(fa, fb)[0, 1]), float(np.corrcoef(fa, fb)[0, 1])

    _cov_d, _corr_d = _joint(pair_data)
    _cov_g, _corr_g = _joint(pair_gap)

    mo.md(
        f"""
        ## Rung 15 — The joint, on the epinet

        Same measurement as Rung 11, now with a continuous index: sample $z$, read
        $f(x_a,z)$ and $f(x_b,z)$, covary.

        | pair | $x_a,x_b$ | joint cov | corr |
        |---|---|---|---|
        | inside a cluster | {pair_data} | {_cov_d:+.4f} | {_corr_d:+.2f} |
        | inside the gap | {pair_gap} | {_cov_g:+.4f} | {_corr_g:+.2f} |

        The same qualitative split as the ensemble — near-zero coupling on the
        data, positive coupling in the gap — but produced by **one** base network
        with a continuous index $z$. Softer numbers (the epinet is an
        approximation), identical story.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap — the ladder in one breath

    A single measurement fuses aleatoric $\sigma^2$ and epistemic $s^2$ into one
    marginal (Rung 1). The **joint** of two measurements pulls them apart:
    $\mathrm{Cov}(y_i,y_j)=\mathrm{Var}(\mu)=s^2$ (Rungs 2–3). Averaging removes
    the aleatoric part and exposes the epistemic floor (Rung 4), which data
    shrinks $\sim1/n$ (Rung 5). To *carry* epistemic uncertainty you keep a
    distribution over the unknown and sample it (Rung 6) — and an ENN just lets
    that unknown be a **function** (Rung 7). A single net has none of this (7a);
    a set of trainable networks each anchored to a **fixed random prior** (Rungs
    8–9) is a posterior over functions — confident on data, uncertain in gaps —
    whose two-band decomposition (Rung 10) and joint structure (Rung 11) are the
    scalar Gaussian, lifted. That ensemble is a valid but *heavy* ENN; the
    **epinet** (Rungs 12–15) gets the same joint behaviour from a single shared
    base plus a tiny $z$-indexed add-on — one forward pass, so it scales to large
    pretrained models — which is the paper's actual contribution.

    *Run: `~/my_jax_env/bin/marimo edit tutorials/epistemic_nn_from_gaussian.py`*
    """)
    return


if __name__ == "__main__":
    app.run()
