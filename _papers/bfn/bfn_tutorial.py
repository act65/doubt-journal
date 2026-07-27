import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    # Force CPU (avoids the cuBLAS init issue) and use float64 — the schedule
    # involves sigma1^(-2t) which gets large, and small gamma near t=0.
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import random
    import numpy as np
    import matplotlib.pyplot as plt

    return jax, jnp, np, plt, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian Flow Networks — a worked 2D continuous-Gaussian example

    Paper: Graves et al., *Bayesian Flow Networks* (arXiv:2308.07037). All equation
    numbers below refer to it.

    **The one-paragraph picture.** A *sender* knows the true data $\mathbf x$ and keeps
    transmitting noisy measurements $\mathbf y$ of it. A *receiver* does **not** know
    $\mathbf x$; it keeps a **belief** about $\mathbf x$ — a simple parametric distribution
    $p_I(\cdot\mid\boldsymbol\theta)$ — and a network that turns that belief into a guess
    $\hat{\mathbf x}$. Each round the receiver Bayes-updates its belief with the new
    measurement, so the belief *flows* from an uninformative prior toward the data.

    The thing we will nail down with code: **what is actually fed to the network, and what
    do the sender / receiver / input / output distributions each sample.**

    A trick that makes this fully runnable with **no training**: for *Gaussian data* the
    Bayes-optimal denoiser (what a perfectly trained net would learn) has a closed form. So
    we plug that in and watch the whole machine work.
    """)
    return


@app.cell
def _(jnp):
    # ---- the data distribution we want the BFN to learn: 2D Gaussian N(m, S) ----
    m = jnp.array([0.3, -0.4])          # data lives in [-1, 1]^2 as the paper assumes
    S = jnp.array([[0.05, 0.03],
                   [0.03, 0.04]])
    D = 2
    sigma1 = 0.02                        # std of the input/belief distribution at t=1 (Eq 70)
    return D, S, m, sigma1


@app.cell
def _(S, m, np, plt):
    def _gauss_grid(mean, cov, lim=1.0, n=200):
        _xs = np.linspace(-lim, lim, n)
        _X, _Y = np.meshgrid(_xs, _xs)
        _P = np.stack([_X.ravel(), _Y.ravel()], -1)
        _d = _P - np.array(mean)
        _ic = np.linalg.inv(np.array(cov))
        _z = np.exp(-0.5 * np.einsum("ni,ij,nj->n", _d, _ic, _d))
        return _X, _Y, _z.reshape(n, n)

    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _X, _Y, _Z = _gauss_grid(m, S)
    _ax.contourf(_X, _Y, _Z, levels=12, cmap="Blues")
    _ax.set_title("data distribution  p_data = N(m, S)")
    _ax.set_xlabel("x[0]"); _ax.set_ylabel("x[1]")
    _ax.set_aspect("equal")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Input distribution $p_I(\mathbf x\mid\boldsymbol\theta)$ — the *belief* (Eqs 42–45)

    $$\boldsymbol\theta=\{\boldsymbol\mu,\rho\},\qquad
    p_I(\mathbf x\mid\boldsymbol\theta)=\mathcal N(\mathbf x\mid\boldsymbol\mu,\rho^{-1}\mathbf I).$$

    - It **is** a parametric family — a diagonal (factorised, independent dimensions)
      Gaussian.
    - Crucially $\boldsymbol\theta$ is **not** model parameters; it's the receiver's current
      *state of knowledge* about $\mathbf x$. $\boldsymbol\mu$ is the current best estimate,
      $\rho$ the precision (confidence).
    - Prior (Eq 44): $\boldsymbol\theta_0=\{\mathbf 0, 1\}$, i.e. a standard normal — "I know
      nothing, my guess is $\mathbf 0$ with precision 1."
    - The paper stresses (p.11): $p_I$ is *never used to make predictions directly* — it only
      **feeds the network**.
    """)
    return


@app.cell
def _(jnp, sigma1):
    # ---- schedule: everything about "how much information by time t" (Eqs 72, 74, 78-80) ----
    def beta(t):    return sigma1 ** (-2.0 * t) - 1.0                  # accumulated accuracy, Eq 72
    def gamma(t):   return 1.0 - sigma1 ** (2.0 * t)                   # Eq 80  (= beta/(1+beta))
    def alpha(t):   return -2.0 * jnp.log(sigma1) / sigma1 ** (2.0 * t)  # instantaneous accuracy, Eq 74
    def rho_of_t(t):return 1.0 + beta(t)                              # input precision = sigma1^(-2t)

    return beta, gamma


@app.cell
def _(beta, gamma, np, plt):
    _t = np.linspace(0, 1, 200)
    _fig, _ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    _ax[0].plot(_t, [float(gamma(s)) for s in _t])
    _ax[0].set_title(r"$\gamma(t)=1-\sigma_1^{2t}$  (signal scale of the belief mean)")
    _ax[0].set_xlabel("t"); _ax[0].set_ylabel(r"$\gamma$")
    _ax[1].plot(_t, [float(1 + beta(s)) for s in _t])
    _ax[1].set_title(r"$\rho(t)=1+\beta(t)=\sigma_1^{-2t}$  (belief precision)")
    _ax[1].set_xlabel("t"); _ax[1].set_ylabel(r"$\rho$"); _ax[1].set_yscale("log")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Sender $p_S(\mathbf y\mid\mathbf x;\alpha)$ + Bayesian update — the belief *flows*

    **Sender (Eq 4):** $p_S(\mathbf y\mid\mathbf x;\alpha)=\mathcal N(\mathbf y\mid\mathbf x,\alpha^{-1}\mathbf I)$
    — the *true* data point blurred with variance $1/\alpha$.

    **Bayesian update (Eqs 46–50):** a noisy Gaussian observation $\mathbf y$ at precision
    $\alpha$ updates a Gaussian belief in closed form:

    $$\rho_i=\rho_{i-1}+\alpha,\qquad
    \boldsymbol\mu_i=\frac{\boldsymbol\mu_{i-1}\rho_{i-1}+\mathbf y\,\alpha}{\rho_i}.$$

    Below: start from the prior, draw three measurements of a fixed true point at increasing
    accuracies, and watch $\boldsymbol\mu$ stagger toward the data while $\rho$ climbs. This is
    the 2D version of the paper's Figure 2.
    """)
    return


@app.cell
def _(jnp, plt, random):
    def _bayes_demo():
        _x_true = jnp.array([0.5, -0.2])
        _key = random.PRNGKey(1)
        _mu = jnp.zeros(2); _rho = 1.0
        _accs = [2.0, 4.0, 8.0]
        _mus = [_mu]; _rhos = [_rho]; _ys = []
        for _a in _accs:
            _key, _k = random.split(_key)
            _y = _x_true + random.normal(_k, (2,)) / jnp.sqrt(_a)   # sender sample
            _rho_new = _rho + _a
            _mu = (_mu * _rho + _y * _a) / _rho_new                 # Eqs 49-50
            _rho = _rho_new
            _mus.append(_mu); _rhos.append(_rho); _ys.append(_y)
        return _x_true, jnp.stack(_mus), jnp.array(_rhos), jnp.stack(_ys)

    _x_true, _mus, _rhos, _ys = _bayes_demo()
    _fig, _ax = plt.subplots(figsize=(4.6, 4.6))
    _ax.scatter(*_x_true, c="k", marker="*", s=200, label="true x", zorder=5)
    _ax.scatter(_ys[:, 0], _ys[:, 1], c="red", marker="x", s=70, label="sender samples y")
    _ax.plot(_mus[:, 0], _mus[:, 1], "-o", c="C0", label="belief mean μ (flows to x)")
    for _i, (_mx, _my) in enumerate(_mus):
        _ax.annotate(f"μ{_i} (ρ={float(_rhos[_i]):.0f})", (float(_mx), float(_my)),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    _ax.legend(fontsize=8); _ax.set_aspect("equal")
    _ax.set_title("Bayesian updates: belief staggers toward x")
    _ax.set_xlabel("x[0]"); _ax.set_ylabel("x[1]")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Bayesian flow $p_F(\boldsymbol\theta\mid\mathbf x;t)$ — the belief at time $t$, in closed form

    Because every update is linear-Gaussian, the *marginal* belief mean at time $t$ (given the
    true $\mathbf x$) has a closed form — no need to simulate the loop (Eqs 76–80):

    $$p_F(\boldsymbol\theta\mid\mathbf x;t)=\mathcal N\!\big(\boldsymbol\mu\;\big|\;
    \gamma(t)\,\mathbf x,\ \gamma(t)(1-\gamma(t))\mathbf I\big),\qquad \rho=1+\beta(t).$$

    So $\boldsymbol\mu$ at time $t$ is a noisy interpolation from $\mathbf 0$ (at $t{=}0$) to
    $\mathbf x$ (at $t{=}1$). This is the analogue of "noised latent $\mathbf z_t$" in diffusion,
    and it's what lets training jump straight to a random $t$. Below: the $x[0]$-component of
    $\boldsymbol\mu$ vs $t$ — closed-form mean $\pm$ std band, with a few simulated trajectories
    on top (the paper's Figure 4).
    """)
    return


@app.cell
def _(beta, gamma, jnp, np, plt, random):
    def _flow_band_and_paths():
        _x_true = jnp.array([0.7, -0.2])
        _t = np.linspace(0, 1, 200)
        _mean = np.array([float(gamma(s)) * 0.7 for s in _t])
        _std = np.array([np.sqrt(max(float(gamma(s) * (1 - gamma(s))), 0.0)) for s in _t])
        # simulate a few actual trajectories via many tiny Bayesian updates
        _n = 300
        _paths = []
        _key = random.PRNGKey(3)
        for _s in range(6):
            _key, _k0 = random.split(_key)
            _mu = jnp.zeros(2); _rho = 1.0; _traj = [0.0]
            for _i in range(1, _n + 1):
                _a = float(beta(_i / _n) - beta((_i - 1) / _n))
                _k0, _k = random.split(_k0)
                _y = _x_true + random.normal(_k, (2,)) / jnp.sqrt(_a)
                _rho2 = _rho + _a
                _mu = (_mu * _rho + _y * _a) / _rho2; _rho = _rho2
                _traj.append(float(_mu[0]))
            _paths.append(np.array(_traj))
        return _t, _mean, _std, np.linspace(0, 1, _n + 1), _paths

    _t, _mean, _std, _tp, _paths = _flow_band_and_paths()
    _fig, _ax = plt.subplots(figsize=(7.5, 3.6))
    _ax.fill_between(_t, _mean - 2 * _std, _mean + 2 * _std, alpha=0.2, color="C0",
                     label=r"closed-form $\pm2\,$std")
    _ax.plot(_t, _mean, "C0", lw=2, label=r"mean $=\gamma(t)\,x$")
    for _p in _paths:
        _ax.plot(_tp, _p, lw=0.8, alpha=0.7)
    _ax.axhline(0.7, ls="--", c="k", lw=1); _ax.text(0.02, 0.72, "x[0]=0.7", fontsize=8)
    _ax.set_xlabel("t"); _ax.set_ylabel("μ[0]")
    _ax.set_title("Bayesian flow: μ fans out, then converges on x")
    _ax.legend(fontsize=8)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The network $\Psi$ and the output distribution $p_O$ — carefully

    **General framework (§3.2, Eq 3).** The input parameters $\boldsymbol\theta$ and the time $t$
    are fed to a network $\Psi$, which *emits an output vector*
    $\Psi(\boldsymbol\theta,t)=(\Psi^{(1)},\dots,\Psi^{(D)})$ that **parameterises** a factorised
    output distribution:
    $$p_O(\mathbf x\mid\boldsymbol\theta,t)=\prod_{d=1}^{D}
    p_O\!\big(x^{(d)}\mid\Psi^{(d)}(\boldsymbol\theta,t)\big).$$
    Unlike $p_I$, each factor $p_O^{(d)}$ depends on *all* of $\boldsymbol\theta$ through the net,
    so the output distribution can exploit cross-dimension context.

    **Continuous specialisation (§4.7, Eqs 84–85, and the `cts_output_prediction` box on p.19).**
    Three *separate* objects — I earlier conflated them, which was your objection:

    1. **Network input** $=(\boldsymbol\mu,t)$. Here $\boldsymbol\theta=\{\boldsymbol\mu,\rho\}$,
       but $\rho=\sigma_1^{-2t}$ is fixed by $t$, so only $(\boldsymbol\mu,t)$ is passed — verbatim
       what the algorithm box feeds the net.
    2. **Network output** $=\hat{\boldsymbol\epsilon}(\boldsymbol\theta,t)$, an estimate of the
       *noise* $\boldsymbol\epsilon$ that generated $\boldsymbol\mu$. This raw vector *is*
       $\Psi(\boldsymbol\theta,t)$ (diffusion-style $\epsilon$-parameterisation).
    3. **Output distribution** $p_O=\delta(\mathbf x-\hat{\mathbf x})$, a Dirac delta whose centre
       $\hat{\mathbf x}$ is a *deterministic reparameterisation* of $\hat{\boldsymbol\epsilon}$
       (Eqs 84–85):
    $$\hat{\mathbf x}(\boldsymbol\theta,t)=\frac{\boldsymbol\mu}{\gamma(t)}
    -\sqrt{\tfrac{1-\gamma(t)}{\gamma(t)}}\,\hat{\boldsymbol\epsilon}(\boldsymbol\theta,t),\qquad
    p_O(\mathbf x\mid\boldsymbol\theta,t)=\delta\big(\mathbf x-\hat{\mathbf x}\big).$$
    (For $t<t_{\min}$ the reparameterisation is undefined, so the algorithm sets
    $\hat{\mathbf x}=\mathbf 0$.)

    So, precisely: **input $(\boldsymbol\mu,t)$ → network emits $\hat{\boldsymbol\epsilon}=\Psi$ →
    reparameterise to $\hat{\mathbf x}$ → output distribution $\delta(\mathbf x-\hat{\mathbf x})$.**
    My earlier "the net outputs $\hat{\mathbf x}$" skipped step 2; the code below now separates the
    network ($\hat{\boldsymbol\epsilon}$, `eps_hat_net`) from the output prediction
    ($\hat{\mathbf x}$, `cts_output_prediction`), matching the algorithm box.

    $\hat{\mathbf x}$ is the network's denoised **guess of the clean data point** (the
    $\mathbf x_0$-prediction). *This* is the $\mathbf x'$ the receiver samples (next section) — not
    a draw from $p_{\text{data}}$, not a posterior over parameters.

    For Gaussian data the Bayes-optimal $\hat{\boldsymbol\epsilon}$ (what a perfect net would emit)
    is closed-form, so we use it as the stand-in "trained network".
    """)
    return


@app.cell
def _(D, S, gamma, jnp, m):
    def eps_hat_net(mu, t):
        # The NETWORK Psi(theta,t): input (mu, t), output a NOISE estimate eps_hat.
        # We substitute the Bayes-optimal noise estimator for Gaussian data:
        #   flow:  mu = gamma x + sqrt(gamma(1-gamma)) eps,  eps ~ N(0, I)
        #   z := mu/gamma ~ N(x, tau2 I);  x_post = E[x|mu,t] is the Gaussian posterior mean;
        #   optimal eps_hat = (mu - gamma x_post) / sqrt(gamma(1-gamma))   [invert Eq 82].
        g = gamma(t)
        tau2 = (1.0 - g) / g
        z = mu / g
        K = S @ jnp.linalg.inv(S + tau2 * jnp.eye(D))       # posterior-mean gain
        x_post = m + (z - m) @ K.T                          # E[x | mu, t]
        return (mu - g * x_post) / jnp.sqrt(g * (1.0 - g))  # -> optimal eps_hat  (= Psi)

    return (eps_hat_net,)


@app.cell
def _(eps_hat_net, gamma, jnp):
    def cts_output_prediction(mu, t, t_min=1e-6, x_range=1.0):
        # Paper's CTS_OUTPUT_PREDICTION (Alg. box p.19; Eqs 84-85). Turns the network's
        # eps_hat into the OUTPUT-DISTRIBUTION parameter x_hat; the output distribution itself
        # is p_O = delta(x - x_hat), factorised over dimensions.
        t_safe = jnp.maximum(t, t_min)                      # avoid gamma->0 blow-up at t<t_min
        g = gamma(t_safe)
        eps = eps_hat_net(mu, t_safe)                       # <- network output (Psi)
        xhat = mu / g - jnp.sqrt((1.0 - g) / g) * eps       # Eq 84 reparameterisation
        xhat = jnp.clip(xhat, -x_range, x_range)            # clip to data range
        return jnp.where(t < t_min, jnp.zeros_like(mu), xhat)  # t<t_min -> 0

    return (cts_output_prediction,)


@app.cell
def _(cts_output_prediction, eps_hat_net, gamma, jnp, plt, random):
    def _denoise_demo():
        _x_true = jnp.array([0.5, -0.2])
        _key = random.PRNGKey(0)
        _ts = [0.05, 0.2, 0.5, 0.9]
        _out = []
        for _t in _ts:
            _g = gamma(_t)
            _key, _k = random.split(_key)
            _mu = _g * _x_true + jnp.sqrt(_g * (1 - _g)) * random.normal(_k, (2,))
            _eps = eps_hat_net(_mu, _t)                     # network raw output (Psi)
            _xh = cts_output_prediction(_mu, _t)           # reparameterised x_hat
            _out.append((_t, _mu, _eps, _xh))
        return _x_true, _out

    _x_true, _out = _denoise_demo()
    _fig, _ax = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for _i, (_t, _mu, _eps, _xh) in enumerate(_out):
        _a = _ax[_i]
        _a.scatter(*_x_true, c="k", marker="*", s=160, label="true x")
        _a.scatter(float(_mu[0]), float(_mu[1]), c="C1", s=50, label="μ (net INPUT)")
        _a.scatter(float(_xh[0]), float(_xh[1]), c="C2", marker="P", s=90,
                   label="x̂ (output-dist δ centre)")
        _a.annotate("", xy=(float(_xh[0]), float(_xh[1])),
                    xytext=(float(_mu[0]), float(_mu[1])),
                    arrowprops=dict(arrowstyle="->", color="0.5"))
        _a.set_title(f"t={_t}, γ={float(gamma(_t)):.2f}\n‖ε̂‖={float(jnp.linalg.norm(_eps)):.2f}",
                     fontsize=9)
        _a.set_xlim(-1, 1); _a.set_ylim(-1, 1); _a.set_aspect("equal")
        if _i == 0: _a.legend(fontsize=7, loc="lower left")
    _fig.suptitle("network emits ε̂ (=Ψ)  →  reparameterise to x̂  →  p_O = δ(x − x̂)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Receiver $p_R$, and the loss — sender vs. receiver

    The receiver predicts the *next measurement* without knowing $\mathbf x$, so it pushes its
    own output guess through the same sender noise (Eq for $p_R$, §3):

    $$p_R(\mathbf y\mid\boldsymbol\theta;t,\alpha)
    =\mathbb E_{\mathbf x'\sim p_O(\cdot\mid\boldsymbol\theta;t)}\,
    p_S(\mathbf y\mid\mathbf x';\alpha).$$

    Here $\mathbf x'$ is the network's denoised guess $\hat{\mathbf x}$ (§4). Since $p_O$ is a
    delta at $\hat{\mathbf x}$, the receiver is just $\mathcal N(\mathbf y\mid\hat{\mathbf x},\alpha^{-1}\mathbf I)$:
    **the same Gaussian as the sender, but centred on the net's guess instead of the truth.**

    - **Sender:** $\mathcal N(\mathbf y\mid\mathbf x,\alpha^{-1}\mathbf I)$ — knows the truth.
    - **Receiver:** $\mathcal N(\mathbf y\mid\hat{\mathbf x},\alpha^{-1}\mathbf I)$ — uses the guess.

    Training minimises $D_{\mathrm{KL}}(p_S\Vert p_R)=\frac{\alpha}{2}\Vert\mathbf x-\hat{\mathbf x}\Vert^2$.
    In continuous time this collapses to a diffusion-style weighted MSE (derived from Eq 41):

    $$L^\infty(\mathbf x)=-\ln\sigma_1\;\mathbb E_{t\sim U(0,1),\,\boldsymbol\theta\sim p_F}
    \!\left[\frac{\Vert\mathbf x-\hat{\mathbf x}(\boldsymbol\theta,t)\Vert^2}{\sigma_1^{2t}}\right].$$
    """)
    return


@app.cell
def _(cts_output_prediction, gamma, jnp, np, plt, random):
    def _receiver_demo():
        _x = jnp.array([0.5, -0.2]); _t = 0.4; _a = 30.0
        _g = gamma(_t)
        _mu = _g * _x + jnp.sqrt(_g * (1 - _g)) * random.normal(random.PRNGKey(5), (2,))
        _xh = cts_output_prediction(_mu, _t)
        return _x, np.array(_xh), 1.0 / np.sqrt(_a)

    _x, _xh, _sd = _receiver_demo()
    _g = np.linspace(-0.2, 1.0, 200)
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.plot(_g, np.exp(-0.5 * ((_g - float(_x[0])) / _sd) ** 2), "C3", label="sender ~ N(x, 1/α)")
    _ax.plot(_g, np.exp(-0.5 * ((_g - float(_xh[0])) / _sd) ** 2), "C2--", label="receiver ~ N(x̂, 1/α)")
    _ax.axvline(float(_x[0]), c="C3", lw=0.8); _ax.axvline(float(_xh[0]), c="C2", lw=0.8)
    _ax.set_title("sender vs receiver (x[0] slice) — gap = denoising error → KL")
    _ax.set_xlabel("y[0]"); _ax.legend(fontsize=8)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Training time vs. inference time — what actually happens

    These are two *different* procedures that only meet at the flow $p_F$.

    **Training (we HAVE $\mathbf x$; goal: fit the denoiser).** One step on a data point $\mathbf x$:

    1. pick a random $t\sim U(0,1)$;
    2. jump straight onto the flow in ONE shot —
       $\boldsymbol\mu=\gamma(t)\,\mathbf x+\sqrt{\gamma(t)(1-\gamma(t))}\,\boldsymbol\epsilon$
       (Eq 82), with $\boldsymbol\epsilon\sim\mathcal N(0,\mathbf I)$;
    3. feed $(\boldsymbol\mu,t)$ to the net $\to\hat{\boldsymbol\epsilon}\to\hat{\mathbf x}$;
    4. loss $=$ weighted $\lVert\mathbf x-\hat{\mathbf x}\rVert^2$; backprop.

    No sender, no update loop — because $\mathbf x$ is known we noise it *directly*. Training visits
    ONE random point on the flow per step. This is exactly diffusion training.

    **Inference (we DON'T have $\mathbf x$; goal: generate one).** There is no data point to noise,
    so we *walk* the flow from $t{=}0$ to $t{=}1$ and let the model stand in for the missing
    $\mathbf x$. Each step $i$:

    1. ask the net for its current best guess of the clean data:
       $\hat{\mathbf x}=\text{net}(\boldsymbol\mu_{i-1},t_{i-1})$;
    2. draw a noisy pseudo-observation of *that guess*:
       $\mathbf y\sim\mathcal N(\hat{\mathbf x},\alpha_i^{-1}\mathbf I)$ — i.e. sample from the
       **receiver** (the sender would use the true $\mathbf x$; we don't have it, so we substitute
       our own guess $\hat{\mathbf x}$);
    3. fold $\mathbf y$ into the belief with a Bayesian update $\to\boldsymbol\mu_i,\rho_i$
       (belief sharpens, precision $\rho$ grows).

    Directly answering your question: **each inference step = (i) get the net's guess
    $\hat{\mathbf x}$, (ii) draw a noisy "observation" $\mathbf y$ around $\hat{\mathbf x}$, (iii)
    Bayes-update the belief with $\mathbf y$.** The thing we "send" is *not* real data — it's a noisy
    sample of the model's own current guess. The randomness in those $\mathbf y$ draws is exactly
    what makes different runs produce different samples (like the injected noise in diffusion
    ancestral sampling).

    | | training | inference |
    |---|---|---|
    | do we have $\mathbf x$? | yes (dataset) | **no** |
    | how $\boldsymbol\mu$ arises | one-shot $\gamma\mathbf x+$noise (Eq 82) | accumulated Bayesian updates |
    | time $t$ | one random $t\sim U(0,1)$ | swept $0\to1$ over $n$ steps |
    | what the net drives | the regression loss | the next pseudo-observation $\mathbf y$ |
    | "sender" centred on | true $\mathbf x$ | the guess $\hat{\mathbf x}$ (= receiver) |
    | loop? | no | yes, $n$ steps |
    | produces | a gradient | a generated sample |

    The plot below traces **one** inference run so you can watch the guess → draw → update rhythm.
    """)
    return


@app.cell
def _(beta, cts_output_prediction, jnp, np, plt, random):
    def _trace(key, n=20):
        _mu = jnp.zeros(2); _rho = 1.0
        _mus = [np.array(_mu)]; _xhs = []; _ys = []
        for _i in range(1, n + 1):
            _tp = (_i - 1) / n
            _a = float(beta(_i / n) - beta((_i - 1) / n))
            _xh = cts_output_prediction(_mu, _tp)               # (i) net's current guess
            key, _k = random.split(key)
            _y = _xh + random.normal(_k, (2,)) / jnp.sqrt(_a)   # (ii) draw y ~ receiver N(x̂,1/α)
            _rho2 = _rho + _a
            _mu = (_mu * _rho + _y * _a) / _rho2; _rho = _rho2  # (iii) Bayesian update
            _xhs.append(np.array(_xh)); _ys.append(np.array(_y)); _mus.append(np.array(_mu))
        return np.array(_mus), np.array(_xhs), np.array(_ys)

    _mus, _xh, _ys = _trace(random.PRNGKey(11), n=20)
    _fig, _ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    _ax[0].plot(_mus[:, 0], _mus[:, 1], "-o", c="C0", ms=3, lw=1, label="belief mean μ (walks flow)")
    _ax[0].scatter(_xh[:, 0], _xh[:, 1], c="C2", marker="P", s=45, label="net guess x̂ each step")
    _ax[0].scatter(_ys[:, 0], _ys[:, 1], c="C3", marker="x", s=25, alpha=0.5, label="pseudo-obs y~N(x̂,1/α)")
    _ax[0].scatter(*_mus[0], c="k", marker="s", s=70, label="prior μ0 = 0", zorder=5)
    _ax[0].scatter(*_xh[-1], c="red", marker="*", s=240, label="final sample x̂", zorder=6)
    _ax[0].set_xlim(-1.05, 1.05); _ax[0].set_ylim(-1.05, 1.05); _ax[0].set_aspect("equal")
    _ax[0].legend(fontsize=7, loc="upper left"); _ax[0].set_title("ONE inference run (n=20)")
    _ax[0].set_xlabel("dim 0"); _ax[0].set_ylabel("dim 1")

    _ns = len(_xh)
    _ax[1].plot(range(_ns + 1), _mus[:, 0], "-o", c="C0", ms=3, label="μ[0] (belief)")
    _ax[1].plot(range(1, _ns + 1), _xh[:, 0], "P", c="C2", label="x̂[0] (guess)")
    _ax[1].plot(range(1, _ns + 1), _ys[:, 0], "x", c="C3", alpha=0.6, label="y[0] (pseudo-obs)")
    _ax[1].set_xlabel("step i"); _ax[1].set_title("dim 0 over steps: guess → draw → update")
    _ax[1].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. The n-step sampler (§3.10) and the proof it works

    The $n$-step sampler, plugging the optimal denoiser in for the network:

    ```
    θ0 = {μ=0, ρ=1}
    for i = 1..n:
        t_prev = (i-1)/n
        x̂  = net(μ_{i-1}, t_prev)         # output dist = δ(x − x̂)
        α_i = β(i/n) − β((i-1)/n)          # accuracy increment for this step
        y  ~ N(x̂, α_i^{-1} I)              # sender draws a measurement of the GUESS
        μ_i, ρ_i = bayes_update(μ_{i-1}, ρ_{i-1}, y, α_i)
    return net(μ_n, 1)
    ```

    Note the self-consistency loop: the receiver draws $\mathbf y$ from its *own* current guess,
    updates the belief, re-guesses, and so on. If the denoiser is good, the generated samples
    match $p_{\text{data}}$ — verified below against the true $N(m,S)$.
    """)
    return


@app.cell
def _(beta, cts_output_prediction, jax, jnp, random):
    def generate(key, n=100):
        mu = jnp.zeros(2); rho = 1.0
        for i in range(1, n + 1):
            t_prev = (i - 1) / n                          # cts_output_prediction handles t<t_min
            a_i = beta(i / n) - beta((i - 1) / n)
            xh = cts_output_prediction(mu, t_prev)       # x' from output dist (Eq 85)
            key, k = random.split(key)
            y = xh + random.normal(k, (2,)) / jnp.sqrt(a_i)   # sender
            rho_new = rho + a_i
            mu = (mu * rho + y * a_i) / rho_new           # Bayesian update
            rho = rho_new
        return cts_output_prediction(mu, 1.0)            # final net pass

    def generate_many(seed=42, count=3000, n=100):
        return jax.vmap(lambda k: generate(k, n))(random.split(random.PRNGKey(seed), count))

    return (generate_many,)


@app.cell
def _(S, generate_many, jnp, m, np, plt):
    _samp = generate_many()
    _smean = np.array(_samp.mean(0)); _scov = np.array(jnp.cov(_samp.T))
    _fig, _ax = plt.subplots(figsize=(4.6, 4.6))
    _ax.scatter(_samp[:, 0], _samp[:, 1], s=4, alpha=0.25, c="C0", label="BFN samples")
    _th = np.linspace(0, 2 * np.pi, 100)
    _L = np.linalg.cholesky(np.array(S))
    _circ = (np.array(m)[:, None] + 2 * _L @ np.stack([np.cos(_th), np.sin(_th)]))
    _ax.plot(_circ[0], _circ[1], "r", lw=2, label="data 2σ ellipse")
    _ax.set_aspect("equal"); _ax.legend(fontsize=8)
    _ax.set_title("generated samples vs data distribution")
    print("sample mean", _smean, " target", np.array(m))
    print("sample cov\n", _scov, "\ntarget cov\n", np.array(S))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary — the five distributions, disambiguated

    | object | what it is | what gets sampled | analogue in diffusion |
    |---|---|---|---|
    | **input** $p_I(\mathbf x\mid\boldsymbol\theta)$ | the *belief* (diagonal Gaussian, factorised) | nothing — it just supplies $(\boldsymbol\mu,t)$ to the net | noised latent $\mathbf z_t$ |
    | **sender** $p_S(\mathbf y\mid\mathbf x;\alpha)$ | the **true** $\mathbf x$ + noise $1/\alpha$ | a measurement $\mathbf y$ | forward corruption |
    | **network** $\Psi(\boldsymbol\theta,t)$ | takes $(\boldsymbol\mu,t)$, emits noise estimate $\hat{\boldsymbol\epsilon}$ | — (deterministic map) | $\epsilon$-network |
    | **output** $p_O(\mathbf x\mid\boldsymbol\theta;t)$ | $\delta(\mathbf x-\hat{\mathbf x})$, $\hat{\mathbf x}$ reparam. of $\hat{\boldsymbol\epsilon}$ (Eq 84) | $\mathbf x'=\hat{\mathbf x}$ — the *guess*, not data | $\hat{\mathbf x}_0$ prediction |
    | **receiver** $p_R(\mathbf y\mid\boldsymbol\theta;t,\alpha)$ | output ⊛ sender noise = $\mathcal N(\hat{\mathbf x},\alpha^{-1})$ | predicted $\mathbf y$ | model's predictive |
    | **flow** $p_F(\boldsymbol\theta\mid\mathbf x;t)$ | closed-form belief at time $t$ | $\boldsymbol\mu$ for training | $q(\mathbf z_t\mid\mathbf x_0)$ |

    Three things that resolve the original confusion:

    1. **The network takes $(\boldsymbol\mu,t)$ and emits a noise estimate $\hat{\boldsymbol\epsilon}=\Psi$**;
       the clean-data guess $\hat{\mathbf x}$ is a *deterministic reparameterisation* of it (Eq 84),
       and the output distribution is $\delta(\mathbf x-\hat{\mathbf x})$. Input is the belief mean
       $\boldsymbol\mu$ (the Bayes-updated mean) — confirmed by the `cts_output_prediction` box.
    2. **The "x" the receiver samples is $\hat{\mathbf x}$**, the network's denoised guess —
       not a data sample, not a posterior over parameters. The receiver uses it precisely
       *because* it does not know the true $\mathbf x$.
    3. **The distinctive BFN move:** the reverse-process state is *parameters of a belief*
       $\boldsymbol\theta$ (a smooth, well-behaved space), not a noised data sample — which is
       what lets the very same recipe extend to discrete data.

    Next natural step: the **discrete/categorical** case, where $\boldsymbol\theta$ is a vector
    of class probabilities and the same five distributions reappear in softmax form.
    """)
    return


if __name__ == "__main__":
    app.run()
