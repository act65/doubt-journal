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

    return jax, jnp, np, plt, random, vmap


@app.cell
def _(mo):
    mo.md(r"""
    # Bayesian Flow Networks on DISCRETE data, from Bayesian fundamentals

    Same one sentence as before — a belief about the clean data, a brain that corrects it with the
    true prior, and generation by bootstrapping — but now $x$ is **categorical**. Two things change,
    and both are improvements in clarity:

    1. The belief is a **categorical distribution** (per dimension), held as **logits** `L`.
    2. A noisy "measurement" is a vector `y`; folding it in is just **`L += y`** — Bayes in
       log-space is *addition*. And the output distribution is now a **real categorical we sample
       from**, not the degenerate Dirac delta of the continuous case.

    | piece | what it is | paper name |
    |---|---|---|
    | `belief = L` | per-dim logits; `softmax(L)` = categorical belief over each $x^d$ | input dist / $\theta$ |
    | `observe(c, a)` | noisy vector `y ~ N(a(C·e_c − 1), aC·I)` pushing the true class up | sender |
    | `update(L, y)` | `L += y` (posterior logits, since likelihood $\propto e^{y}$) | Bayesian update |
    | `denoise(L)` | per-dim evidence + **true joint prior** → corrected categoricals | network + output dist |

    We use $x=(x_1,x_2)$, each in $\{0,\dots,C-1\}$, with a **correlated** joint prior (a shape on
    the grid). A per-dimension belief physically *cannot* represent correlation, so the brain has to
    put it back — the discrete analog of the multimodal pull from the continuous notebook.
    """)
    return


@app.cell
def _(jnp, random):
    # ---- true joint prior p_data(x1, x2): a ring shape on the CxC grid ----
    C = 8
    D = 2
    _ii, _jj = jnp.meshgrid(jnp.arange(C), jnp.arange(C), indexing="ij")
    _r2 = (_ii - 3.5) ** 2 + (_jj - 3.5) ** 2
    prior = jnp.where((_r2 >= 3.0) & (_r2 <= 9.5), 1.0, 0.0)
    prior = prior / prior.sum()                       # (C, C) joint
    log_prior = jnp.log(prior + 1e-12)

    def sample_data(key, n):
        _idx = random.choice(key, C * C, (n,), p=prior.ravel())
        return jnp.stack([_idx // C, _idx % C], axis=1)

    return C, D, log_prior, prior


@app.cell
def _(C, plt, prior):
    _fig, _ax = plt.subplots(figsize=(4.0, 4.0))
    _ax.imshow(prior.T, origin="lower", cmap="Blues", extent=[-.5, C - .5, -.5, C - .5])
    _ax.set_title("true joint prior  p_data(x1, x2)  — a correlated shape")
    _ax.set_xlabel("x1"); _ax.set_ylabel("x2")
    _ax.set_xticks(range(C)); _ax.set_yticks(range(C))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The belief, and two Bayesian primitives

    A **belief** is a categorical distribution over each variable, held as logits `L` (shape
    `D × C`). `softmax(L[d])` is our current probability over class values of $x^d$. Uniform prior =
    all logits zero.

    - **`observe(c, a)`** returns a noisy measurement of the true classes `c`. In sender space
      $y \in \mathbb R^{C}$ per dimension, with $y \sim \mathcal N\!\big(a(C\,e_c - 1),\; aC\,I\big)$.
      The mean points up at the true class; larger accuracy `a` = less noise.
    - **`update(L, y)`** folds a measurement into the belief. The likelihood of `y` under class $k$
      works out to $\propto e^{y_k}$, so in log-space the posterior is just **addition**:

    $$
    L \;\leftarrow\; L + y .
    $$

    That's the whole discrete update — no matrices, no $\gamma$: accumulate log-evidence.
    """)
    return


@app.cell
def _(C, D, jax, jnp, random):
    def observe(key, c, a):
        e = jax.nn.one_hot(c, C)                       # (D, C) one-hot of true classes
        return a * (C * e - 1.0) + jnp.sqrt(a * C) * random.normal(key, (D, C))

    def update(L, y):
        return L + y                                   # posterior logits

    return observe, update


@app.cell
def _(C, jax, jnp, np, observe, plt, random, update):
    # Demo: accumulate REAL measurements of a fixed config. The (naive) joint belief =
    # outer product of the two per-dim categoricals sharpens onto the true cell.
    def _accumulate():
        _c = jnp.array([5, 2])
        _L = jnp.zeros((2, C)); _key = random.PRNGKey(1)
        _snaps = {}
        _steps = [2, 8, 30]
        for _i in range(1, 31):
            _key, _k = random.split(_key)
            _L = update(_L, observe(_k, _c, 0.05))   # small accuracy -> gradual sharpening
            if _i in _steps:
                _p = jax.nn.softmax(_L, axis=1)
                _snaps[_i] = np.array(_p[0][:, None] * _p[1][None, :])   # naive joint
        return _c, _snaps

    _c, _snaps = _accumulate()
    _fig, _ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for _n, (_i, _P) in enumerate(_snaps.items()):
        _ax[_n].imshow(_P.T, origin="lower", cmap="Blues", extent=[-.5, C - .5, -.5, C - .5])
        _ax[_n].scatter([int(_c[0])], [int(_c[1])], marker="*", c="red", s=200)
        _ax[_n].set_title(f"after {_i} measurements"); _ax[_n].set_xlabel("x1"); _ax[_n].set_ylabel("x2")
    _fig.suptitle("belief (naive joint) sharpens onto the true config ★")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The brain: per-dim evidence + true joint prior → corrected categoricals

    The belief `softmax(L)` treats the two variables as **independent** — its joint is a rank-1
    outer product. It can say "x1 is probably 5 and x2 is probably around 2/6", but it can never say
    "x1 and x2 are *correlated*". That correlation lives only in the true prior.

    **`denoise(L)` puts it back.** The logits `L[d]` are exactly the accumulated log-evidence for
    each class, so the true joint posterior over a config $(i,j)$ is

    $$
    P(i,j \mid \text{evidence}) \;\propto\; p_{\text{data}}(i,j)\; e^{\,L[0]_i + L[1]_j},
    $$

    and the brain's output for each dimension is that joint's **marginal**. (Note it needs *only*
    `L` — no time input — because the logits already encode how much evidence we have.) Below: the
    naive belief (blurry rank-1) vs the brain's joint (snapped onto prior-consistent cells).
    """)
    return


@app.cell
def _(C, jax, jnp, log_prior):
    def denoise(L):
        # joint posterior over (i, j) = true prior x per-dim evidence, then take marginals
        J = log_prior + L[0][:, None] + L[1][None, :]         # (C, C)
        P = jax.nn.softmax(J.ravel()).reshape(C, C)
        return jnp.stack([P.sum(axis=1), P.sum(axis=0)])      # (D, C) corrected per-dim cats

    def joint_posterior(L):
        J = log_prior + L[0][:, None] + L[1][None, :]
        return jax.nn.softmax(J.ravel()).reshape(C, C)

    return denoise, joint_posterior


@app.cell
def _(C, jax, jnp, joint_posterior, np, plt):
    # Partial evidence pointing x1 toward ~5 (and x2 weakly): compare naive vs brain joints.
    def _brain_demo():
        _L = jnp.zeros((2, C)).at[0].set(jnp.array([0, 0, 0, 0, 1.5, 3.0, 1.5, 0.]))
        _L = _L.at[1].set(jnp.array([0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0.]))
        _p = jax.nn.softmax(_L, axis=1)
        _naive = np.array(_p[0][:, None] * _p[1][None, :])
        _brain = np.array(joint_posterior(_L))
        return _naive, _brain

    _naive, _brain = _brain_demo()
    _fig, _ax = plt.subplots(1, 2, figsize=(8.4, 4.0))
    for _a, _M, _t in [(_ax[0], _naive, "naive belief (independent) — rank-1 blur"),
                       (_ax[1], _brain, "brain's joint — respects the prior")]:
        _a.imshow(_M.T, origin="lower", cmap="Blues", extent=[-.5, C - .5, -.5, C - .5])
        _a.set_title(_t, fontsize=9); _a.set_xlabel("x1"); _a.set_ylabel("x2")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Schedule: how fast to accumulate evidence

    As before we grow total accuracy over `n` steps. Discrete BFN uses a **quadratic** schedule
    (the paper's $\beta(t) = t^2\,\beta_1$); step `i` adds `a_i = total_acc(i/n) − total_acc((i-1)/n)`.
    `beta1` just needs to be large enough that the final belief pins down a class.
    """)
    return


@app.cell
def _(np, plt):
    beta1 = 8.0
    def total_acc(s):
        return (s ** 2) * beta1

    _s = np.linspace(0, 1, 100)
    _fig, _ax = plt.subplots(figsize=(5, 2.4))
    _ax.plot(_s, [total_acc(v) for v in _s], c="C1")
    _ax.set_xlabel("progress s"); _ax.set_ylabel("total accuracy β(s)")
    _ax.set_title("β(s) = s²·β₁  (quadratic schedule)")
    _fig
    return (total_acc,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Generation = bootstrap (now sampling a real categorical)

    Each step, exactly as before but the "guess" is a *sample* from the brain's categorical (this is
    the non-degenerate output distribution the continuous case never had):

    1. **guess**: `q = denoise(L)`, then **sample** a class per dimension from `q`;
    2. **observe the guess**: `y = observe(guess, a_i)`;
    3. **update**: `L += y`.

    The belief's joint prediction collapses from the whole ring onto one self-consistent cell. The
    randomness in the sampled guesses (early, when the belief is vague) is what spreads samples
    across the ring.
    """)
    return


@app.cell
def _(C, D, denoise, jnp, observe, random, total_acc, update, vmap):
    def _guess(key, q):
        return jnp.stack([random.choice(random.fold_in(key, d), C, p=q[d]) for d in range(D)])

    def generate(key, n=60):
        L = jnp.zeros((D, C))
        for i in range(1, n + 1):
            q = denoise(L)
            key, ks, ky = random.split(key, 3)
            a_i = total_acc(i / n) - total_acc((i - 1) / n)
            L = update(L, observe(ky, _guess(ks, q), a_i))
        return jnp.argmax(denoise(L), axis=1)

    def generate_snapshots(key, n=60, steps=(6, 20, 40, 60)):
        L = jnp.zeros((D, C)); snaps = {}
        for i in range(1, n + 1):
            q = denoise(L)
            key, ks, ky = random.split(key, 3)
            a_i = total_acc(i / n) - total_acc((i - 1) / n)
            L = update(L, observe(ky, _guess(ks, q), a_i))
            if i in steps:
                from_ = denoise(L)
                snaps[i] = (from_[0][:, None] * from_[1][None, :])   # predicted joint
        return snaps

    def generate_many(seed=0, count=6000, n=60):
        return vmap(lambda k: generate(k, n))(random.split(random.PRNGKey(seed), count))

    return generate_many, generate_snapshots


@app.cell
def _(C, generate_snapshots, np, plt, random):
    _snaps = generate_snapshots(random.PRNGKey(3), n=60)
    _fig, _ax = plt.subplots(1, len(_snaps), figsize=(11, 3.2))
    for _n, (_i, _P) in enumerate(_snaps.items()):
        _ax[_n].imshow(np.array(_P).T, origin="lower", cmap="Blues", extent=[-.5, C - .5, -.5, C - .5])
        _ax[_n].set_title(f"step {_i}", fontsize=9); _ax[_n].set_xlabel("x1"); _ax[_n].set_ylabel("x2")
    _fig.suptitle("ONE generation run: brain's predicted joint collapses ring → one cell")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(C, generate_many, jnp, np, plt, prior):
    _samp = generate_many()
    _H = jnp.zeros((C, C)).at[_samp[:, 0], _samp[:, 1]].add(1.0)
    _H = np.array(_H / _H.sum())
    _tv = 0.5 * float(np.abs(_H - np.array(prior)).sum())
    _fig, _ax = plt.subplots(1, 2, figsize=(8.4, 4.0))
    _ax[0].imshow(np.array(prior).T, origin="lower", cmap="Blues", extent=[-.5, C - .5, -.5, C - .5])
    _ax[0].set_title("true prior"); _ax[0].set_xlabel("x1"); _ax[0].set_ylabel("x2")
    _ax[1].imshow(_H.T, origin="lower", cmap="Blues", extent=[-.5, C - .5, -.5, C - .5])
    _ax[1].set_title("6000 generated samples"); _ax[1].set_xlabel("x1"); _ax[1].set_ylabel("x2")
    print("mass on allowed cells:", round(float(_H[np.array(prior) > 0].sum()), 3),
          "  total-variation vs prior:", round(_tv, 3))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the discrete case is the point of the whole framework

    Same architecture as the continuous notebook, but notice what changed:

    - **belief** = per-dim **categorical** (logits `L`) instead of `(m, A)`. It lives on the
      probability simplex — a **continuous** space — *even though the data is discrete*. That is the
      deep reason one framework covers both: the network always ingests continuous belief parameters.
    - **update** = `L += y`: Bayes in log-space is addition. No forward process, no transition
      matrices (contrast discrete diffusion).
    - **output distribution** = a **genuine categorical we sample from** — not the Dirac delta of the
      continuous case. Here "the network parameterises a distribution" is finally non-trivial, and
      the receiver draw (sample a guess, then measure it) is a real two-stage sample.
    - **denoise** still = the correction: it injects the true joint prior the per-dim belief can't
      hold. You can see it work — a product-of-marginals belief could never be ring-shaped, yet
      generation reproduces the ring (TV ≈ 0.02).

    Training would again be the simplest thing: take a data config, add noise to a random accuracy
    to get a belief `L`, and train `denoise(L)` to predict the clean classes by cross-entropy — the
    discrete-data loss. It's still "learn a denoiser," but because the belief is a continuous
    categorical, the exact same recipe that gave a diffusion-like model for images gives a coherent
    generative model for symbols, with no bespoke machinery.
    """)
    return


if __name__ == "__main__":
    app.run()
