---
layout: paper
slug: bfn
title: Bayesian Flow Networks
authors: Graves, Srivastava, Atkinson, Gomez
year: 2023
venue: arXiv
ref: https://arxiv.org/abs/2308.07037
permalink: /papers/bfn/
read: 2026-07-27
depth: rebuilt
rating:
tags: [probabilistic-methods, generative-models, discrete-data, reconception]
question: What is actually fed to the network, and what comes out?
tldr: >
  Keep a belief about the clean data — a posterior built from noisy observations under a naive
  prior — and a network that corrects that naive posterior into a smart estimate using the true
  data prior. Generation bootstraps: guess, observe your own guess, update the belief, repeat.
notebooks:
  - file: bfn_tutorial.py
    what: >
      The paper on its own terms: a worked 2D continuous-Gaussian example, walking the five
      distributions ($p_I$, sender, flow, $p_O$, receiver) with equation numbers, the closed-form
      flow, the loss, and the $n$-step sampler.
  - file: bfn_simple.py
    what: >
      The same machinery rebuilt from Bayesian primitives with the paper's notation dropped.
      Continuous data, a deliberately multimodal 3-Gaussian prior, so the "brain" visibly pulls a
      naive estimate toward a mode.
  - file: bfn_discrete.py
    what: >
      The categorical case — where the framework actually earns its keep. Belief is per-dim
      logits, a Bayesian update is `L += y` (addition in log-space), and the output distribution
      is a real categorical rather than a Dirac delta.
related:
  - enn
---

## Review

_Under review — I haven't written the verdict yet._

## Problem

A generative model that runs the diffusion idea on the *parameters of a belief* rather than on
noisy data — which means it works on discrete data without any relaxation hack. The paper's
notation ($\theta,\mu,\rho,\gamma,\beta,p_I,p_S,p_R,p_O,\Psi$) hides how simple the mechanism is,
and the five distributions are easy to conflate. The goal was to pin down exactly **what is fed
to the network and what comes out**.

## Overview

### Notation

| paper | here | what it is |
|---|---|---|
| input dist / $\theta$ | `belief = (m, A)`, or `L` for discrete | noisy estimate `m` of $x$ with total precision `A`; logits `L` in the discrete case |
| sender $p_S$ | `observe(x, a)` | draw a noisy measurement of the true $x$ at precision `a` |
| Bayesian update $h$ | `update(belief, y)` | fold a measurement in — precision-weighted mean, or `L += y` |
| network $\Psi$ | `denoise` / "the brain" | corrects the naive posterior using the true data prior |
| output dist $p_O$ | `xhat` | Dirac delta at $\hat x$ (continuous); a genuine categorical (discrete) |
| accuracy schedule $\beta(t)$ | `schedule` | how fast evidence accumulates over time |

### What I took away

- The network's only job is **prior correction**. The belief is a posterior under a *naive*
  (factorized, uninformative) prior; $\Psi$ maps it to what the posterior would have been under
  the *true* data prior. Everything else is exact Bayes.
- The continuous output distribution is a **delta**, so nothing is sampled at the output — all
  the stochasticity lives in the sender. The discrete case is the honest one, and is where the
  construction stops looking like a reparameterized diffusion.
- Float64 matters: the schedule involves $\sigma_1^{-2t}$, and $\gamma$ is tiny near $t=0$.

## Additional thoughts

### Open questions

- [ ] How does the learned $\Psi$ degrade when the true prior is much sharper than the model
      class can represent?
- [ ] The $n$-step sampler's proof assumes the belief update is exact — how much does an
      imperfect $\Psi$ compound over steps?
