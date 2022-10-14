---
layout: post
title: Performative prediction
permalink: performative-prediction
rating: 8
ref: https://arxiv.org/abs/2002.06673
---

# Performative prediction

## Review

<div class="center">
<strong>Rating</strong>: {{ page.rating }} / 10
</div>
<hr>

## Overview

<blockquote cite="Perdomo 2020">
Consider a simplified example of predicting credit default risk. A bank might estimate that a loan applicant has an elevated risk of default, and will act on it by assigning a high interest rate. In a self-fulfilling prophecy, the high interest rate further increases the customer's default risk.
</blockquote>


$$
\text{PR}(\theta) = \mathop{\mathbb E}_{Z\sim D(\theta)} \ell(Z ; \theta)
$$


$$
\text{PR}(\theta) = \mathop{\mathbb E}_{P, R \sim D(\theta)} \ell(P, R ; \theta)
$$




$$
\theta_{t+1} = \mathop{\text{argmin}}_{\theta} \mathop{\mathbb E}_{Z\sim D(\theta_t)} \ell(Z ; \theta)
$$


<blockquote cite="Perdomo 2020">
<strong>Theorem 1.1</strong> (Informal). If the loss is smooth, strongly convex, and the mapping D(·) is sufficiently Lipschitz, then repeated risk minimization converges to performative stability at a linear rate. Moreover, if any one of these assumptions does not hold, repeated risk minimization can fail to
converge at all.
</blockquote>




<blockquote><strong>Example 2.2</strong> (biased coin flip). Consider the task of predicting the outcome of a biased coin flip where the bias of the coin depends on a feature X and the assigned score fθ(X). In particular, define D(θ) in the following way. X is a 1-dimensional feature supported on {±1} and Y | X ∼ Bernoulli(1
2 +µX + εθX) with µ ∈ (0, 1 2) and ε < 1
2 − µ. Assume that the class
of predictors consists of linear models of the form fθ(x) = θx + 1
2 and that the objective is to
minimize the squared loss: ?(z;θ) = (y −fθ(x))2. The parameter ε represents the performative aspect of the model. If ε = 0, outcomes are independent of the assigned scores and the problem reduces to a standard supervised learning task where the optimal predictor is the conditional expectation fθSL
(x) = E[Y | X = x] = 1 2 +µx,
with θSL = µ.
</blockquote>

$\text{Bernoulli}(\frac{1}{2} +\mu X + \epsilon \theta X)$ . Believing that $\mu = \theta X$ influences which samples we see (maybe because these are of higher value so we are more likely to choose them).


<hr>

## Thoughts

So how can we view the data as being generated?

Ok, wait a minute. Retraining seems really expensive? We need to be able to gather many data points from our new distrubtion?!

Memorisation doesnt work! You remember that Joe's credit risk was X. But then your estimate influences the interest rate, which changes Joe's credit risk.

How is this related to policy iteration? What trajectories does RRM / RBM tend to take? What is the difference between these trajectories?


$$
\theta_{t+1} = \mathop{\text{argmin}}_{\theta} \text{DPR}(\theta, \theta_t) \\
\theta_{t+1} = \mathop{\text{argmin}}_{\theta} \text{DPR}(\theta_t, \theta) \\
$$

Or alternating!?
What is the difference?


What do these assumptions mean for our debtor and creditor?
- Strong convexity:
- Jointly smooth:
- $\epsilon \ge \frac{\gamma}{\beta}$:


Doesnt prove the uniqueness of optima?!

Model based performative prediction!?
We have some estimate / knowledge of $\nabla D(\theta)$? (kinda like the transition fn) This allows us to optimise without requiring extra samples!?
