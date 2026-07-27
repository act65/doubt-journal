---
layout: paper
slug: bdia
title: Bi-directional Integration Approximation (BDIA)
authors: Zhang et al.
year: 2023
venue: arXiv
ref: https://arxiv.org/abs/2307.10829
permalink: /papers/bdia/
read: 2024-01-01
depth: rebuilt
rating: 2
tags: [generative-models, numerical-methods, disputed]
question: Can a bi-directional coupled integrator make diffusion inversion exact without doubling the NFEs?
tldr: >
  Coupling variables lets you construct reversible integration updates. BDIA extends that idea to
  a bi-directional coupled integrator. The idea may have potential, but this paper does not
  demonstrate it.
scholar:
  bibliography: "bdia.bib"
notebooks: []
related: []
---

## Review

Numerical solution of integrals introduce errors. Coupling variables allows us to construct reversible integration updates. BDIA is a method that extends this idea to construct a bi-directional coupled integrator. The idea may have potential, but this paper does not demonstrate it.

## Problem

When integrating ODEs there are integration errors (linearisation errors). We can write the discretised integral as;

$$
\int_0^T x(t) dt = x_0 + \lim_{n\to \infty}\sum_{i=1}^n f(x_i^*, t_i^*) (t_{i} - t_{i-1}) \\
t^*_i \in [t_{i-1}, t_{i}] \\
x^*_i \in [x_{i-1}, x_{i}] \\
$$

Where $n$ is not infinite, and non-linear $f$ we get different results for;

- integrating from $0 \to T$ vs $T \to 0$. 
- choosing $${t^{*}}_i =t_{i-1}$$ and $$x_i^* = x_{t_i-1}$$ vs choosing $$t^*_i=t_{i}$$ and $$x_i^* = x_{t_i}$$.

Solutions?

- can mimimise this problem by increasing $n$.
- by adaptively choosing $t^*_i$.
- by adaptively choosing the intervals $[t_i, t_{i+1}]$.

When training neural networks to solve ODEs we can use the adjoint method.
This requires the forward and backward pass to be consistent. This is not the case when using numerical integration.
We want to construct an ODE such that the forward and backward pass are consistent. (aka invertible).


<!-- $$
x_{i-1} = x_{i+1} - \Delta(t_i \to t_{i+1} \mid x_i) + \Delta(t_i \to t_{t-1} \mid x_{i-1}) \\
x_{i+1} = x_{i-1} + \Delta(t_i \to t_{i+1} \mid x_i) - \Delta(t_i \to t_{t-1} \mid x_{i+1}) \\
$$ -->

## Overview

### Coupling to construct invertible fns

How can we construct a fn that is easily invertible?

For general classes of $f$, constructing $f^{-1}$ is not always possible (ie if $f$ is many-to-one).
Consider this simple example where we start with $x$ and recover $y$.
To invert this, we are given $y$ and want to recover $x$.

$$
x + f(x) = y
$$

$x$ cannot be recovered from $y$ unless we know $f(x)$ or can calculate $f^{-1}(y)$.

What if we couple this equation with another?
{% cite dinh2017densityestimationusingreal %}

$$
x + f(x) = y \\
x + y = z \\
$$

Or more generally;

$$
a + b = c \\
a + c = d \\
$$

We can recover $a, b$ from $c$ and $b$ since we have 2 equations and 2 unknown variables.

$$
a = d - c \\
b = c - a
$$

Can rewrite as;

$$
a + b = c \\
2 a + b = d \\
$$

$$
\begin{bmatrix}
1 & 1 \\
2 & 1 \\
\end{bmatrix}
\begin{bmatrix}
a \\
b \\
\end{bmatrix}
=
\begin{bmatrix}
c \\
d \\
\end{bmatrix}
$$

And to solve this we do Gaussian elimination.

<div class="code" markdown="1">

Could use many other parameterisations.

$$
A = \begin{bmatrix}
1 & 1 \\
1 & -1 \\
\end{bmatrix} \\
a + b = c \\
a - b = d \\
a - (c - a) = d \\
2a - c = d \\
$$

$$
A = \begin{bmatrix}
1 & 1 \\
-1 & 1 \\
\end{bmatrix} \\
a + b = c \\
- a + b = d \\
- a + (c - a) = d \\
- 2a + c = d \\
$$

$$
A = \begin{bmatrix}
1 & 1 \\
0 & 1 \\
\end{bmatrix} \\
a + b = c \\
b = d \\
$$

- __Q__: Are there advantages to using one parameterisation over another?
- __Q__: Is there a way to view this conjugate variable as some kind of memory?

</div>

<!-- What if we want to accumulate 2 variables? Or add 2 variables?

3 variables.

$$
a + b + c = d \\
2a + b + c = e \\
2a + 2b + c = f \\
$$


$$
a + b + c = d \\
a + b + d = e \\
a + b + e = f \\
$$

Rearranged gives;

$$
a + b + c = d \\
a + b + (a + b + c) = e \\
a + b + (a + b + (a + b + c)) = f \\
$$

$$
a + b + c = d \\
2a + 2b + c = e \\
3a + 3b + c = f \\
$$

$$
\begin{bmatrix}
1 & 1 & 1 \\
2 & 2 & 1 \\
3 & 3 & 1 \\
\end{bmatrix}
\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix}
=
\begin{bmatrix}
d \\
e \\
f \\
\end{bmatrix}
$$ -->


### EDICT

EDICT uses this coupling idea to allow us to find the inverse of a non-linear function. {% cite wallace2022edictexactdiffusioninversion %}

For the forward integral update we have;

$$
\begin{align*}
y_i &= y_{i-1} + \eta \frac{dx}{dt}(z_{i-1}) \tag{1}\\
z_i &= z_{i-1} + \eta \frac{dx}{dt}(y_i) \tag{2}\\
\end{align*}
$$

and for the backward integral update we have;

$$
\begin{align*}
z_{i-1} &= z_t - \eta \frac{dx}{dt}(y_i) \tag{3}\\
y_{i-1} &= y_t - \eta \frac{dx}{dt}(z_{i-1}) \tag{4}\\
\end{align*}
$$


<div class="code" markdown="1">
Note that we can rewrite these as;

$$
y_t = y_0 + \sum_{i=1}^{t-1} \eta \frac{dx}{dt}(z_i) \\
z_t = z_0 + \sum_{i=1}^{t-1} \eta \frac{dx}{dt}(y_{i+1}) \\
$$
</div>


<side>
It is not clear to me why this divergence is a problem.
</side>
They also include a mixing layer. The problem this fixes is that the sequences of $y$ and $z$ can diverge. They demonstrate the utility of this mixing layer by improved performance in the diffusion model samples. 

$$
\begin{align*}
x' = px + (1-p)y \\
p \in [0, 1] \\
\end{align*}
$$

<!-- Why not add another coupled variable?

Note;

$$
\begin{align*}
a_i &= a_{i-1} + \eta \frac{dx}{dt}(b_{i-1})\\
b_i &= b_{i-1} + \eta \frac{dx}{dt}(a_i) \\
c_i &= c_{i-1} + \eta \frac{dx}{dt}(b_i) \\
\end{align*}
$$

$$
a_t = a_0 + \sum_{i=1}^{t-1} \eta \frac{dx}{dt}(b_i) \\
b_t = b_0 + \sum_{i=1}^{t-1} \eta \frac{dx}{dt}(a_{i+1}) \\
c_t = c_0 + \sum_{i=1}^{t-1} \eta \frac{dx}{dt}(b_{i+1}) \\
$$

These are like look-forward gradients / peeking into the future.

$$
\frac{dx}{dt}(b_{i+1}) = \frac{dx}{dt}(b_i + \eta \frac{dx}{dt}(a_i+ \eta \frac{dx}{dt}(a_i)))
$$
*** -->

<div class="code" markdown="1">
I implemented EDICT to test it's invertibility.

<img src="{{ site.baseurl }}/assets/images/bdia/edict-lk.png" width="80%">
<figcaption>Here we implement EDICT (without mixing) and use it to solve the Lotka-Volterra equations. RK4 is computed at 1000x more resultion to act as 'ground truth'.
</figcaption>

</div>

<!-- <img src="{{ site.baseurl }}/assets/images/bdia/edict-sc.png" width="80%"> -->

<!-- Here we integrate $x_1=sin(t)$ and $x_2=cos(t)$ using EDICT (without mixing). RK4 is computed at 1000x more resultion to act as 'ground truth'. -->


<!-- ***

- If $y_t$ is the variable we care about then what is $z_t$?
- Why do we need the mixing?
- Is there a bound on how far $y_t$ and $z_t$ may diverge (assuming they are started from the same point)? -->

### BDIA

_(from what I can gather, as I find the paper confusing)_

The key ideas to construct BDIA;

- coupled transformations
- averaging of the forward and backward gradients
- develop an update expression for each $x_{i+1}$ as a linear combination of $(x_{i-1}, x_i, \nabla f(x_i))$ (and vice versa).

Which yields updates of the form;

$$
\begin{align*}
x_{i+1} &= x_i + f(x_i) \\
&= x_{i-1} + (x_i - x_{i-1}) + f(x_{i}) \\
\end{align*}
$$

Where the $(x_i - x_{i-1}) $ term was previously computed in the previous iteration as $f(x_{i-1})$.
However, we could also choose to combine (/ average) these two terms, which would give us;

$$
\begin{align*}
x_{i+1} &= x_{i-1} +  \gamma f(x_{i-1}) + (1- \gamma) f(x_i) \\
\end{align*}
$$

Th main advantage of BIDA is (supposedly); by using the coupled variable as the past / future gradients it does not require a doubling of the NFEs (like EDICT does).

<!-- 
$$
\begin{align*}
x_{i} &= x_{i-1} + f(x_{i-1}) \\
x_{i+1} &= x_{i} + f(x_{i}) \\
&= x_{i-1} + f(x_{i-1}) + f(x_{i}) \\
\end{align*}
$$ -->


<img src="{{ site.baseurl }}/assets/images/bdia/scheme.png" width="120%">
<figcaption>BDIA scheme {% cite zhang2023exactdiffusioninversionbidirectional %}. Though this image hints a $2n$ complexity just via a counting unique arrows argument.</figcaption>

Intuitively, they seem to have swapped the coupled variable in EDICT with a past / future gradient.
However, how to write this as a 2 variable system of equations (as I did for EDICT) is not clear to me since we now have 3 variables.


<!-- ***

Averaging forward and backward gradients;

$$
\int_{\tau_1}^{\tau_1} d(z_t, t) dt \approx (1-\theta) \Delta(\tau_1 \to \tau_2 \mid z_{\tau_1}) - \theta \Delta(\tau_2 \to \tau_1 \mid z_{\tau_2}) \\
$$

$$
\lim_{\tau_1 \to \tau_2} \int_{\tau_1}^{\tau_2} d(z_t, t) dt = (1-\theta) \Delta(\tau_1 \to \tau_2 \mid z_{\tau_1}) - \theta \Delta(\tau_2 \to \tau_1 \mid z_{\tau_2}) \\
$$

- What's the accuracy? As $\epsilon = \tau_1 - \tau_2 \to 0$.

But we dont have access to $z_{\tau_1}$ and $z_{\tau_2}$. -->

## Feedback

A new integrator has been introduced. You should disentangle this idea from its application to diffusion models.
And since a new integrator has been introduced it should be benchmarked against existing methods.
Further, important questions such as; "How accurate is the BDIA integrator? How does it compare to EDICT on some simple tasks? What is the trade-off between accuracy and computational cost?" are not addressed.

## Bibliography

{% bibliography --cited %}
