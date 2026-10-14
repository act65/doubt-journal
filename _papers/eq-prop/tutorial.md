---
layout: paper-part
slug: eq-prop
part: tutorial
title: "Tutorial · Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation"
---

## Overview

We start with an energy function that defines our model.

$$
E(u) = \frac{1}{2} \sum u_i^2 - \frac{1}{2} \sum_{i \neq j} W_{ij} \rho(u_i) \rho(u_j) - \sum_i b_i \rho(u_i)
$$

where $u$ is the state of the system, $W$ is the weight matrix, $b$ is the bias vector, and $\rho$ is the activation function. The energy function is a function of the state of the system and the parameters of the model.

To achieve minimum energy, strongly connected units should have similar states.
