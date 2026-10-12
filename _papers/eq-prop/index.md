---
layout: paper
slug: eq-prop
title: "Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation"
authors: "Scellier & Bengio"
year: 2017
venue: Frontiers in Computational Neuroscience
ref: http://journal.frontiersin.org/article/10.3389/fncom.2017.00024/abstract
permalink: /papers/eq-prop/
# Migrated from _drafts/2024-08-01-eq-prop.md; `read` is that draft's filename date.
read: 2024-08-01
depth: read
rating: 9
tags: [neuroscience, deep-learning, optimisation, hidden-equivalence]
question:
tldr:
notebooks: []
related: []
draft: true
published: false
---

## Review

Equilibrium Propagation (EP) is a learning algorithm that is inspired by the way the brain works. It is a biologically plausible learning algorithm that is based on the principle of energy minimization. The idea is to minimize the energy of the system by iteratively updating the state of the system until it reaches an equilibrium state. The key contribution of EP is that it demonstrates a connection between energy-based models and backpropagation.

## Problem

## Overview

We start with an energy function that defines our model.

$$
E(u) = \frac{1}{2} \sum u_i^2 - \frac{1}{2} \sum_{i \neq j} W_{ij} \rho(u_i) \rho(u_j) - \sum_i b_i \rho(u_i)
$$

where $u$ is the state of the system, $W$ is the weight matrix, $b$ is the bias vector, and $\rho$ is the activation function. The energy function is a function of the state of the system and the parameters of the model.

To achieve minimum energy, strongly connected units should have similar states.

## Additional thoughts

So as we scale to larger networks, we need;

- more time / iterations for an equilibrium to be reached?!
- how do you efficiently fit the topology into 3d...!?
- ???

- [ ] Still curious about connections between langevin dynamics and distributions. Want the ability to transform between implicit and explicit distributions easily?! To integrate the noise in langevin dynamics or to sample from a distribution!?

## Refs

- [Equilibrium Propagation with Continual Weight Updates](https://arxiv.org/abs/2005.04168)
- [Training End-to-End Analog Neural Networks with Equilibrium Propagation](https://arxiv.org/abs/2006.01981)
- [A solution to the learning dilemma for recurrent networks of spiking neurons](https://www.nature.com/articles/s41467-020-17236-y)
