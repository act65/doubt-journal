---
layout: paper-part
slug: rl-as-inference
part: tutorial
title: "Tutorial · Probabilistic Inference in Reinforcement Learning Done Right"
---

## Overview

The 'RL as inference' approach frames RL as a probabilistic inference problem by introducing a binary optimality variable. This variable is 1 if the state-action is optimal and 0 otherwise.

Our goal is to construct posterior

$$
M = \{\mathcal S, A, L, P, R, \rho\}
$$
