---
layout: paper-part
slug: rasp
part: thoughts
title: "Additional thoughts · Thinking Like Transformers"
---

<!-- Tracr is a compiler from code to linear algebra!? -->

Caveats. The link between the RASP program and a learned transformer for the same task is tenuous.
NNs often learn unusual representations and solutions to problems.
Just because RASP shows us that a transformer can solve a problem in a certain way, does not mean a transformer can learn to solve the problem.

Other interesting work on transformers

- (restricted) transformer to code. [@friedman2023learning]
- Similar to the distill.pub circuits investigation of cnns. https://transformer-circuits.pub/ explores transformers in a similar visual / interpretable way.
- [@zhou2023algorithms] RASP-Generalization Conjecture: Transformers tend to length generalize on a task if the task can be solved by a short RASP program which works for all input lengths
- [transformer superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
