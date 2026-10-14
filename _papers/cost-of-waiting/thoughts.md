---
layout: paper-part
slug: cost-of-waiting
part: thoughts
title: "Additional thoughts · The Hidden Cost of Waiting for Accurate Predictions"
---

<!-- ranking metrics evolve over time -->

- [ ] A more general framework for probablistic ordering?

The assumptions I'd want to relax:

- future interventions cannot undo failure out of the population
- fixed unit size (non-adaptive to need). can only be applied once.
- constant failure probabilities
- the process of observing is uniform accross the population

What about a setting where;

- adaptive intervention size.

Their framing of the problem isn't really an allocation problem? It seems closer to just a (probablistic) ranking problem?

It's like have a bandit problem and seperating acting into; observing and allocating?
(bandit problem where we can pull multiple levers. ie the different interventions? related to parallel bandits?)

- [ ] A key aspect fo this kind of allocation problem should be; discounting? I mean I could have not spent anything now, waited 20 years (earned interest) and then interveened.

All outcomes are given the same weight.
