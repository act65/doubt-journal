---
layout: paper
slug: eye
title: "What if Eye...? Computationally Recreating Vision Evolution"
authors:
year: 2025
venue: arXiv
ref: https://arxiv.org/abs/2501.15001
permalink: /papers/eye/
# Migrated from _drafts/2025-40-10-eye.md (filename month/day were transposed; an identical
# copy at 2025-04-10-eye.md was dropped). `read` is the corrected filename date.
read: 2025-04-10
depth: read
rating: 2
tags: [neuroscience, rl]
question:
tldr:
notebooks: []
related: []
draft: true
published: false
---

## Review

*I haven't written this yet.*

## Problem

## Overview

The three claimed contributions, and what I made of them:

> (1) By strictly changing the visual task an agent is subject to, orientation (Navigation) task vs. object discrimination (Detection) task [4, 31], we observe a bifurcation in our evolved agents between camera-type and compound-type eyes (Figure 1a);

(sure. but is this a statistical fluke or are these optimal solutions?
wait. so all this is really say is; if you are navigating a maze, it helps to have eyes capable of seeing walls pointing in many directions. and if you need to discren between different objects, it helps to have eyes capable of distinguishing highly detailed features.
)

> (2) We show that the emergence of optical structures, such as focused lensing from primitive simple eyes [5], is a key innovation that addresses the fundamental trade-off between light collection and spatial precision (Figure 1b);

> (3) We reveal that sensory acuity and neural capacity scales as a power-law, where decreasing task error requires complimentary improvement in both dimensions

## Additional thoughts

- so we have reduced vision to a set of problems; navigation, detection, and tracking. What about; segmentation, ...?
    - is chasing / hunting (say a bird chasing a fly) simply a combination of navigation, detection, and tracking? or does it require something more?
    - what about reading?
    - a heirarchy of visual tasks? if you can track, ...?
- what are all the factors that affect the evolution of vision? (what about head stabilisation, dead reckoning, )
- why are we using RL here? cos we find it hard to answer; how 'useful' is X for acheiving Y?
- why use ES here? cos there are too many gene combinations to explore.
- why aren't simple physical models enough? ie understanding the relationship between refractive index, lens shape, and focal length -> resolution?
