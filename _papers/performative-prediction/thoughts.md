---
layout: paper-part
slug: performative-prediction
part: thoughts
title: "Additional thoughts · Performative Prediction"
---

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

- [ ] Model based performative prediction!? We have some estimate / knowledge of $\nabla D(\theta)$? (kinda like the transition fn) This allows us to optimise without requiring extra samples!?
