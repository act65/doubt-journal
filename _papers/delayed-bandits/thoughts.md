---
layout: paper-part
slug: delayed-bandits
part: thoughts
title: "Additional thoughts · Linear Bandits with Stochastic Delayed Feedback"
---

### A prior on delay distributions

In this setting it seems there is a natural prior;
the distribution over delays, $D$, probably has a long tail that exponentially decays.

$$
p(D) \propto e^{-D} \\
$$

This prior suggests an alternative approach.
The more time that passes after we have chosen an action, the less likely it is we that we are going to see a reward.

<side>there is probably a more principled way, via Bayes, to derive this, or something like it.</side>

$$
\hat Y_{s, t} = e^{-\beta (t-s) (1-Y_{s, t})} \\
$$


This starts by assuming each action is likely to give a reward of $1$, but the longer that $1$ is not observed, the more we decay it's value.

Rather than needing to act optimistically with respect to an estimated model. This approach builds optimisim into the model.

### Setting extensions

#### Allow delays to be action dependent

In the delayed linear bandit problem, delays are sampled from the distribution $D_t \sim p(\mathcal D)$. This could be easily extended to $D_t \sim p( \mathcal D \mid A_t)$.

This setting is interesting as the learner must;

- model the delays.
- tradeoff near-term performance gains via feedback from actions with small delay vs potential performance gains from feedback about actions with longer delays.

Intuitively, a good heuristic might be: pick actions with long delays first. And while they take their time to convert, pick the actions with shorter delays.
Otherwise, if we pick actions with short delays first then we are left with nothing to do while we wait for actions with long delays to convert.

#### Delayed and accumulating rewards

Rather than the Bernoulli rewards in $\{0, 1\}$ for a past action which gets revealed  after some delay.
How about an accumulating reward?

After taking an action, at time $s$, we observe the future rewards ${y_t(a_s)}$s. We care about maximising the accumulation of all those rewards incurred by the action $A_s$, ie $r(A_s) = \sum_{0}^{\infty} y_t(A_s)$.

In this setting we would require;

- actions do not effect the outcomes of other actions,
- the accumulated rewards are bounded $\sum_{0}^{\infty} y_t(A_s) < \infty$.


<!-- #### Delayed and aggregated rewards

Rather than $y_s^t$, we recieve $\hat y^t = \sum_s y_s^t$

Now we have a credit assignment problem. Leaving the player wondering;

<blockquote cite='Mr. Nobody'>
What did I do to deserve this?
</blockquote>

Without some kind of structure in the problem, learning is impossible!?
Unless we add a noop action? Can simply wait??
If we get a reward and have only take a single action then we can assign credit.
If we have taken n actions and recieve a reward then we can rule out actions that were not taken, and assign credit based on how long ago the actions were taken?!! -->
