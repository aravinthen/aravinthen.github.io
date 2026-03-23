---
title: Literature review 11 - Advanced actor-critic methods
category: literature
math: True
---
# Literature review - Week 11
This will be a shorter literature review than what I usually turn out. My previous literature review was later in the week and I was a bit sick over the last week, but I regained steam later through the week. 

The main theme around this (on-going) literature review was post-PPO actor-critic methods. These are the modern techniques that many advanced applications use and they typically continue the basic trends discussed in the previous methods. One difference, however, is the fact that *scaling* becomes a more important aspect to the problem: this is addressed in the final paper of this literature review. 

## Addressing Function Approximation Error in Actor-Critic Methods - Fujimoto et al.  
Overestimation errors occur in reinforcement learning when noise in the environment causes a *maximisation* bias where the reward assigned to a state is *not* the true expected reward of that state, but a noisy overestimate. This is a very important class of error in reinforcement learning and one that shows up early in the chain of reinforcement methodology: we first encounter it as a flaw in Q-learning when reading Sutton and Barto. 

Many of the papers that I read over the last two weeks have observed the overestimation errors occur in Q-learning and it's deep learning equivalent. As it happens, the same issues show up in the actor-critic formulation as well. 
There is a subtlety to how the error affects training when used in the Q-learning approach versus the actor-critic approach. In Q-learning, an overestimation error will cause the action-value function estimate to be far too optimistic for particular states. Given that Q-learning bootstraps, this will leak value into the other actual-value estimates and in general slow down the training process *significantly*. 
In actor-critic, overestimation simply yields an overoptimistic critic. Policy gradient methods optimise locally, and if your critic is consistently overoptimistic, the actor will fall into a bias itself: it will get stuck recommending sub-optimal actions because the critic is incorrectly telling it that those actions will lead to high rewards. 

The standard way of dealing with this is by using a second network to act as a check on overestimates. This is a safeguard against maximisation bias: a second pair of eyes to provide a damper on any overoptimistic estimates. This is the motivation behind double-Q learning and deep double-Q networks (DDQN), and it also the same kind of logic that is offered in this contribution. 

There is a difference, however, in how double Q-learning is implemented here. The method actually doesn't use the Q-network exchange ideas that are pioneered in previous examples. Rather, two critics are trained simultaneously *and the one that provides the lower value estimate is used for the actor*. This actually favors underoptimism and lower estimates of action-value, which are significantly less damaging than operoptimism. The technique is referred to in paper as *clipped double Q learning*: the basic idea shows up in later methods, too. 

Aside from this, the system employs delayed policy updates to ensure sufficient stability for policy improvement - this is similar to the basic strategy used in deep-Q learning. Target policy smoothing is also used, where some noise is added to the action to reduce the occurrence of sharp peaks in the actor probability distribution. The result is a method that has very strong performance on a variety of complex continuous environments. 

Something that can be seen quite clearly is the fact that all of the techniques discussed here are very general. They can pretty much be assigned to *any* actor-critic technique. You're essentially just using a primitive consensus mechanism whilst stabilising the target and the control policy. It should be explicitly mentioned, however, this this is a *response* paper: the authors are actively trying to fix the issues with the previously discussed DDPG method.

## Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor - Haarnoja et al. 
In statistical physics and optimisation, it's quite common to employ an annealing process that balances exploration and exploitation. This can often be visualized as a very flat probability distribution that, with respect to the lowering of some parameter (the temperature in statistical physics) becomes sharper and more specific. You start off with all possibilities being equal and you end up in a very specific distribution of states. This is *entropy*: going from a high entropy state to a low entropy state.

Typically, due to the exploration that takes place at high entropy, the low entropy state that you end up in is usually some kind of optimum. In statistical physics, this is usually the state of your system that has a very low energy - maybe even the ground state of your system (or close to it). There is a similar correspondence in optimization (in fact, there's even a technique that I've used quite heavily in the past that analogizes the statistical physics approach for complex minimization problems).

The Soft-Actor critic approach was my first introduction to a methodology that follows this protocol in reinforcement learning. In fact, when I was reading around this topic, it became quite apparent that modern implementations of SAC actively anneal based on a target entropy. How cool is that!? Statistical physics and reinforcement learning. The two great interests of my life, converging into one. 

SAC is off policy and makes use of a replay buffer. Furthermore, unlike a lot of the other off-policy algorithms that I've been looking it, the initial stochastic policy is fundamental to it's formulation. A similar technique of double critic networks is used here too, with five total networks being trained (the actor, the twin critics and the twin target critics). The main contribution to this approach is of course the entropy maximisation aspect. However, as can be seen, SAC is really a full-featured combination of some of the most effective technqiues developed in the three years preceding its publication. 

## IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures - Espeholt et al. 
I initially thought that IMPALA was an architectural advance. I mean, it technically is:
* Actors are decoupled from learners, with actors living in CPU cores whilst learners train n GPUs,
* Actors send full trajectories to the learners and receive the latest trained policy parameters in return.

This naturally results in very, very high throughput training. The only issue is however a basic issue that emerges whenever you try and distribute: training lag. This takes place when 
1. The learner L sends it's policies to actors A and B.
2. Actor A sends a trajectory back to the learner, which then trains to produce a new policy L+.
3. Actor B sends a trajectory back to L+. However, Actor B trained with a different policy: the one corresponding to learner L! This renders the training process to be off-policy, and as we know from earlier investigations into deep-Q learning, this causes a distributional discrepancy where data from the old policy is no longer quite as relevant to the new policy.

The genuinely incredible advance that this paper pushes is *actually* an algorithmic advance: the introduction of V-traces.
Traces in reinforcement learning are markers that determine lingering relevance. Eligibility traces, for instance, are short term memory markers that determine how much of a past action sequence contributes to that reward. 
V-traces are a different kind of marker that corrects for the difference between two policies. It does this by calculating weights based on importance sampling: the ratio between the target policy and the actor policy. The goal here is to determine just how much a given trajectory should influence an update.

The importance sampling is controlled via clipping, with two specific threshold parameters being specified: these parameters control the temporal difference updates *as well as* the policy importance sampling. It took me *ages* to figure out what is actually going on here:
* $$\rho$$: this parameter clips the probability of the action being taken by the actor vs the current learner. What I'm getting here is that this is something of a measure how much we trust the actor's data. $$\rho$$ is a smooth parameter. If you set it to a very low value, you actually get the actors old policy whereas a higher parameter setting would completely recover the target policy. By clipping, you're working on a compromise between the target and the actor policy.
* $$\kappa$$: this is the temporal difference error parameter for multi-step trajectories that controls the speed of propagation with respect to the new policy. This ensures that the TD error isn't propagated backwards if the actions taken by the actor are unlikely under the new policy. 

The result here is that we have an *active* means of transforming an engineering problem (the distribution and asynchronicity of the decoupled actors) into a mathematical problem that can be solved with importance sampling. *Utterly* genius!