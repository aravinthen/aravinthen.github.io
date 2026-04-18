---
title: REINFORCE: From theory to implementation
category: technical
math: True
---

# REINFORCE: From theory to implementation
I've spent much of the last month slowly chipping away at understanding classical reinforcement learning algorithms, which you can read about here. My goal now is to make the transition to true deep reinforcement learning algorithms.
The field of deep learning is concerned with the engineering of powerful function approximators (specifically, multi-layered neural networks) for analysis, pattern recognition and decision making. The core of these function approximators is a set of parameters that are typically tuned using an optimization procedure with respect to some target value. 

Deep reinforcement learning, then, is the use of such paramaterized function approximators to develop autonomous decision making agents. The tuning is necessarily based around tweaking these parameters with respect to a reward signal. Basically,
* if the neural network that represents your agent (which we'll call a policy) produces an action that does something good, it'll get high reward. 
* That high reward should somehow be used to tweak the parameters of the agent policy so that it does that good thing more often.

As it happens, ideas of using parameterized representations of agents did not emerge from deep learning literature. They predate the hey-day of deep NNs by a good few decades, constituting an entire class of reinforcement learning algorithms called policy based methods. They don't necessarily have to be represented by deep neural networks, which I guess makes it so that they're not true *deep* reinforcement learning algorithms. That being said, the ideas developed with respect to policy based methods are essential to deep reinforcement learning and are well worth a thorough investigation.

The goal of this note to introduce policy-based reinforcement learning methods, as well as how they might be implemented using all the functionality of a modern deep learning library like `PyTorch`. The specific method that I'll be implementing (`REINFORCE`) is a very simple class of policy based method that attempts to optimize a parameterized policy directly. I don't anticipate I or anybody I know will ever seek to use the code or method, but I'm hoping that it can at the very least be useful as a reference.

## Introduction
A policy-based method is an approach to reinforcement learning where the agent directly learns a mapping from states to actions. This is a departure from the core of the discipline, which is mostly a means of assessing *value* functions. Whilst value functions are very important regardless, they are merely tools for the agent that allow it to accurately update it's internal state.

Policy-based methods present a real philosophical difference to the main alternative, which are called value-based methods. Value methods estimate the worth of an action, with that estimate of worth being used to reduce the strategy to greedily carrying out the highest-value action. Policy based methods directly scope out the best strategy directly, which gives them significantly more flexibility as an RL technique. 

The first indication of this flexibility comes from the fact that you no longer have to explicitly define your RL approach with respect to the Bellman equation,

$$
v(s) = \sum_a \pi(a\vert s) \sum_{s' \in S} p(s', r \vert s, a) \left[ r + \gamma v(s') \right].
$$

The object of the game becomes to determine an expression for optimising the parameterized policy. Let $$\theta$$ be the parameters that determine your neural network (or any other function approximator). You can improve your policy by improving it with respect to some measure of performance $$J$$. There's no real condition on what $$J$$ need be.

### Policy optimization
How does that optimization take place? How do you modify $$\theta_k$$ so that the new policy $$\theta_{k+1}$$ is more effective than the last?
Take the gradient of your policy performance measure, $$\nabla J$$. This directly provides you with the 'direction' you need to nudge your gradient in order to improve it. Adjusting that step directly provides a means of changing your parameters so that they maximise the policy performance measures,

$$
\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\pi_\theta)\vert_{\theta_k}.
$$

This is what is known as a policy gradient algorithm and makes up foundation of the bulk of the RL techniques that are used heavily in the modern day.

### The policy gradient theorem
It's all well and good have an expression that makes use of the performance measure gradient. How do you actually compute the gradient?
First, let's define the performance objective. For now, we need a simple measure of performance that can be optimized in the first place. Why not start with... the value function, which I just said doesn't have to be used?

Aside: whilst you can use anything as your performance measure, the value function is essentially the foundation of improvement and is still used in effectively every possible performance measure. It just isn't the main objective anymore.

Anyway, moving on, let

$$
J(\theta) = V^{\pi_{\theta}}(s_0),
$$

which is just the value of the starting state. It's worth nothing that we don't need the exact gradient, which anyhow is impossible: we simply need an expression that is proportional to the gradient as the difference is absorbed by $$\alpha$$ anyway. Such an expression for the gradient is given by the *policy gradient theorem*, which is one of the most important results of reinforcement learning as a whole.

Following Sutton and Barto, we prove the policy gradient theorem from first principles. Writing the state value in terms of the action-value,

$$
\begin{align}
\nabla v_{\pi}(s) 
&= \nabla \left[ \sum_a \pi(a \vert s) q_\pi(s, a)\right] \forall s \in \mathcal{S} \\
&= \sum_a \left[ \nabla \pi \quad q_\pi + \pi \nabla q_\pi \right] \\
&= \sum_a \left[ \nabla \pi \quad q_\pi + \pi \nabla \left[p(s', r \vert s, a) (r + v_\pi (s')\right] \right] \\
\end{align}
$$






### Deriving REINFORCE

## Experiments
### Results
## REINFORCE with a baseline
### Comparison
## Conclusion
