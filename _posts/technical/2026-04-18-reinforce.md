---
title: "REINFORCE: From theory to implementation"
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
It's all well and good have an expression that makes use of the performance measure gradient. How do you actually compute the gradient or a performance measure?
First, let's define the performance objective. For now, we need a simple measure of performance that can be optimized in the first place. Why not start with... the value function, which I just said doesn't have to be used?

Aside: whilst you can use anything as your performance measure, the value function is essentially the foundation of improvement and is still used in effectively every possible performance measure. It just isn't the main objective anymore.

Anyway, moving on, let

$$
J(\theta) = V^{\pi_{\theta}}(s_0),
$$

which is just the value of the starting state. It's worth nothing that we don't need the exact gradient, which anyhow is impossible for anything that is too complex to be presented as a table: we simply need an expression that is proportional to the gradient as the difference is absorbed by $$\alpha$$ anyway. Such an expression for the gradient is given by the *policy gradient theorem*, which is one of the most important results of reinforcement learning as a whole.

Following Sutton and Barto, we prove the policy gradient theorem from first principles. Writing the state value in terms of the action-value,

$$
\begin{align*}
\nabla v_{\pi}(s) 
&= \nabla \left[ \sum_a \pi(a \vert s) q_\pi(s, a)\right] \forall s \in \mathcal{S} \\
&= \sum_a \left[ \nabla \pi \ q_\pi + \pi \nabla q_\pi \right] \\
&= \sum_a \left[ \nabla \pi \ q_\pi + \pi \nabla \left[p(s', r \vert s, a) (r + v_\pi (s')\right] \right] \\
\end{align*}
$$

There are a few important substitutions taking place here. The first is the product rule in the second step, followed by the substitution of the Bellman equation in the following line. Note also that the reward drops due to the fact that the gradient of the reward is zero. At this point, it's clear to see that you can repeat the step of Bellman equation substitution until whenever you like. This is called *unrolling* and is a common strategy in proving general trajectory features - it's also seen in the policy improvement theorem that underlies most of classical value-function based reinforcement learning. 

Moving on, if we unroll the policy for infinite timesteps, we'll get

$$
\nabla v_{\pi}(s) = \sum_{x \in \mathcal{S}} \sum_{k=0}^\infty \text{Pr}(s \rightarrow x, k, \pi) \sum_a \nabla \pi(a \vert x) q_\pi(x,a).
$$

I saw this expression in Sutton and Barto and my immediate reaction was "...**huh?**". In fact, the only reason that I now understand what's going on here is because I remember seeing and interpreting the idea of "visitation count" in the trust policy region optimization paper. In any case, to explain, I'm going to simplfy the derivation by abstracting away the reference to the sum of the gradients,

$$
g(s) = \sum_a \nabla \pi(a \vert s) q_\pi(s,a).
$$

With this, we can represent the value gradient as

$$
\nabla v_\pi(s) = g(s) + \sum_{s'} P(s \rightarrow s') \nabla v_\pi (s').
$$

We can do this because the reward and the actions are essentially factored out by including them within the definition of $g$. If we carry out the unrolling process now, we'll get something that looks like

$$ 
\begin{align*}
\nabla v_\pi(s) &= g(s) + \sum_{s'} P(s \rightarrow s') \left[ g(s') + \sum_{s''} P(s' /rightarrow s'') \nabla V_{\pi} (s'') \right] \\
&= g(s) + \sum_{s'} \left[ P(s \rightarrow s')  g(s') + \sum_{s''} P(s \rightarrow s') P(s' /rightarrow s'') \nabla V_{\pi} (s'') \right] 
\end{align*}
$$

This is a visitation measure: the total number of times you expect to visit a state across an entire episod. Each individual term in the unrolling (which has been carried to two steps above) is the probability of being in that state at that exact moment (so, two timesteps in).

This can now be replaced with the shorthand $$\text{Pr}(s \rightarrow s')$$, and the apostrophes can be replaced by time indices. What we're really getting here is
1. At $$t=0$$, the probability of being in state $$s_0$$ is 1. The weighting over $$g$$ is simply 1, as can be seen in the original expression.
2. At $$t=1$$, the weighting over $$g(s_i)$$ for some state $$s_i$$ is $$\text{Pr}(s_0 \rightarrow s_i)$$. 
3. So on.

Sub in $$g(s)$$ and you have the original expression for $$\nabla v_\pi(s)$$. This expression is really just the gradient of $$v$$ represented as a sum over all states in the future. 
The quantity $$\sum_k^\infty \text{Pr}(s_0 \rightarrow x)$$ can be interpreted as the expected number of timesteps spent in $$s$$ during a single episode - you can verify this by assuming that given a hundred timesteps and a generic hitting probability of 0.3 for some state $$s$$, you'd expect to hit that state thirty times. Essentially, the quantity tells yo uhow much weight a state ought to have in the gradient update based on the total expected occupancy. 

Substituting $$\eta(s) =\sum_k^\infty \text{Pr}(s_0 \rightarrow x)$$ and switching to the initial defintion of our performance measure $$J$$, we get

$$
\begin{align*}
\nabla J(\mathbf{\theta}) &= \sum_{s \in \mathcal{S}} \eta(s) \sum_a \nabla \pi(a \vert s) q_\pi(s,a) \\
&= \sum_{s'} \eta(s') \sum_s \frac{\eta(s)}{\sum_{s'} \eta{s'}} \sum_a \nabla \pi(a \vert s) q_\pi(s,a) \\
&= \sum_{s'} \eta(s') \sum_s \mu(s) \sum_a \nabla \pi(a \vert s) q_\pi(s,a) \\
\end{align*}
$$

These steps shift from considering the visitation counts to representing a full probability distribution over the visited states: this is called the on-policy distribution and essentially weights the localized gradient updates by how often they are visited by the policy overall.

Now, the issue here is the calculation of $$\sum_{s'} \eta(s')$$. As mentioned, this is the expected number of total time steps spent in all states during a single episode. Does this mean that in order to even use this expression, you have to *finish an episode*? That's not useful! The trick here is to remove the hard condition of equality and work with proportionality instead. As a result, you get 

$$
\nabla J(\mathbf{\theta}) \propto \sum_s \mu(s) \sum_a \nabla \pi(a \vert x) q_\pi(x,a)
$$

If $$\mu(s)$$ is a probability distribution of states and we're summing over the states, then that means that the above is an **expectation**. 

$$
\nabla J(\mathbf{\theta}) \propto \mathbb{E}_{\pi} \left[ \sum_a \nabla \pi(a \vert x) q_\pi(x,a) \right]
$$

Through this, we have a mathematically rigorous way to directly optimize a policy performance without requiring any knowledge of the environment's internal dynamics. It took me a while to realise the raw power of this result, but in a single sentence: it allows you to take meaningful derivatives of agent performance without differentiating through a state distribution (in model free settings). 


### Deriving REINFORCE
As can be seen from the expression of the performance measure gradient, all we need to do in order to update our policy parameters is to increment them in a way that 
$$
\theta_{t+1} = \theta_{t} + \alpha \left[ \sum_a \nabla \pi(a \vert x) q_\pi(x,a) \right].
$$
That is, we just need to be able to estimate an expected value. As mentioned in Sutton and Barto, you can just use this in order to generate a valid policy gradient rule: just update according to a single realisation of $$\sum_a \nabla \pi(a \vert x) q_\pi(x,a)$$ and be done with it. I get the impression that this is quite messy.

On a basic level, the raw policy gradient theorem requies you to calculate the actual gradient of the policy. We'd rather not calculate hard derivatives at all, so one way of simplifying the expression is operate over expected value and simply *sample* what we need. We can inject the probability distribution into the main expression for the parameter update by simply multiplying by unity,

$$
\nabla J(\theta) \propto \mathbb{E} \left[ \sum_a \pi (a \vert S_t, \theta) q_\pi (S_t, a) \frac{\nabla \pi (a \vert S_t, \theta)}{\pi(a \vert S_t, \theta)}\right]
$$

Following the mathematics and applying the famous *log derivative* trick, we get

$$
\nabla J(\theta) \propto \mathbb{E} \left[ G_t \log \nabla \pi(A_t \vert S_t, \theta) \right] 
$$

A bit of explanation here:
1. $$A_t$$ is the replacement of the sum over actions with a single sample action from the policy. 
2. $$G_t$$ comes from the fact that $$\mathbb{E}_\pi[G_t \vert S_t, A_t]$$ is nothing more than the definition of the value function. 

The final rule, and the rule that we'll be using to carry out `REINFORCE`, is

$$
\theta_{t+1} = \theta_t + \alpha G_t \nabla \log \pi(A_t \vert S_t, \theta_t)
$$

One outstanding question remains. How on earth do we calculate the gradient of the log probability?
Well, the answer to this is two-fold:
1. If you're using standard functional representations of a parameterized policy, then you... simply differentiate it.
2. If you're using a neural network, then you forget about it and let backpropagation in PyTorch do the job for you. :)

And so, we come to the first of a policy gradient techniques. Now it's time to test how it works!

## Experiments
### Results
## REINFORCE with a baseline
### Comparison
## Conclusion
