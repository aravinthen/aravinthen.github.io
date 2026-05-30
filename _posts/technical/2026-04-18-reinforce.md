---
title: "From REINFORCE to Actor-Critic"
category: technical
math: True
---

# A War Story: from REINFORCE to Actor-Critic
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
\nabla v_\pi(s) &= g(s) + \sum_{s'} P(s \rightarrow s') \left[ g(s') + \sum_{s''} P(s' \rightarrow s'') \nabla V_{\pi} (s'') \right] \\
&= g(s) + \sum_{s'} \left[ P(s \rightarrow s')  g(s') + \sum_{s''} P(s \rightarrow s') P(s' \rightarrow s'') \nabla V_{\pi} (s'') \right] 
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
Find my implementation of `REINFORCE` [here](https://github.com/aravinthen/deep_rl_experiments/blob/main/algorithms/reinforce/vanilla.py).

I'm working with a standard `gym` environment called `cartpole-v1`. This is a very simple game where your goal is to balance a stick upright by simply rolling it left and right. This is a simple game, but `REINFORCE` is a simple algorithm and one that typically doesn't actually work for most problems. My goal here is to iteratively improve `REINFORCE` so that it can successfully solve this simplest of problems.

The network that I'm using - for the very first example - is incredibly simple: a single linear regressor. 

```
self.model = nn.Sequential(
   	         nn.Linear(obs, action)
	     )
```
An observation is passed to this network, which then provides a tensor representing the action space. This is then sampled from via `torch.distributions.Categorical` - the action probabilities are stored whenever they're calculated.

I don't expect that anything will really happen here, but it'll at the very least give me some "low-hanging fruit" to pick off during my first investigations.

The implementation follows a simple structure:
* A `run_policy` function that simply samples the policy for a single episode, stores the reward trajectory in a class attribute and then returns the cumulative reward (although this last feature is mostly used for testing). 
* An `update` function that   
    1. Takes a stored reward trajectory (or generates one if there isn't one already) and iteratively apply the discount to build a returns list for the full trajectory,
    2. Calculate the subsequent policy loss with respect to the `REINFORCE` update rule,
    3. Ready the optimizer, sum the policy loss and update the parameters of the agent policy.
    4. Reset storage.

Simple enough. How does it do?

### Results
My experiment, which can be found [here](https://github.com/aravinthen/deep_rl_experiments/blob/main/experiments/reinforce_soln.py), has the following structure:
1. For a fixed set of ten seeds, I carry out a run of a hundred episodes.
2. The episodic rewards are logged per timestep for each seed,
3. I take the mean and standard deviation of the rewards over each seed for each timestep.
4. I plot and get...

![reinforce](/images/shitty_reinforce.png)

Haha! That sucks! I expected nothing less. Let's fix up some low-hanging fruit and see how much we can push the simple `REINFORCE` algorithm.

## Low-hanging fruit
### Hidden layers
The first thing we can do is add a hidden layer into our network. I'm going to add just one hidden layer for now, just for simplicity's sake. The results are as follows:

![hiddens](/images/better_reinforce.png)

Okay. That definitely has an effect. How much of an effect? It would be very, very disappointing if all I had to do was increase the number of layers...

![morelayers](/images/big_layers.png)

Much nicer, but the effect dies down after around 512 hidden units. Nice!

### Normalized returns
Our next change will consist of normalizing the rewards: a common step in reinforcement learning in general. The goal of this change is to stabilize training and ensure consistency in the optimization landscape: in general, normalized outputs make it easier to tune hyperparameters.

We can normalize here with respect to the mean of the rewards like so:

```
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```
where the standard deviation is nudged to prevent blowup. The result can be seen as follows:

![normreturn](/images/normalized_rewards.png)

Very little difference - nothing to write home about, that's for sure.

### Hyperparameter tuning
A standard means of tuning neural networks is to simply tweak hyperparameters as a means of getting learning dynamics under control. What would happen if we changed the learning rate?

My first instinct was to drop the learning rate to `1e-4`. This was... a disaster:

![lrdrop](/images/lr_drop.png)

As it happened, changing the learning rate also made larger networks less powerful. I dropped the network size and then carried out four runs to assess the joint effect of the learning rate and network size:

![lrhike](/images/learning_rate_network_size.png)

Very, very interesting. These experiments were run with a higher learning rate of `1e-2`: note the enormous variance in the rewards and the deep instability that emerges with a larger network!

Just to fully explain the issues we're facing, observe the following graph:

![lrhike](/images/systematic_loser.png)

This is really instructive. What we're seeing here is
* The algorithm actually learns how to balance in the short term.
* Updates become very, very unstable as the episode length grows. Despite the fact that we do in fact get to a reasonably good score, we *don't* see a consistent score. In fact, all we see is enormous variance!

After analysis, we can see the following:
1. A higher learning rate improves the score, but this is a transient effect and prevents higher scores after what seems to be the use of temporary heuristics.
2. A lower learning rate doesn't do anything: the score doesn't change at all.

What this is telling me is that REINFORCE is getting trapped in a loss minima... but I wondered to myself: why is this not more explicit?

I've been running all of my experiments for just a hunded episodes. I've been doing so because in other contexts, when I have to test an RL algorithm, 100 episodes is typically enough to at the very least get some reasonable learning in.

The issue here is that REINFORCE is not a modern algorithm and my expectations of solving an environment using it are probably somewhat mismatched. Running for 500 episodes clearly demonstrates that maybe... I should have just run the algorithm for longer.

![long_run](/images/long_run.png)

The traditional solution to this problem is to tame the gradient updates with a baseline, so this will be our next step. I don't doubt that we could continue to squeeze out performance by tweaking hyperparameters and varying network sizes, but where's the fun in that?  

## REINFORCE with a baseline
Sutton and Barto advocate for using another learned representation to represent the value function: this is a baseline, and is typically subtracted from the total reward in order to reduce variance in policy gradient methods. There are two basic reasons why this usually works:
1. If all rewards are positive, subtracting a baseline value allows us to distinguish between actions worse than a baseline and actions better.
2. Removing the baseline allows centering of gradient updates. This naturally reduces variance and is supposed to make learning much faster.

The main difference here is in the update rule,

$$
\theta_{t+1} = \theta_t + \alpha (G_t - b(S_t) \nabla \log \pi(A_t \vert S_t, \theta_t)
$$

where the addition of a baseline $$b(S_t)$$ represents the baseline. 

Does it, now? Let's implement REINFORCE with a baseline. I've implemented this as a child class of my original algorithm over [here](https://github.com/aravinthen/deep_rl_experiments/blob/main/algorithms/reinforce/baselines.py), although it required me to respecify the `run_policy()` and `update()` methods anyhow. 

### Comparison
This is the performance of REINFORCE, both with and without a baseline.

![comparison_bl](/images/comparison.png)

The use of a learned baseline is, for the most part, more effective than without. I'm surprised at the variance, though: there's not *really* much of a reduction in variance. For the most part, the variance is actually *higher* with the baseline, although it ought to be remembered that the baseline value function approximator is learning the correct value *throughout* the training process. The learned value of the first step itself changes as the agent gets better, which is another occurance of a moving target. 

## All the way to Actor-Critic 
I was quite disappointed with REINFORCE with a baseline, so I went ahead and implemented the true Actor-Critic method from with PPO and other more advanced techniques spring forth. 

The main difference between REINFORCE with a baseline and the Actor-Critic method is that the former is essentially a Monte-Carlo method, whereas the latter is a temporal difference method. In practice, this manifests as a slightly more complex update step, where the actor (policy) and the critic (value function) are updated simultaneously,

$$
\begin{align*}
\theta_{t+1} &= \theta_t + \alpha \nabla \log \pi(A_t \vert S_t, \theta_t) \delta_t \\
w_{t+1} &= r_{t+1} + \beta \nabla V(S_t) \delta_t
\end{align*}
$$

The ubiquitous temporal difference error shows up as

$$
\delta_t = r_{t+1} + \gamma ( V(S_{t+1}) - V(S_t) ),
$$

and this is really the core of how things change. It ought to be noted that, as usual, the critic update is conducted via stochastic gradient descent: this makes it very simple to include via `PyTorch` (as implemented [here](https://github.com/aravinthen/deep_rl_experiments/blob/main/algorithms/reinforce/actor_critic.py)). In general, the basic class structure of the policy gradient method lends really nicely to such extensions, so it barely took an hour to get the following result:

![actorcritic](/images/actor_critic.png)

We've definitely solved the issue of variance. This result seems good enough, but note that the results are being generated *per step*. We're able to generate reasonable results within just a thousand steps of `cartpole-v1`! This lends credence to the idea that Actor-Critic is *data efficient*. It's easy enough to motivate why this is the case: training steps are occuring with every environment step instead of after every game.

# Conclusion
This was a journey, but it's one that has been well-trodden. I only really went along with this so I could get a bit of practice with PyTorch, but the most useful part of this exercise wasn't the implementation but rather the clarity of thought that reaching a point where I could implement that algorithm. I now have a comfortable understanding of Actor-Critic - a pretty unshakeable model in my mind of
1. Why it is used,
2. Why it was developed compared to REINFORCE
3. Where future developments might lie. 

All in all, a good project.

Now, speaking of those future developments, it's on to the `PPO` algorithm. We move on to modern RL! :) 
