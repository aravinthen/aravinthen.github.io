---
title: 10. Landing on PPO
category: literature
math: True
---
# Literature review - Week 10
This entire literature review was written during a flight from Delhi to London Heathrow. I wasn't really trying to be ambitious here: as of writing, I'm at a point where I ought to be able to understand the content of the *Proximal Policy Optimization* paper. However, given that I have about 8 hours of time to kill I think I'm going to add a section here that summarises *all* of the ideas that make this exceptionally important technique work. 

## Prerequisite ideas
PPO is a *policy gradient method*. This is a departure from the standard ideas of reinforcement learning, which are typically represented in a tabular form that in *some way* derive from the Bellman equation. This entire class of methods has one chapter dedicated to it in Sutton and Barto, which is our starting point.

### Sutton and Barto: Chapter 13 - Policy Gradient Methods
The majority of Sutton and Barto is based around the use of *value* functions. The policy is typically denoted as either greedy with respect to the value function or $$\epsilon$$-greedy, where it randomly selects an action at a small probability. The first foray into machine learning techniques applied to reinforcement learning lies in value function approximation, where the value function is replaced by some kind of learned model. This is typically some form of regressor. Th main application of a function approximator is to defeat the curse of dimensionality (1), where the number of states for which a value is to be calculated are too many to ever fit on a usable tabular representation.

Policy parameterization is a completely new approach to reinforcement learning with function approximators. The value function is cast aside (or at the very least assumed to be approximated effectively). Instead, the *policy* itself is represented by a function approximator. There are a number of very powerful reasons to do this:
1. They allow the use of reinforcement learning on continuous problems.
2. The policies can represent probability distributions, which allow for stochastic policies to be represented. These stochastic policies have exploration built into them, with *policy updates* being an effective means of gradually sharpening the distribution over the action space. 

Paramterized policies are updated via a version of gradient descent,
$$\theta_{t+1} = \theta_t + \alpha \mathbb{E}\left[\nabla J(\theta_t)\right]$$,
where $$\theta$$ represents the parameters of the policy, $$\alpha$$ is a tunable step parameter and $$J$$ is a generic measure of performance. This is a very general formulation: $$J$$ could be anything as long as it represents a useful form of the reward signal. By adopting this approach, we reduce reinforcement learning into a classical optimization problem. 

The most important result in the study of policy gradient methods is the *policy gradient theorem*, which is a specific form for the value of the gradient of $$J$$,
$$\nabla J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a)\right]$$.
This is an exceptionally powerful technique in that it makes it unnecessary to differentiate the dynamics of the policy - just replace it with an expectation. 

The policy update rule can be simplified. Note how the original formulation is over an expectation - this requires an average over actions. The basic `REINFORCE` algorithm that implements this idea has two specifications:
1. It dispenses of the average over actions and instead uses Monte Carlo sampling to adjust the contribution of an action to the parameter update.
2. It employs the full return as the measure of performance. Note that the use of the full return makes this particular method a Monte Carlo approach.

There are some quite serious issues with this general approach. The first is that it can become extremely unstable: a tiny change in $$\theta$$ could cause enormous deviations in the reward signal, which overall will slow down training to often times unsustainable levels. We fix this by introducing a *baseline* that subtracts from the full return. This baseline is typically an estimated value function, which basically implies that you update your parameters based on how much they deviate from your estimate of the value function. This depends on the strength of your value function approximation, which oftentimes can be pretty bad. 

The **actor-critic** policy gradient algorithms emerge from two considerations:
1. What if you could update your value function alongside your policy?
2. The current formulation of the policy gradient can't actually assess the action itself in the context of policy update. Is there a way of analysing the strength of the action? The previous method, `REINFORCE`, doesn't actually use the state-value function to inform action strength.

The actor-critic method can most aptly be summarised as the following:
1. The actor chooses an action based on it's parameterization. 
2. The value function is calculated for the *next* state that the chosen action leads to, which is then used to generate the temporal difference error: this is then used to adjust the policy. This is the *critic* step, where the value of the next step is subjectively estimated.
3. The actor parameters are modified based on how the actual return measures up to the critic estimate. Essentially, if the actor surprises the critic and produces a state that has much higher reward than anticipated, the actor parameters are subjected to a larger policy gradient update. 
4. The critic parameters (assuming that the critic is also parameterized) are also adjusted based on the actual return. 

The rest of the papers here are elaborations upon this core idea of parameterizing the policy, making it both more stable and more effective in a range of situations. 

(1) - I came across this term for the first time during my graduate work, where it was introduced to me in the context of computational physics. As it happens, the very origin of the term *curse of dimensionality* comes from Bellman himself in the context of dynamic programming, which itself is a classical RL method. 

### Deterministic Policy Gradient Algorithms - Silver et al.
This is an extension of policy gradient methods to deterministic policies, but also introduces a very useful metric called the *advantage function*, which is the difference between the state-value and the action-value function. Given that the former is an average over actions, the advantage function serves to isolate the action and thereby reduce variance most effectively. The main trick here is to update the policy in the direction of the Q-value, which is effectively an extension of the policy gradient update but with the Q-function instead.

Specifically, instead of having the gradient being calculated through log probability, we use flow through the Q-function instead. 
$$\nabla J = E[\nabla_\theta \mu_\theta(s)\nabla_a Q(s,a)]$$

Note that *this technique is not a deep reinforcement learning algorithm*. It is made into a DRL algorithm in the next paper.

### Continuous Control with Deep Reinforcement Learning - Lillicrap et al. 
The goal of this work was to develop a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces. The basic way that this is achieved actually draws from work on transforming Q-learning into a deep reinforcement learnign technique (this is discussed in the previous literature review). To summarise, this step takes the actor-critic approach and augments it like so:
1. Employing a replay buffer, which is a set of experiences that are sampled from (without replacement) in order to handle the issue of overly correlated sequence steps.
2. A target network that is trained less frequently than the action selection network is used to stabilise the training process. It should be noted that this is actually a direct extension of the deep Q-learning method in which a double-Q learning approach is employed to avoid issues from variances causing overestimation issues. In this situation, the target Q network is transformed into a *critic* - it is actually given entirely different parameters.
3. Aside from this, the original DPG technique remains intact.

### Asynchronous Methods for Deep Reinforcement Learning - Mnih et al. 
This paper introduces the Asynchronous Advantage Actor Critic method, often referred to as A3C algorithm. 
The *replay* buffer is a tool to remove the correlations that emerge with respect to sequences of steps. It is typically used by sampling state/action/reward tuples at random and then sampling from this list of experiences to train the network. 
The basic issue with this technique is that it effectively forces you to use an off-policy algorithm. The data used to train the policy was not generated by that policy. Does this mean that any application of deep reinforcement learning can't be adapted to the wealth of literature on on-policy methods?

The answer is no. The work in this paper uses parallely generated *batches* per timestep in order generate training data for the network. As the batches are generated independently, generating a set of N trajectory timesteps can essentially provide the same benefits of a replay buffer *whilst allowing the policy to update on a per-timestep basis*. A3C stabilizes learning by decorrelating data via multiple actors generating batches of data, a feature that is only supported via advances in asynchronous gradient updates for neural networks.

## Trust Region Policy Optimization - Schulman et al. 
We now move on to the precursor methods that eventually developed into PPO. It should be noted that PPO is a *simplification* of a significantly more complicated set of techniques. 

The predecessor of PPO is TRPO, a method that itself is a result of extending a prior method (conservative policy optimization) in a way that allow for fast, stable updates. The problem that this paper attempts to solve is to extend this previous technique to handle the large, complex policies that modern deep learning typically handles. 

Trust region policy optimization effectively introduces a new optimization criterion that is completely distinct from the techniques described previously. The basic idea here is that 
1. You can define the advantange difference between two policies as long as their visitation counts are known. This allows one to generate a new policy by simply modifying the old one, whilst also being able to quantify just how much better the new policy is. In theory, this means we can randomly generate a bunch of policies and optimize against those that are better than the existing policy. Problem solved, right?
2. Not so. You get to know visitation counts by *running multiple policies*... which we don't want to do, as
    * it's typically very computationally intensive to run a bunch of policies. 
    * There are so many ways of picking new policies (by tweaking the parameterizations) that simply sampling at random is inefficient. 
3. The first approximation that we need to employ in order to make this a feasible problem is to assume that the visitation counts of the old and new policies are approximately the same. This naturally implies that we need to ensure that the visitation counts don't *actually* diverge too much from each other. This allows us to define a *surrogate objective* (to maximise the relative advantage between the old policy and the new policy) in terms of the *initial policy*. 
4. We ensure this limitation on visitation counts by *implementing a trust region* that in general minimises the "distance" that one can take when generating a new policy from the old one.
5. How do we compute the distance? We use a constraint on the KL divergence. Of course, calculating the full KL divergence isn't possible: that is a big calculation. The *average* KL divergence is employed instead. 

The most prominent issue with this approach is the fact that it's very complicated and necessarily requires very heavily numerical calculation techniques. As part of the algorithm, one must calculate the Fisher information matrix (the Hessian of the KL divergence matrix). This, primarily, is the reason that PPO was introduced in the first place - to reduce the complexity of the calculation from relying on second-derivatives into a pure first-order optimization problem.

## High-dimensional continuous control using generalized advantage estimation - Schulman et al.
The previous paper focused heavily on the policy itself, but less so on the advantage that drives the improvement.
The advantage is a difficult calculation to estimate. As mentioned, in the basic formulation it's a measure of how much better an action is on average compared to an averaged representation inherent in the value function. However, by design this estimation will have a *ton* of variance for complex problems when used in an iteratively improving trading method. The idea only really makes sense as it is when you know the value functions perfectly.... and you typically do not.

The aim of generalized advantage estimation is to inject bias into the calculation in order to reduce the variance, allowing for a more stable and consistent training process. 
The `GAE` estimator,
$$A_t = \sum_k (\gamma \lambda)^k \delta_{t+k}$$,
is specified by two parameters (outside of the TD error $$\delta$$: 
1. $$\lambda$$ - controls how far the advantage estimator looks forward in order to approximate effectively. 
2. $$\gamma$$ - defines the discount factor and incorporates it directly into the advantage calculation. 
This is effectively a bias-variance trade-off. Granted, the paper notes that these parameters are most effective at different ranges - indeed actually *finding* the value of $$\lambda$$ and $$\gamma$$ is to experiment with a range of values.
 

## Proximal Policy Optimization - Schulman et al.
We have reached our destination, with all of the prerequisites required to understand proximal policy optimization. Let's get on with it!
The paper starts by identifying it's goal as an algorithm that is scalable, data-efficient and robust. It notes that
* Deep-Q learning is poorly understood and fails on a multitude of simple problems,
* Vanilla policy gradient methods have poor data effiency and robustness, and most importantly
* TRPO is very complicated and doesn't actually work well with architectures that include noisy elements *drop-out, for instance) or parameter sharing.

PPO is introduced as a method that attains the reliability and efficiency of TRPO without having to deal with te quite intensive calculation steps. This is achieved not by placing heavily, nasty constraints... but by clipping the probability ratio of the probabilities. What does *that* mean??

We've discussed the surrogate objective of TRPO above. PPO introduces a new objective,
$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[ \text{min}(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right]$$, 
where $$r(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$ is the importance sampling ratio mentioned countless times, $$\hat{A}_t$$ is the advantage estimate (in this situation the GAE estimator) and $$\text{clip}(a, b, c)$$ is a non-smooth function that, when combined with the advantage function, prevents the ratio between the old and the new probabilities from ever crossing the distance defined by $$\epsilon$$.

The PPO algorithm is literally a manual means of ensuring that the same constraint of TRPO (that the policies don't change too much) is satisfied. There's something a little deeper here, though: in TRPO, you're working with a constrained optimization problem. PPO on the other hand is actually an *unconstrained* optimization problem - the loss itself is just clipped. Genius!

The paper also describes an adaptive KL approach, where the parameter that controls how much the KL divergence is permissible from the old policy to the new is adapted so that some target value of KL divergence is attained each timestep. This was reported to be less effective than the simpler PPO calculation. 

Now that you don't have to calculate all sorts of complex second derivatives, all you have to do is to carry out gradient descent on the new objective function, which doesn't require second derivatives to be calculated. The results aren't *absolutely amazing, beating every game*, but they're **very** good.