---
title: Lessons from Classical RL
category: technical
math: True
---

As a physicist by training, I'm no stranger to having to go through the classical results of a discipline before reaching the fun stuff: students of physics must essentially master classical mechanics and electromagnetism before the weird stuff like quantum mechanics and general relativity can be truly appreciated. Some physicists (like myself) never really leave the classical side of the discipline: there are many beautiful results and outstanding questions in fields like soft matter physics and biophysics that scientists are still dedicating lives to resolving. 

Despite this, I was a little wary of spending too much time on the classical results of reinforcement learning. It stands to reason that, in an age where the astounding results are driven by modern *deep* reinforcement learning, moving immediately to the glamorous techniques of the field would be the fastest path to generating the best results. Having read a ton of papers on deep reinforcement learning and modern artificial intelligence, it appears that 
1. Theoretical justification, which is super important in classical RL, is typically glossed over or absent from the application of deep learning to problems in reinforcement learning,
2. The power of deep reinforcement learning doesn't come from the algorithmic side - PPO is quite simple conceptually despite its effectiveness. The amazing results appear to originate from the use of increasingly sophisticated deep learning architectures and highly scalable architecture implementations.
3. The classical RL algorithms themselves are highly limited anyway, which is why you need to use deep learning architectures to even make a dent in anything more complex than grid-world.

So why even study reinforcement leanring á la Sutton and Barto? Why not just jump into lectures from Silver and skip the bits where temporal difference and Monte Carlo shows up?

The reason is that, like physics, reinforcement learning has a basic set of conceptual ideas that remain true *regardless* of the fancy deep learning architectures that you apply to the problem. The basic flaws that emerge in classical reinforcement learning both remain in deep reinforcement learning... and in fact are made *even more intractible* by having complex function approximators in the first place. 

In this post, I'm going to go through the classical reinforcement learning algorithms that I'll steadily be implementing over the coming weeks. I'll then describe some of the lessons that I've learned in actually implementing them from scratch.

## Markov Decision Processes
The basic environment that reinforcement learning algorithms try to solve is a *Markov decision process*. This is the theoretical foundation of any reinforcement learning environment - it's a highly flexible mathematical object that can represent most environments under the assumption that the future state depends only on the current state and the action taken rather than the history of previous states/actions.

Mathematically, you define a Markov decision process as a tuple of 
1. $$S$$ - the set of states that the MDP might assume,
2. $$A$$ - the set of moves that are available to an agent in a given state,
3. $$P(s' \vert s, a)$$ - the transition function that dictates the probability of assuming another state given the current state and a selected action. This function represents the dynamics of the problem. 
4. $$R(s' \vert s, a)$$ - the immediate reward immediately achieved after transitioning to a new state given a current state and an action.

Implementing a Markov decision process was an exercise in formalism for me. I spent a few days messing about with a highly explicit dictionary representation that was meant to enforce usability. I quickly realised however that *this is not a good way of going about these kinds of things*. The most simple (and usable) something can be is when it is represented in mathematical formalism. Departing from formalism is to add complexity, so to "engineer" simplicity is in fact a paradox.

I stripped back my implementation to represent $$P$$ and $$R$$ entirely in terms of tensors, each constructed from a set of basic parameters (the number of non-terminal states, the number of terminal states and the number of actions). I then built these objects into a class that closely followed the `gym` format. The results can be seen [here](https://github.com/aravinthen/deep_rl_experiments/blob/main/utils/markov_decision_process.py).

In retrospect, I was very lazy with my Markov Process definition. I went for a very general, abstract representaton that - despite being a useful test ground, is actually absolutely terrible for reinforcement learning in particular. I'll defer the reason as to why it was so bad to later sections. We resume as usual. 

## Dynamic programming
The basic approach to solving optimal control problems is the use of dynamic programming, which is a framework for solving the Bellman equation. The dynamic programming approach to RL consists of mapping out every possible trajectory and then iteratively improving the value estimates per state. Bootstrapping is the core of dynamic programming, where the value estimates of future states are used to update the value estimates of current states. 

There are two basic approaches to dynamical programming: value iteration, which directly solves for the optimal Bellman equation and policy iteration, which evaluates a specific policy and then improves it greedily. The key aspect of the MDP that is being directly updated here is the transition function.

### Value iteration
Value iteration is an approach to directly solving for the optimal value function. The update rule for value iteration is as follows:
$$ V_{k+1}(s) = \max_{a}\sum_{s'} P(s, a, s')\left[ R(s, a, s') + \gamma V_k (s')\right] $$
The method repeatedly updates the value of a state by taking the maximum expected value over all available actions, which means that it doesn't actually require the use of a policy. Instead, once the value function has converged appropriately, we can generate a policy by being greedy with respect to the provided value function. The problem depends entirely on having an accurate model $$P$$, but this is rarely if ever available for even idealised reinforcement learning environments. 

Implementing value iteration was probably my first lightbulb moment in how `numpy` and `torch` ought to be used. The value function update as provided in Sutton and Barto loops over states and actions, but I realised that the full calculation can be vectorized in a single line using `np.sum` and `np.max`. Some array tricks were required: the value function itself is one-dimensional, so adding the 3D reward tensor required the padding of the value function with `None` indices.

### Policy iteration
Policy iteration starts with a policy, evaluates the value function according to it and then improves the policy until the policy stabilizes between iterations.
The algorithm is composed of two stages:
    * The first is policy evaluation, where the value function associated to a particular policy is calculated. This is carried out in a similar way to value iteration, only with the added caveat that we no longer need to sum over all actions: the actions are supplied by the policy itself. This step did require a bit of thought on how to broadcast this effectively. I eventually settled on calculating $$P(s,s')$$ and $$R(s, s')$$ explicitly.
    * The second is policy improvement, which is just calculating the actions that are greedy with respect to a given state.

Both algorithms can be found [here](https://github.com/aravinthen/deep_rl_experiments/blob/main/algorithms/dp.py). The experiments I've run (which are admittedly very barebones) can be found [here](https://github.com/aravinthen/deep_rl_experiments/blob/main/experiments/dynamic_programming.py). As hoped, both algorithms provide almost exactly the same result when run for the same Markov decision process.

## Monte Carlo simulation
Monte Carlo simulation is one of my favorite methods. It is both so simple that it can be taught to a reasonably bright child whilst being so powerful that it forms the workhorse of essentially *any* field that deals with complex phenomena.
On the other hand, there's something of a complexity here. One might as why anybody even want to use the Monte Carlo method in the first place given how easy value estimation and policy iteration are. The answer to this lies in the fact that normally, the transition dynamics $$P(s',r \vert a, s)$$ aren't known accurately. There's a really important distinction here: when the transition dynamics are not known, one is forced to operate under a **model-free** assumption. Model-free reinforcement learning algorithms are probably the most important class of RL techniques.

The rough idea of Monte Carlo simulation is to just... *try it out*. The entire process can be broken down to 
1. **Generation** - the agent interacts with the environment to create an episode.
2. **Evaluation** - the trajectory sampled from the previous step processed to update the value estimate for each state.
3. **Repetition**: The same process of updating the value estimate is carred out for may episodes until the value function converges to a suitable level.

In **first-visit** MC, the value of a state is estimates by averaging the returns only from the first time that the state is visited in each episode. The standard logic is to generate a full trajectory and then iterate backwards over the state to calculate the cumulative return. 

In my experiments, I managed to get this to work for my very basic MDP. However, I quickly noticed that, despite having the same seed as for value iteration and policy iteration, the environment returned a different value. Weird, right?
In retrospect, it *was* weird. I convinced myself however that this was fine because MC is only supposed to converge at the limit of $$N_{\text{eps} \rightarrow \infty$$. I presumed that, given my very crappy hardware and the fact that it was taking me a good few minutes to run even 50,000 steps of the MC process, it would take me too long to get my algorithm working. Besides, on average the results seemed to converge to something anyway. I decided to move on, happy with the fact that I was reaching some level of convergence.

## Temporal difference methods
A temporal difference method is a technique that updates value estimates based on other, future estimates rather than waiting for the final reward. This is the essence of reinforcement learning as a discipline: the careful management of uncertainty whilst building a running estimate of the value function. In a sense, a temporal difference method is something of a hybrid between the dynamic programming approach of bootstrapping and the sampling approach of Monte Carlo simulation. Like MC, this technique is employed in model-free circumstances.

In dynamic programming, you can bootstrap very effectively because you have a perfect knowledge of the environment. Temporal differences on the other hand need to update based on comparison between current estimates and the results of future state estimates. The essence of this can be described in terms of the temporal difference error,
$$
\delta = r + \gamma V(s') - V(s)
$$, 
which measures the difference between the expected value and actual value of the rewards received. There are subtle variations that can be introduced at the temporal difference, each of which yields a different RL algorithm.

I implemented three basic temporal differene algorithms: 'SARSA', Q-learning and Double Q-learning. However, I also wanted to test the provided reasons as to why these algorithms worked in the first place. To this, I built a testing framework modelled closely on the ten-armed bandit testbed discussed in Chapter 2 of Sutton and Barto.

### The algorithms
`SARSA` is an on-policy algorithm, which means it updates its estimates based on the actual actions taken by the policy. It has the following update rule:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right] 
$$

This is in essence the same as the temporal difference error, although in this circumstance the TDE is weighted: the parameter $$\alpha$$ is the learning rate of the update, which determines how much new information overrides the old information. 

`SARSA` is supposed to be more cautious and stable, avoiding risky paths where exploratory mistakes can lead to larger penalties. It's supposed to be used in situations where *safety* is the priority, where exploration is inherently risky.

Q-learning is the significantly more aggressive cousin of `SARSA`. It is also off-policy, which means that the policy itself isn't used to update the Q-values. All of this can be seen in the update rule:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'}Q(s', a') - Q(s, a) \right] 
$$

The action that is actually used to update the action-value estimate is *not* generated by the policy itself but the action that yields the largest estimate of Q on the next step. 

The aim of this update is to directly aim for the most optimum path. Safety be damned - Q-learning will get your your results quickly and it'll fall into a few holes on the way. This approach is dangerous when failure causes enormous issues, but in general it's a somewhat better algorithm than `SARSA` in that it attempts to obtain the absolute best path to the goal. As can be imagined, it's supposed to converge to the optimal solution faster.

There's an issue with being aggressive. Aggression is naturally myopic - it can be fooled in circumstances where the environment itself isn't reliable. One symptom of this confidence is *overestimation bias*, which is where an agent that consistently overestimates the reward of a state bleeds that overestimation into the rest of its value estimates. Now, what happens when you *consistently* go for the maximum value of your next state... oh wait, not the maximum, but *your estimate* of the maximum? You're going to be overoptimistic. This is why Q-learning in particular is really bad for overestimation bias. 

Double-Q learning is something of an upgrade to Q-learning that operates by maintaining two independent estimates of the Q-function, $$Q_A$$ and $$Q_B$$. The update rule looks like this:

$$
Q_A(s, a) \leftarrow Q_A(s,a) + \alpha \left[ r + \gamma Q_B(s', \text{argmax}_{a'} Q_A(s', a')) - Q_A(s, a)\right],
$$

where A is interchanged with B with equal probability. The reasoning here is that the same Q-function is used to both evaluate the state and select the state: by decoupling this by feeding results through to two independent states, it is unlikely that both will overestimate the same action state simultaneously. For a better analogy, have a look at [my literature review](https://aravinthen.github.io/2026/03/07/literature_review/) on double deep-Q learning.

The goal of double-Q learning is to be *robust*. This means that, in a noisy system, it should definitely be more stable and accurate than the other methods. Granted, you'd expect it to converge more slowly than the others as two separate Q-value estimates are being updated throughout.

### The test harness
My test harness design is simple, but I went out of my way to make it as flexible as possible: I wanted be able to exchange the name of the algorithm as an argument to run the test automatically. 

The entire thing, of course, was dependent on all of my temporal difference algorithm classes having the very same structure: a `step` function, an `update` function and a standard means of obtaining the policy from the value functions.

The harness has three components:
1. **Policy evalulation**: Here, given a *trained* algorithm, the policy is evaluated. A set number of episodes are run, as well as a given number of maximum evaluation steps. The total reward for the episode is obtained and averages as the mean performance of that algorithm.
2. **Single experiment**: This function uses policy evaluation to systematically assess the performance of a single algorithm on a seeded MDP. The algorithm is updated for a series of testing steps and at fixed intervals it is evaluated using the previously described policy evaluation method. The experiment resets the MDP when the algorithm update lands on a terminal state. I made a consistently silly mistake here where I was using the MDP of the experiment to evaluate the policy. This led to jumpy and broken tests - the solution here was just ot initialise the MDP from scratch using the `__class__` attribute. The results of this stage are a trajectory of total rewards over time.
3. **Seeded experiments**: Finally, experiments were carried out over a predefined set of seeds. The trajectories for all of these seeds were averaged and a single trajectory of steps, mean total rewards and of course, the standard deviation of those results. The latter is especially important, as RL results are typically *very* noisy.

And so, it was time to test. Everything seemed to work alright in single runs, but remember: this is reinforcement learning. It doesn't work unless it works on average. 

### Results 
With great anticipation I ran my comparison script and got

![this](/_images/what_the_hell.png)



## Conclusion
