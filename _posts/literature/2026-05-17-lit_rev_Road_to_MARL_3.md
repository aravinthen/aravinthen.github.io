---
title: Multi-agent Reinforcement Learning 3 - Algorithms for Markov Games
category: literature
math: True
---
# Multi-agent Reinforcement Learning 3 - Algorithms for Markov Games
## Preamble
In the previous note, I approached MARL through the frame of learning dynamics. I've thinking a lot about what this really means in context and after about a month of digesting the information here, I've come to the realisation that the basic problem is that in MARL, the environment faced by each learner contains other adaptive learners, so the effective dynamics are non-stationary and strategically coupled. In a few points, the core of last month's review can by summarised as the following line of reasoning:
1. Agents adapt while playing repeated games.
2. The stationarity assumption that drives most of the results in single-agent RL falls apart. 
3. A new requirement emerges: stable policy improvement under adaptation. 
4. Self-play/no-regret provides a view of how these new requirements can be met from the lens of learning dynamics. 
5. The hoped-for pattern isn't so much a single "optimal policy", because as justified in Part 1 those don't really exist in games - we're looking for a *stable interaction pattern*, also known as *equilibrium*.

We've defined the terms and motivated a solution. Another way of phrasing the goal equilibrium is as the discovery of a state of joint behavior that might arise after many rounds of mutual adaptation.

Now, equilibrium would be a nice goal. It is often-times *not* achieved for a variety of reasons - these will be discussed in their own section. Before we proceed, let's examine the theoretical setting of multi-agent reinforcement learning as a distinct object from reinforcement learning.

## Markov Games
A **Markov** game is a generalisation of the single-agent Markov decision process. There's nothing magical here and the extension is quite straightforward:


| Single Agent RL       | Multi-agent RL                                       |
| ---                   | ---                                                  |
| States                | Unchanged                                            | 
| Actions               | A set of actions available to each agent             | 
| Transition function   | A transition function that accepts the joint action  | 
| Reward function       | A reward function for each agent                     |  
| Discount              | Unchanged                                            | 

Of course, observations are also a part of the formulation (although not essential). Whilst the formalism is complex enough, the crux of why Markov games are difficult lies almost exclusively in the *joint action*. It is this sudden leap in complexity that yields so much complexity, as it transforms the problem from finding an optimal policy in a fixed environment into the multi-agent problem of learning stable behaviour under joint action, strategic coupling, and possibly adaptive opponents/teammates.

Another more subtle complication arises from the simple fact that reward functions have to be assigned for each agent. This brings forward the problem of credit assignmnent... but *on steroids*. Before, the credit assignment problem was based on figuring out which series of actions was responsible for an increased reward signal. 
Now, the problem becomes figuring out which series of subsets of the joint actions was responsible for the same. This is, as one might expect, very hard. 

## Non-stationarity in practice 
We've been talking a lot about non-stationarity. However, this term - whilst standard and easily interpretable - encompasses a *lot* of failure modes and complications. What are they? What do those actually look like? 

From looking at the literature so far, the central difficulties I've read about when considering the ramifications of non-stationarity alongside adaptive learning can be separated into three major categories:
1. Learning instability, which is mostly a consideration of the fact that you're trying to optimize against a moving target. 
2. Self-play and adaptation problems that 
3. Consequences of the game itself, where the equilibria that are attainable aren't actually conducive to stable and well defined policies in the first place. 

A lot of what I'll write here is based on *A Comprehensive Survey of Multi-Agent Reinforcement Learning* by Buşoniu, Babuška and De Schutter.

### Learning-instability problems
This is what constitutes most of the mismatch against single-agent RL and MARL. The issue here is that a lot of the basic assumptions that allow single-agent RL become suspect: examples include
* moving targets for value estimation. The learner is not merely estimating a difficult value function, it is estimating a value function whose meaning changes as the other agents update .
* stale experience in replay buffers: if your agents are learning and improving as they train, then the *second* you put an experience into a replay buffer it is at risk of quickly becoming stale and useless for training.
* policy cycling, which is where other agents will continuously adapt to each other. This is a nefarious mix, as there's nothing wrong in the *algorithms* here: they'll work as intended, but they'll never settle as they change *with respect to each other*.

### Self-play and adaptation 
If previously we were worried about the learning process jumping all over the place, this class of pathologies is one where we worry about the policies evolving to be too *narrow*. Another way I've seen this described is as *adaption to the population* versus adaption to the environment. Examples of these are
* arms races, which is a specific case of cycling: agents repeatedly discover exploits, counter exploits, and so on. 
* overfitting to current opponents, which is where an agent performs exceptionally within a population that it has co-evolved with... but falls apart if a novel agent pops up. I've seen this in person and it's the *worst*.
* catastrophic forgetting, or when a policy loses robustness against older strategies as it adapts to newer ones.
* another problem that I came across when studying cooperative MARL specifically was that of brittle conventions, which is what happens when the agents communicate using a highly specific language that doesn't translate well to other agents.

### Strategic issues
Finally, there's another risk that is less dependent on the policies obtained but *the game itself*. Some games just don't have satisfying solution concepts! These aren't so much pathologies: the algorithms could work perfectly fine and discover these unsatisfactory polices without extra computing. 
* the first of these is the presence of non-transitive dynamics, which emerges in games like rock-paper-scissors. In these games, all you can really do is be maximally random. Not a good policy. Granted, in very complex situations there can be multiple degrees of non-transitivity: there might be a subset of policies that are transitive, but in general dominant strategies exist. 
* multiple possible equilibria, which isn't actually a *pathology* but can make the exploration and training phase painful as well as an incomplete representation of the solution space.
* convergence to poor or exploitable equilibria, which *is* a pathology and is a consistent producer of stable but low-quality solutions.

## Centralized training with decentralized execution
What is the basic commonality in all of these issues? What is the missing ingredient in the training process that yields these problems?
It is simply this: the agents are trying to learn from a **partial, local view** of a system whose dynamics are based on actions that are not generated by just one agent. They are, simply put, operating with too little information if their value functions ignores other agents. 

However, we don't need to keep things this way. The process of generating a usable policy is based around training and evaluation, two distinct processes. Just like how policy gradient methods use a "helper" in the form of a critic during training, it is possible to augment the agent experience in a similar way to account for multi-agent scenarios. You're essentially conditioning the learning process on the actions of the other agents!

This is the core of many modern MARL algorithms: centralized training, where agents are provided additional context in order to form the necessary links. This is typically followed by decentralized execution, where the context is removed and the agents deploy with just their own policies. 

In detail: during training, the learner may access anything from 
* global state,
* joint observations,
* joint actions,
* other agents, policies,
* shared rewards,
* centralized value functions or critics,

However, during execution, each agent acts using only:
* its local observation,
* its own policy,
* *possibly* its own recurrent memory.

CTDE is the architectural bridge between the learning-dynamics view of MARL discussed in the previous review and practical deep multi-agent algorithms that we'll be discussing in the next section. It may be a little obscure what the relation between the two are - the former is a theoretical online optimization framework whereas the latter are typically very reasonable algorithmic choices. The core here is that CDTE *explicity encodes the fact that the other agents operating in the environment aren't noise and are part of the learning dynamics*. The practical element here is that CTDE achieves this whilst still yielding policies that can operate *outside* of the shared information training phase. 

A final note: I mentioned that many MARL algorithms are just examples of no-regret dynamics. What does CTDE have to do with no-regret?
The answer to this (and one that I have spent a lot of time thinking about) isn't so much that CTDE implements no-regret dynamics. Rather, it's a setting that allows no-regret dynamics to be encoded in the first place. In other words, to know whether an action was good, bad, exploitable, robust, or strategically sensible, the learner needs information about what other agents observed, what other agents did and the full state information and reward signal of the environment. This is what is necessary for no-regret dynamics to be computed and this is what CTDE provides. 
 
## MARL Algorithms
Here are some algorithms that nicely encapsulate this basic methodology of centralized training over decentralized execution. I was getting quite excited reading these papers, but it should be noted that these aren't *perfect* solutions to the problem of non-stationarity: they're just modifications to the training signal that allows for action profiles over single actions.

The main split in techniques is between 
* centralized-critic methods, which improve policy-gradient learning, and 
* value-factorization methods, which are purpose built for cooperative learning. 

Very briefly, here are some examples of MARL algorithms and the ideas that make them work. 

### Centralized Critics
This the most direct encoding of information sharing. All this technique really does is incorporate the full action profile into the critic so that agents are informed by the joint interaction during training. Note that information sharing here is fully isolated within the critic: when you imagine critics as simply providing an estimate of how good a state is, it's clear to see that the actors are still completely decentralized.

The first paper in which I encountered this idea was that of Lowe et. al. (2017) [2], but similar ideas kept cropping up in the other papers. It's safe to say that this is fundamental to the modern MARL approach.

### COMA: Counterfactual Credit Assignment in Cooperative MARL

COMA [3] uses the CTDE pattern for *credit assignment* under a shared team reward. The quandary here is this: if all agents receive the same reward, what does the reward signal say about the overall effect of an agent’s action? How can one know if an action helped or hurt the team? COMA addresses this using a centralized critic and a counterfactual advantage: you compare the action an agent actually took against the value it would have received under alternative actions, holding the other agent actions fixed and employing a critic to predict the *baseline* action.

### Value Factorization
Value-factorization approaches CTDE by considering the individual generation of value by different agents. In cooperative MARL, the basic training target is a centralized team value function - this is down to the fact that rewards are generated via action profiles rather than individual actions. However, decentralized execution requires each agent to choose actions using local information. Value factorization bridges this gap by learning a joint value function during training while constraining it to be expressible with respect to single agents. The basic technical solution here is to employ a particular form of the team value function so that it can be appropriately factorized (broken down into a per-agent basis). The idea of value factorization is powerful and naturally, there are multiple ways to convincingly carry it out.

VDN [4] makes the strongest and simplest assumption: the joint team value can be written as the sum of per-agent value functions. This provides a natural route into decentralized execution as improving an individual utility directly improves the total value. The limitation is that additivity is far too harsh and reductive: given that many coordination problems involve interactions where the value of one agent’s action depends strongly on what the others do, how can one even begin to justify simply mashing together value functions into a single sum?

On the other hand, QMIX [5] relaxes VDN’s additive assumption by using a *mixing* network to combine individual utilities into a centralized Q-function. Basically, the function approximators of the value functions are *explicitly* combined using a more complex architecture - this was cool to me, as it showed novel and interesting network architecture design that I haven't seen elsewhere. The crucial constraint is monotonicity: increasing an individual agent’s utility should not decrease the joint value: this is achieved by being *very* explicit about the weights being positive in the network. 

## References
[1] Buşoniu, L., Babuška, R., & De Schutter, B. (2008). A comprehensive survey of multiagent reinforcement learning. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), 38(2), 156-172

[2] R. Lowe, Y. I. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, "Multi-agent actor-critic for mixed cooperative-competitive environments," in Advances in Neural Information Processing Systems

[3] Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., & Whiteson, S. (2018). Counterfactual multi-agent policy gradients. Proceedings of the AAAI Conference on Artificial Intelligence

[4] Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., Lanctot, M., Sonnerat, N., Leibo, J. Z., Tuyls, K., & Graepel, T. (2018). Value-decomposition networks for cooperative multi-agent learning based on team reward. Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems

[5] Rashid, T., Samvelyan, M., De Witt, C. S., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. Proceedings of the 35th International Conference on Machine Learning