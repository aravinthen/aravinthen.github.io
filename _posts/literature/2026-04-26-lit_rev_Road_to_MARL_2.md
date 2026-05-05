---
title: 2. Multi-agent Reinforcement Learning - Learning Dynamics
category: literature
math: True
---
# Multi-agent Reinforcement Learning 1 - Learning Dynamics
## Preamble
Last week, I spent much of my time thinking about the concept of equilibrium as an emergent consequence of rational players and well quantified consequences. I also spent a *lot* of time realising that the problem of multi-agent reinforcement learning is *anything but* rational and well quantified. It was very obvious that the basic premise of game theory - the emergence of a Nash equilibrium - doesn't really apply so much in situations where players are adapting whilst playing.

So... what does? We *are* seeing competitive scenarios unfold and adaption taking place. Does game theory fall apart if we add this very reasonable consideration? Isn't adaption how *everybody* learns? It is, and naturally this has been noticed by game theorists over the last century of development. The body of theory associated with this set of circumstances explodes from just game theory into optimisation and forecasting - the end result is the emergence of a *new* kind of equilibrium that is the end goal of multi-agent reinforcement learning.

I'm glad I spent a fortnight looking into the theoretical basis of MARL. There are *so many* extraordinarily nuanced concepts that seem so simple at first but hide real complexity and design choices. Take, for instance, the idea of self-play. The way it's often described is something like  
* You have an agent,
* It plays against itself, (**Note**: I know this isn't necessary true, but this is seemingly the generic consensus for how people think self-play works).
* It learns to beat itself repeatedly and becomes really strong over time.

That makes intuitive sense, right? But why on Earth would this work in the context of AI agents? These aren't humans that we're talking about, these are function approximations. In depth, the questions here are
* How do we know if self-play even works? Why should we trust it? Anybody who plays a game of chess will have played against themselves. By this logic, can someone get really good at chess by simply playing ags 
* How do we know where this process of self-play will take us? What does a solution even look like here?

Simply put, we need another layer of technical machinery in order to answer these questions - both about self-play and general MARL. 
  
## Equilibrium whilst learning
As a recap, Nash equilibrium is based around achieving a stable state of play where opponents have no reason to unilaterally change their own strategies, as any deviation would put them in a worse position. Naturally, the Nash equilibrium that is attained within a game isn't necessarily one that is best for all agents: it is merely a fixed point that emerges as a consequence of the agent actions as a whole. 

The idea of Nash equilibrium (and the probabilistic variant of mixed equilibrium) falls apart completely when considering agents that adapt. This is because
1. you attain Nash equilibrium by assuming that your agents are perfectly rational and thoroughly informed to the point where they can evaluate the effect of their actions immediately. The former falls apart because whilst RL agents can be "optimal" in that they adapt to their environments, they can't be said to be rational: they aren't reasoning, they're maximising. As anybody who has seen an RL agent carry out reward hacking might tell you, maximising a reward blindly is the opposite of rationality. 
2. You don't initially start off with a perfect understanding of what your actions entail: you must *learn* this. It may take a very long time before you can truly consider your value function to accurately represent the payoff of a given strategy. 

So, with this in mind, how can the goal of multi-agent reinforcement learning - be it adversarial or cooperative - be realised? How can agents learn good strategies through interaction withouth solving the game explicity?

Let's revisit what "learning good strategies through interaction" means. The multi-agent reinforcement learning exercise can be framed in game theory as a *repeated game*, where the same base game is played over and over so that agents can engage in long-term strategic interactions. It must be noted that this framework violates a *lot* of the assumptions employed in single agent reinforcement learning, specifically with the assumption that the environment is stationary: when your opponents are learning and adapting with respect to a repeated game, then the strategy you just used suddenly becomes out of date.

Most of this review is based around this question: how can an agent learn appropriately when they're in an adaptive environment? The answer to this lies in *online learning*, which by itself is a completely different approach to the problem of **forecasting** based on sequences.

## Regret as an objective
My first source in getting my head around the topic of regret minimisation was a textbook by Cesa-Bianchi and Lugosi [1]. This book introduced the *forecaster with expert advice* problem, which is roughly speaking a game where a learner merges the predictions from multiple experts in order to minimize cumulative loss compared to the best of the experts over history. The actors here are
* The sequence, or the stream that is supposed to be predicted,
* The experts, which are black box algorithms offering predictions
* The learner, which is responsible for aggregating multiple predictions and
* A loss function, which quantifies the accuracy of a guess in hindsight.
 
The goal of this problem isn't to simply minimize loss: that is impossible, as there is still a learning process that must be undertaken. The goal is to *minimize regret*, but in a way that is **independent of the stochastic process underlying the sequence**. This last bit is really important: it even encompasses adversarial behaviour that adapts to the predictions that you make. Another way of framing regret is as difference between the total loss of the best expert as well as the total loss of the learner,

$$
\text{Regret } = \max_{\text{expert}} (L_{\text{learner}} - L_{\text{expert}})
$$

Back to the target of the problem: we're trying to minimize regret, which is essentially the cumulative loss over multiple steps. This is a measure of the performance versus the best fixed action in hindsight.
There is an important distinction to be made here that I realised a little too late.
* A prediction process with zero *loss* implies that you've always made the best choice. This is of course impossible unless you've got access to a machine that immediately spits out the correct answer (an oracle, which will show up later).
* A prediction process with zero-regret, on the other hand, means that your loss converges to the loss of the best expert in hindsight as $$T \rightarrow \infty$. This still allows your experts to be terrible whilst allowing you to have zero regret: you're *significantly* weakening your goal by aiming for zero regret, as all you have to do is be equal to or better than your best expert. 

Regret is a much, much more suitable target than loss. We can emphasize this by considering that the regret itself measures against *hindsight*, whilst loss only makes sense in the context of a localized interaction. Likewise, the most powerful aspect to regret is that it completely avoids the stationarity assumption that is pretty much inherent to single-agent reinforcement learning. In fact, you don't even have to factor in opponent behaviour - *there are literally no assumptions about how the data is generated*. 

The goal for predicting over trajectories thus becomes one of **no-regret learning**.

## No-regret learning
The goal of no-regret learning [2] is to develop a guaranteed performance in the face of uncertainty in an environment, or more precisely a *retrospective guarantee*. You have to abandon the traditional idea of statistical learning here (more specifically the idea of i.i.d, which assumes that all data is uncorrelated and by extension forces the future to resemble the past). All you can do in such circumstances is to do your best to learn in hindsight.

As it happens, this is an example of *online convex optimization* [3]. In fact, it can be demonstrated that no-regret learning is actually the online equivalent of classical convex optimization and, from my quite brief read and vague understanding of the work of Hazan, a purely geometric problem. The most important part of this connection is that is gives a really strong intuition into what is actually going on in any algorithm that tries to minimize regret: the prediction of a learner with respect to a regret vector over the learning steps is a projection into a convex decision set. Very interesting, but definitely beyond the details of this blog!

## Regret matching and an emergent equilibrium
The foundational strategy for carrying out no-regret learning is called **regret-matching** [4]. The essence of this approach is to simply update your strategy based on the regret of not having chosen that action previously.
The rough process is as follows:
* Calculate the regret for every action not taken during a defined game history.
* Accumulate this regret over time.
* Use the regret for each action to sample over a probability distribution of actions, where the probabilities themselves are informed by the cumulative regret associated with that action.
* Keep playing until the average regret for not playing an action reduces to zero.

When I was studying this, I came up with the following illustration:
* You have two players, A and B. They play a game and A loses after playing a move that isn’t M (which would have won the game).
* A regrets not playing M, so A is now highly likely to play it.
* B, who played N, has no regrets. She’ll continue to play N regardless of the future.
* The next round, A plays M and B plays N. A naturally wins.
* Following this, B has regret whilst A has none. B will shift her move to one that wins, whilst A will not.
* Using the full history of actions, A and B can come up with a means of picking their move that will cause their total regret to shrink to zero over time. 

What is the nature of the strategy that the agents employing regret-matching settle on? It *could* be a Nash equilibrium, especially if the game is simple. However, the real equilibrium that is likely to emerge is typically a little weaker. In terms of ranking,

* Nash equilibria: a state where no player can gain by changing their strategies alongside their opponents.
* Correlated equilbria: a weakened form of Nash equilibrium where a mediator is present to recommend strategies. The mediator is trusted, so the players have no reason to distrust their recommendation.
* Coarse correlated equilibria: a further weakening of correlated equilibrium where the players are required to choose whether they want to follow the mediator’s advice or not.

In a no-regret scenario, we typically settle upon the last of these: a weak form of equilibrium where actions are suggested by a mediator signal: in this case, *the learning history* that all agents typically have access to. As such, when people attempt to obtain solutions to games using a self-play strategy, the best we can really hope for is a coarse correlated equilibrium... which can still be very, very powerful. 

## Onto multi-agent reinforcement learning
With the idea behind regret-matching in mind and noting its similarities to self-play, it becomes quite easy to see that well-designed self-play (just like regret matching) is merely a manifestation of no-regret dynamics. We should note here that by well-designed I mean stable and by stable I mean that the resulting strategy will tend to some sort of game-theoretic equilibrium. In a single word, this explains why self-play even works: as an example of no-regret dynamics, self-play actively projects the unbounded strategy space of an agent into a predictable bounded target. The projection here operates exclusively on regret: by *minimizing* regret, agents explore broad and complex strategy profiles from with cooperation and adversarial play emerge.

As it happens, *all* working multi-agent reinforcement learning algorithms will typically be some flavor of a no-regret dynamics generation protocol. This study answers a question that I've had since I started working on reinforcement learning: *what actually is stability*? Without regarding the additional complexity of stable training, a truly stable multi-agent reinforcement algorithm is one that *gradually* shifts towards game-theoretic equilibrium. 

What also this means from a practical perspective is that you can pretty much yank a lot of the work that has been done on no-regret learning (like the dense optimization of Hazan) to generate new multi-agent reinforcement learning algorithms. As we'll be seeing in the next few posts, this is very much the pattern that multi-agent reinforcement learning algorithms actually take. 

## References
[1] Prediction, Learning, and Games - Cesa-Bianchi & Lugosi

[2] [CS364A: Algorithmic Game Theory Lecture #17: No-Regret Dynamics](https://theory.stanford.edu/~tim/f13/l/l17.pdf) - Roughgarden

[3] Introduction to Online Convex Optimization - Hazan

[4] [A simple adaptive procedure leading to correlated equilibrium](https://www.jstor.org/stable/2999445) - Hart & Mas-Colell (2000)

