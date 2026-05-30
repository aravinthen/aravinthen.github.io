---
title: 3. Multi-agent Reinforcement Learning - Algorithms for Markov Games
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
5. *Equilibrium*, or a set of mutual policies that are unilaterally stable  as an emergent target. 

We've defined the terms and motivated a solution. The goal now is to figure out how the target of equilibrium is 


## Markov Games

## Non-stationarity in practice 

## CTDE: Centralized training with decentralized execution

## MARL Algorithms
### Centralized Critics: Making Other Agents Visible During Training
### COMA: Counterfactual Credit Assignment in Cooperative MARL
### Value Factorization: Structuring Cooperation Through the Value Function
### VDN and QMIX: From Additive to Monotonic Decomposition