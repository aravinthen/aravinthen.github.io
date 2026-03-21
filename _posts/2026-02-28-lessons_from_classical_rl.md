---
title: Lessons from Classical RL
category: technical
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
3. $$P(s'| s, a)$$ - the transition function that dictates the probability of assuming another state given the current state and a selected action. This function represents the dynamics of the problem. 
4. $$R(s'| s, a)$$ - the immediate reward immediately achieved after transitioning to a new state given a current state and an action.

Implementing a Markov decision process was an exercise in formalism for me. I spent a few days messing about with a highly explicit dictionary representation that was meant to enforce usability. I quickly realised however that *this is not a good way of going about these kinds of things*. The most simple (and usable) something can be is when it is represented in mathematical formalism. Departing from formalism is to add complexity, so to "engineer" simplicity is in fact a paradox.

I stripped back my implementation to represent $$P$$ and $$R$$ entirely in terms of tensors, each constructed from a set of basic parameters (the number of non-terminal states, the number of terminal states and the number of actions). I then built these objects into a class that closely followed the `gym` format. The results can be seen [here](https://github.com/aravinthen/deep_rl_experiments/blob/main/utils/markov_decision_process.py).
