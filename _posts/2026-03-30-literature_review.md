---
title: Literature review 12 - Experience as a resource
category: literature
math: True
---
# Literature review - Week 12
This week's literature review is dedicated to the use of the replay buffer. I've always thought that replay buffers were both so simple as to be obvious, but also complex to make use of and motivate. I feel as though most people would intuit how experiences could be used to make reinforcement learning more efficient whilst simultaneously missing the statistical underpinnings of why they're especially necessary in deep reinforcement learning.

The experience replay buffer is essential in the formulation of deep-Q networks - this was actually where I personally first came across them. Even without worrying about the efficiency of training, the replay buffer solves the practical issue of *how* you feed an experience into a neural network. Normally, you could just chuck in a ton of labelled data with the assumption that the data is independent and identically distributed. You *cannot* generate such data from a single trajectory, because useful trajectory data always depends on the events that followed it. 
Take, then, the humble replay buffer. It's a data structure that just takes in the trajectory and *samples each experience* with a fixed probability. This immediately breaks the issue of correlation and renders exceptionally complex neural networks as useful tools of functional approximation in reinforcement learning.

This literature review goes into some depth on how this technique can be enhanced by both theoretical and hardware based enhancements. Let's proceed!

## Prioritized experience replay - Schaul et al.
The basic effect of the experienced replay buffer is that experiences are uncorrelated with each other, as each experience has a fixed probability of being used to train the function approximator. However, there's something of an issue with the assumption: it implies that every experience by itself is equally useful to the training process. 
This just isn't true. I'm on a train right now, and if I were to design an autopilot for the train I'd first separate the regions of the track that require most control. Long, straight and safe tracks *probably* don't require much attention, whereas nasty twisty bits will require quite a significant amount. 
If I sampled each point of this journey with equal probability, it's more likely that the straight tracks where I don't really need to do much are overrepresented in the training data. The complex control that I need to laern exists within the bends. So what do I do?

The solution is to weight the bends, or indeed any of the regions of the track that require most attention. And the tracks the require the most attention are those that typically cause the most temporal difference error in the early stages of the simulation. As such, it makes *perfect* sense that you should weight the experiences that are sampled with the temporal difference error that they generate. 

This is *very* similar to the basic ideas used in computational statistical physics. Indeed, the prioritized experience replay technique includes a $$\beta$$ parameter that controls the level of bias that is involved in the sampling: it plays the role of temperature and supports the exploration/exploitation process. 

One important note: using prioritized experience replay adds unacceptable levels of bias into the simulation, and employing a low $$\beta$$ parameter is not going to train your systems effectively. This is typically why $$\beta$$ is *annealed* so that it starts with high bias to locate the most important states and then reduces this parameter to allow exploitation.

## Distributed prioritized experience replay - Horgan et al.
I'll be honest. There's really not much I can go into here in terms of the theory. The distributed prioritized experience replay buffer is simply a parallelized version of the original idea, where the replay buffer exists as a separate pool of experience that is added to by actors and sampled from by a learner. Of course, this technique can only really be used by off-policy learning algorithms, as the buffer itself is fed with information from potentially many sources. 

Something interesting here is that actors and learners *are* separated in this architecture. The learners are located on GPUs, whilst the actors operate within the CPUs as they interact with environments. This is a very, very straightforward and logical way to conduct this kind of learning and I can see in other literature how this basic architecture of actors and learners has become an important way of looking at the problem.  

## Hindsight experience replay - Andrychowicz et al.
HER is *genius* and perhaps the first experience replay technique that stunned me when I first got it. It also requires something of a leap in terms of thinking about RL agents. 

HER is used in situations with sparse rewards, or when the positive reward signal is rare. An example of this given in the paper is scoring a goal in hockey, but there is no dearth of examples: reward, as a whole, is sparse for *most* activities. 

The psychological insight that led to the invention of her is the idea that when you fail at something, there's *still* useful behaviour that is to be gleaned from your failure. In the hockey example from before, if you shoot a puck and miss the goal you've at the very least learned something about what happens when you shoot the puck in the way you did. You learn the connection between your actions and your outcomes, even if it was a failure. 
Normal reinforcement learning discards most of this experience as low signal and won't train the function approximator on this experience. Here is where HER kicks in.

Let me share this analogy that I came up with when I was reading the paper for the first time:
Let's pretend that I'm a RL agent.
1. I try and get to a particular point in a valley.
2. I amble around and finally get somewhere that is not that point.
3. I think to myself "Well, if this *was* where I wanted to end up then the actions that I took to get here would have been perfect".
4. I make the connection between my actions and "getting to places".
5. I'm soon able to get to the place that I want to get to.

The core of the technqiue actually depends on the architecture being used to represent the model, which in the paper is given the name "universal value function approximator". The universality here comes from the fact that this value function takes in a goal as a *variable*, which in turn implies that the goal is something that can be recognized and encoded as a predicate. 
Now, once you've generalized the value function like this, you now have the ability to change your goals on the fly.

HER essentially works by changing the goals with regards to this universal value function approximator. It
* generates a trajectory, 
* rejigs the trajectory to make it so that the terminal state of the trajectory was the goal, 
* converts every step in that trajectory to be a reward signal, which in turn converts the sparse rewards trajectory into dense rewards, 
* allows the agent to taste success, with the goal of generalizing that success to more complex examples. 

It is a magnificent technique, albeit one that is very *very* unintuitive.  

## Dueling Network Architectures for Deep Reinforcement Learning - Wang et al.
This is a network architecture paper - the first that I've read so far in the context of reinforcement learning. The impetus behind this paper is the observation that there is a functional difference between the value function and the advantage function. 
The value function seeks to give a numerical score to each state. It averages over actions, which in turn carries out a full assessment of the state as a whole. 
The advantage function (which is the difference between the state-action value function and the state value function) is move dependent. The paper itself opens by acknowledging that there are many situations where having a state-action value estimate is really not that useful  - one example being driving on a motorway (or, alternatively, the train example I gave earlier). In some states, like being in the middle of a straight train track, actions don't really *do* anything. 

The solution that the authors introduce is to explicitly separate value approximations and advantage approximations. The *real* magic, however, is how they put these two approximations back together.
First, why is this even so difficult? Isn't advantage just the difference between between $$Q$$ and $$V$$? Yes, but there's an issue here called the problem of identifiability. If you add a constant to $$V$$, you can simultaneously subtract a constant to $$A$$ to recover the same $$Q$$, which naturally causes instability in the training process if you carry out the calculation in the naive way.
The trick is to subtract the mean advantage from the raw advantage estimate. The advantages themselves are forced to sum to zero and you suddenly have the uniqueness necessary to carry out stable training.

It took me *ages* to figure out what is even going on here. Surely if you subtract the mean advantage away from $$Q$$-function estimate you're *directly* changing the value of $$Q$$? That's technically not the $$Q$$-function anymore! And that's correct. However, it happens that we don't really even care about the actual values of the $$Q$$ function when using it in RL. We really only care about the relative size per action. By subtracting the average, we're *still* allowing for this relative size to be unique! In a sense, this is a *hack*. A clever idea that preserves the basic requirement of a $$Q$$-function but does so in a mathematically flexible way.

This was an illuminating paper. It changed my entire perspective on what the $$Q$$-value is versus what is should be used for. The general concept of a duelling network also seems to have become commonplace in RL applications after this paper was released. I *still* don't really get why it's called a duelling network, though.

## Recurrent Experience Replay in Distributed Reinforcement Learning - Kapturowski et al.
In this paper, memory is given memory. On a conceptual basis the concept of recurrent experience replay isn't particularly challenging: all we're seeing here is the generalization of experience replay to sequences over single transition tuples. 
Let's go through this from first principles.
1. Replace your value network `MLP` with a `RNN`, potentially with some kind of long-short term memory. 
2. You now would be wasting your time if you sent completely disparate experiences through your network. The point of using an RNN is to predict in terms of sequence to allow backprop over time, so the basic idea of a replay buffer goes out of the window. What do you do here?
3. You take *sequences* of overlapping experiences and then assign priority by either taking the maximum or the mean of the temporal difference within the sequence. 

This is the core of the technique. However, there are a few issues that have to be sorted out here for maximum benefit. For one, you still have the issue of policy lag when handling the distributed learning problem. To counter it, you can 
* Use a storage tactic where you store the hidden state as part of the replay and use it to initialize the network at training, 
* Apply a burn-in tactic where you use part of the sequence in order to reinitialise the network, after which you use the remaining part of the sequence to update the network. 

As it happens, recurrent experience replay is used a lot in modern (post 2020) reinforcement learning. This is a standard when dealing with **partially-ordered Markov decision processes**, where you *need* some degree of memory in order to efficiently solve the learning problem. 