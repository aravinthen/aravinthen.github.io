---
title: 13. Dreamer
category: literature
math: True
---
# Literature review - Week 13
I spent my literature review this week exploring the **Dreamer** series of RL agents. Dreamer was my first introduction to modern model-based reinforcement learning and, by extension, a new set of techniques that in my mind I think of as world-actor-critic models. The focus here is on the idea of *world models*, which are latent representations of the world that the agent operations in. 
Yann LeCun argues that world models are essential for achieving human-level AI. Honestly, after reading about Dreamer, I can't see how anybody could disagree with that. The world model itself, as a function, represents to agents what imagination is to humans. It's an easily queryable representation of reality, which - from work by Schmidburger and Ha that I covered a few months ago - is *pretty much* how humans experience reality in the first place. In fact, now that I think about it, nobody *really* learns in the way that model-free algorithms do. If I touch a stove whilst learning to fry a steak, I typically learn that *I should probably not touch a stove*. I learn this because the impression of being burned is now something I can imagine: I query my own mental apparatus before I carry out an action to see if I'll get burned by doing so. If my brain was a model-free reinforcement learning technique I'd probably just... keep touching the stove until my value function updates. The difference here is that with a model-based approach, I can imagine what happens. With a model-free approach, I just bloody well do it!

An important consideration from a reinforcement learning perspective: *world models aren't, technically speaking, part of reinforcement learning*. A world model is a tool that can be used to develop agents via reinforcement learning, but they're more appropriately an example of representation learning applied to reinforcement learning. After reading about forty papers in this field I'm starting to see the deeper architecture that surrounds the field of artificial intelligence as a whole (although not to the point where I can put it into words just yet). Roughly speaking however, I'm getting a sense that the problem of intelligence really boils down to perception and generalization, where
* perception is a fast and immediate process necessary to represent the complexity of sensation in a usable, interpretable form and 
* generalization (or perhaps reasoning) is the ability to combine the usable/intepretable representation to derive consistent and coherent truths. 

Going through the Dreamer papers was intensely technical and quite exhausting. A lot of the content here cannot passively be understood - I feel like building, experimenting, messing up and seeing the difference between a design choice of Dreamer VX and Dreamer VY is the only way to really get to the core of what is going on here.

Anyway, musing aside, let's get on with the literature review.

## Dream to Control: Learning Behaviors by Latent Imagination - Hafner et al.
The first of the Dreamer models. The basic structure of a Dreamer model is that it has a world model that it can query in order to carry out multi-purpose predictions. Now, the *really* cool thing about Dreamer is that it has the ability to imagine latent representations of trajectories and train the actor and critic based on these latent representations. There are two basic benefits to this approach:
* The latent representation of the environment state is probably significantly more information rich than just the observation and can make for better training,
* the agent now has the ability to assess trajectories without actually having to run them, 
* the world model, which is differentiable, can be used to flexibly update the actor based on imagined possibilities.

The world model is the core of this advance. It is effectively the function approximation of the transition function that defines a Markov process. This advance isn't actually very intuitive: if you have a means of querying the world simply by interacting with it, then *why would you need a model to mimic it*? Because you can query a world model a billion times before you carry out an action, whilst you can query the real world *by* carrying out an action. This ability to generate trajectories on a whim is very, very useful when training. 

An important aspect of this paper to note is that, as mentioned previously, the planning process takes place within the latent space represented by the world model. In a sense, it makes more sense to think about a world model as a translation layer:
1. Generate an observation from the world, 
2. Convert it into the latent space representation, 
3. Carry out all planning with the more compact latent-space representation, which overcomes the issue of generating and processing pixels, 
4. Convert into an action which in turn allows interfacing the environment. 

There are some finer points to discuss here, but before we go onto that it's worth noting that the world model is a pretty specific class of approximator: a recurrent state-space model. This is a very specific class of model that 
* has a deterministic state, which is represented by RNN and is something of a long-term memory buffer,
* has a stochastic state, which captures uncertainty. 
RNNs operate by splitting the latent state representation into a combination of the deterministic and stochastic state. The impression I got when briefly reading over this is that an RSS is better at handing uncertainty than an RNN and better at remembering than a purely stochastic model. 
Quick note, because I can't help myself. Isn't this *incredibly* cool? Simply by combining a model that handles deterministic memory-driven outputs and a model that generates with a degree of randomness, you can combine *memory* as well as the ability to represent different futures at once. Amazing!
It's worth mentioning that a great advantage of using the world model in this way (as well as the fact that the world model is differentiable) is that now, *analytical* gradients can be estimated and used in training. 

It should be noted that Dreamer trains three models simultaneously:
1. The world model, which is trained by minimising the reconstruction error (decoder specific) and the transition error, or the difference between the stochastic state and the actually observed state,
2. The actor, which minimises an error that is *somewhat* similar to the generalized advantage estimate (but uses imagined trajectories instead),
3. The value function, which regresses on the reward return.

All in all, the main result here is that of efficiency: twenty times less data was employed whilst reaching the same level of performance as other state-of-the-art algorithms.

## Mastering Atari with Discrete World Models - Hafner et al.
Following the first paper is Dreamer V2. One of the nice things about covering subsequent developments in a basic technique is that it makes writing a literature review somewhat easier. :) 

The major changes from Dreamer V1 are the following:
1. **The use of discrete latent variables over the continuous variables.** I must admit that on first reading I didn't really understand why this was such an important advancement: in my personal notes (which can be read on my 03/04/2026 log), I don't even mention it. However, this is actually a very important addition... *with a caveat*.
    * In the previous model, the latent space was represented by Gaussian variables. This makes sense in terms of bias: you're assuming that the world is smooth, which is really good for continuous control. However, in video games this isn't so useful a representation, mostly because there are many circumstances where the transition is *naturally sharp* (for example, if you're playing Doom and you shoot a bullet). Dreamer V2 uses discrete, categorical latent variables, which work better for these sharp transitions. 
2. Given that you can't use the previous trick for calculating continuous gradients anymore, you need to come up with a different way to train the model. The technique described in the paper for training the world model is something of a *hack*: you pretty much just pass through the gradient as though it's continuous anyway. 
3. Another technique (and one that I'm definitely the fuzziest on) is the use of KL balancing.   The impression I'm getting is that this technique is used to handle the fact that the world model prior is less accurate in the beginning, which in turn causes issues when you're trying to train the posterior towards it. The prior and the posterior are thus weighted differently: by setting the weighting parameter to, say, 0.9, the KL divergence can be used to drive the prior to a closer fit to the posterior. 
    * I will be 100% honest and say that I'm still a little fuzzy on the details of what is going on here. I think the only way I'll be able to understand this segment is by implementing the model myself.  

Something that I found a little unusual when reading this paper was that the functionality of predicting into the future using the world model (that impressed me so much from the first version) wasn't really given much emphasis in this paper. It's still there, but it didn't look like anything really *changed* with regards to this core technique. 

## Mastering Diverse Domains through World Models by Hafner et al. 
The main contribution of Dreamer V3 isn't so much brute performance but the ability to perform well in terms of *diverse* environments. In fact, the core of the paper is framed as a means of defeating the trap of hyperparameter tuning: Dreamer V3 is meant to operate with a single set of parameters. 

In light of this, the most important advancement that I could detect here was the use of the **symlog transformation**. 
* The essence of this technique is to normalize agent input in a way that large values are squashes and small values are preserved. At first, I though that this was only really relevant to the reward structure: this is wrong, as on second reading I realised that symlog is applied to observations and even the entire critic output as well. 
* On a deep learning level this makes a ton of sense. How are you supposed to train a model to play games where a score of 2 million is considered middling in one game and a score of 1 is total victory in another? That would surely mess with your gradient calculations. In a sense, this is a *normalization* layer that is applied to handle a variety of situations. 

An important technique for allowing the critic to generalize is the use of two-hot encoding. I'll be honest, the details somewhat evade me here. From what I can tell, this is sort of an evolution from asking the critic to guess a single value versus guessing a probability distribution over values, after which the actual value predicted by the critic is found via the expected value of that distribution. Granted, the distribution is simple: it's a weighting over a fixed set of bins. Presumably, having a standard input (a fixed set of bins) is a more flexible way of representation that is suitable for a way larger variety of environments. Naturally, the critic goes from being trained with respect to mean-square error to being trained with categorical cross-entropy.

Interestingly, there was an algorithmic change that took place in this paper as well - instead of using the target critic to calculate reward, the *current* critic is used. I suppose the reason for this is to propagate value information quicker throughout the agent-world-critic loop, but I'm a little unsure as to what the risks that this technique entails and how the bootstrapping affects the value estimates during early training. 

All in all, Dreamer V3 is probably the most technically complex and formidable piece of AI that I've had the pleasure to study so far. As might be expected from this literature review, I'm going to have to go through a few passes of these models before they actually make full sense to me. I daresay that the only way I'll truly understand this paper is by trying to reproduce it... and that sounds quite difficult. Still, it must be done!