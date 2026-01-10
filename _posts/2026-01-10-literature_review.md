---
title: "Literature review - Week 1 "
category: literature
---

# Papers read
* Mastering the game of Go with deep neural networks and tree search - Silver et. al. 2016
* Graph networks as learnable physics engines for inference and control - Sanchez-Gonzales et. al. 2018

# Mastering the game of Go with deep neural networks and tree search - Silver et. al. 2016  
Bonafide classic. The results of this paper changed my life! It's nice to finally read it as a serious academic endeavour.

It's also exceptionally readable, so I'm not going to analythe results in depth. Some remarks:
1. The systems engineering here is amazing. They take a problem, break it down, split it into it's base components and augment the necessary aspects with deep learning. I think that's the coolest thing about this paper: you can still quite clearly see the game playing mechanics in the full methodology.
2. Interesting to see the use of convolutional neural networks, and *very* deep ones too. Just on principle I can see why this would be useful: the game of Go operates on *structures*. It's a magnificent game.
3. Again, the essence here isn't really the deep learning stuff. It's the fact that two techniques - reinforcement learning and Monte Carlo tree search - are used to augment a very natural analysis of how the game of Go is actually played. This is a very, very subtle form of *bias*: I can see why later developments built away from this methodology.
4. The value function is representative of slow and strategic thinking, whereas the policy function used to evaluate rollouts is fast thinking. I *really* appreciated the discussion of the value function reinforcement learning approach, where self-play is used to generate *more* games over just improving a policy. This has shaped my thinking on supervised learning as applied to games. 

All in all, good legacy reading.

# Graph networks as learnable physics engines for inference and control - Sanchez-Gonzales et. al. 2018
Graph neural networks are *advanced* architectures. They're a means of implementing inductive biases in to deep learning models, something that can dramatically improve performance in a number of tasks (prediction and inference, as well as subsequent control).

By partitioning a system in objects and their relations, the problem simplifies: instead of building massive models that treat every component of the system as an individual block, you define general models for
* each class of object,
* the pairwise relationships between each kind of object.

To illustrate, take a paintball game with ten players on two separate team. Before you'd have to build a massive model that somehow takes into account all twenty players, a bunch of environment objects (say for now, a barrier that players can use to hide behind), *as well as their interactions*. With a graph network approach, 
* define a **single** model for friendly players, enemy players and the barrier.
* define models that represent just the different classes of interaction that can take place between the object. 

Overall, that's *five* models in total, related as a graph like so:
* Player node
* Enemy node
* Barrier node
* Player/Enemy - Barrier edge
* Player - Enemy edge

Significantly simpler and *way* more scalable. Say we *don't* use a graph network and just chuck all of this into a big model. What would we do if we wanted to add a new player to our game? With graphs, nothing changes. With the single model, you have to scrap it and start from scratch. 

Some notes:
* The edge models allow for knowledge of disparate object to be shared between the full system.
* The paper introduces a number of functionalities. Forward models models predict the difference between the current state and the next state. Inference models carry out system identification, which is where the unobserved properties of the system (the physics not encapsulated in the observations alone) are predicted. 
* Forward models and inference models are used in conjunction to allow for *control*. There's a very subtle balance here: system identification and control are yet another face of the exploration/exploitation conundrum. 
* GNNs outperformed quite a bit in increasingly complex systems. The effect was more pronounced when the systems had repeated subunits - I think this is to be expected.
 

Interesting features:
* The static and dynamic properties are represented as different graphs. This is interesting to me: in a sense, this is a very natural distinction to make. Looking back on state spaces in physics: positions and velocities are completely independent coordinates of each other. 
* What I found *really* interesting was that GN models were capable of zero-shot generalization... to a point. As you scale the system to a larger size the performance begins to degrade. I feel like the cause of this has to be due to new properties that emerge as a consequence of *system size*. **How the hell do you fix something like that? How do you account for the complexities that emerge at scale?**.

# [Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html?utm_source=chatgpt.com) - Andy Jones
A collection of useful advice for people trying to debug reinforcement learning systems. In summary,
* RL is really hard in general - expecting magic is a surefire way to disappoint yourself.
* The primary source of difficulty lies in the fact that errors aren't local: everything messes up at once, which in turn makes it hard to locate exactly what is causing who to screw up.

On testing: fast, reliable tests are key. I'm guessing that one means of going about this is to have a toy problem base scenario that can be used freely for testing. I was at first wondering how this would apply to debugging something like a deep learning architecture, but according to a HackerNews post from Ilya Sutskever linked, architectures actually aren't usually the biggest problems in the debugging process.

The common fixes listed include:
1. Hand-tuning your reward scale, which is intuitive. 
2. Using a really large batch size, which is also intuitive. Large batches provide less variance.
3. Using a very small network, which goes in hand with the previous point. Granted, the "small" network size in the post seems to be pretty large to me!

I found the discussion of loss curves being a red herring to be really useful. The reasoning here is that loss functions don't localise errors and the shape of the loss function, which is a global quantity, doesn't say much about the localised parts of the code that are contributing to the mess. This is a trap I fall into frequently. 

The most important features of the loss function are the **speed of learning** and the **plateau point**. 

The description of metrics that are instructive is perhaps the most useful part of this literature review.
1. Relative policy entropy describes the variance inherent in the policy network outputs: this is a measure of whether the policy is actually learning: it should start at 1, rapidly fall and then flatten out. This is a very good indicator of whether your policy is stuck in a maximum or minimum.
2. KL divergence - this should be small but positive. KL divergence is a estimate of how the approximation of the distribution is different from the real distribution - it's a measure of relative change based on the scenario. One line sticks out here: if the KL divergence is very large, it means that the agent is learning from experience that is very different to the current policy. Granted, it looks like this is also dependent on the algorithm used.
3. Policy and value losses fall and then level out. I need to investigate occasions where this doesn't happen.

I'll update the other metrics on mentioned here when I work with them directly. Probably going to be referring to this blog post a *lot*.