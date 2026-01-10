---
title: "Literature review - Week 1 "
category: literature
---

# Papers read
* Mastering the game of Go with deep neural networks and tree search - Silver et. al. 2016
* Graph networks as learnable physics engines for inference and control - Sanchez-Gonzales et. al. 2018

# Mastering the game of Go with deep neural networks and tree search - Silver et. al. 2016  
To be added

# Graph networks as learnable physics engines for inference and control - Sanchez-Gonzales et. al. 2018
To be added

# [Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html?utm_source=chatgpt.com) - Andy Jones
A collection of useful advice for people trying to debug reinforcement learning systems. In summary,
* RL is really hard in general - expecting magic is a surefire way to disappoint yourself.
* The primary source of difficulty lies in the fact that errors aren't local: everything messes up at once, which in turn makes it hard to locate exactly what is causing who to screw up.

On testing: fast, reliable tests are key. I'm guessing that one means of going about this is to have a toy problem base scenario that can be used freely for testing. I was at first wondering how this would apply to debugging something like a deep learning architecture, but according to a HackerNews post from Ilya Sutskever linked, architectures actually aren't usually the biggest problems in the debugging process.

The common fixes listed include:
1. Hand-tuning your reward scale, which is intuitive. 
2. Using a really large batch size, which is also intuitive. Large batches provide less variance.
3. Using a very small network, which goes in hand with the previous point. Granted, the "small" network size in the post seems to be pretty large to me!

I found the discussion of loss curves being a red herring to be really useful. The reasoning here is that loss functions don't localise errors and the shape of the loss function, which is a global quantity, doesn't say much about the localised parts of the code that are contributing to the mess. This is a trap I fall into a lot! 

The most important features of the loss function are the **speed of learning** and the **plateau point**. 

The description of metrics that are instructive is perhaps the most useful part of this literature review.
1. Relative policy entropy describes the variance inherent in the policy network outputs: this is a measure of whether the policy is actually learning: it should start at 1, rapidly fall and then flatten out. This is a very good indicator of whether your policy is stuck in a maximum or minimum.
2. KL divergence - this should be small but positive. KL divergence is a estimate of how the approximation of the distribution is different from the real distribution - it's a measure of relative change based on the scenario. One line sticks out here: if the KL divergence is very large, it means that the agent is learning from experience that is very different to the current policy. Granted, it looks like this is also dependent on the algorithm used.
3. Policy and value losses fall and then level out. I need to investigate occasions where this doesn't happen.

I'll update the other metrics on mentioned here when I work with them directly. Probably going to be referring to this blog post a *lot*.

