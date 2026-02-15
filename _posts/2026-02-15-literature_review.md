---
title: Literature review - Week 6
category: literature
---
# Literature review - Week 6
A somewhat smaller, but still quite important literature review for this week. I moved away from my previous investigations on architectures somewhat, although I did spend a bit of time on graph neural networks in the beginning. 
My main question here was the following: **What is the generic structure of a reinforcement learning library?** 
It was clear from the beginning that deep reinforcement learning is a somewhat different beast to the other disciplines of machine learning in the focus on algorithms over architectures. Deep learning libraries, like `PyTorch`, are mostly based around giving users the ability to abstract out the details of the algorithms whilst providing exceptional customizability in architecture and data engineering. Reinforcement learning has great emphasis on the algorithms used to train the architectures involved, which are typically either not too structurally complicated or are composed of multiple different architectures. 

It does not look like reinforcement learning is in any way *solved* yet. Other deep learning fields have grown to the point where very stable workflows exist for the use of the technologies they champion. However, there's a rough progression of workflows in reinforcement learning software that is roughly determined by *modularity* - that is, how easily one might break apart the pieces of the library and rearrange them to make new software. 

The basic verdict that I'm seeing here is that
* a **baselinining** library like `Stablebaselines3` should be used to test environments,
* a **research-based** library like `TorchRL` should be used to test new methods and advanced architectures due to the offered modularity,
* a **distribution** based library like `RLlib` should be used for the purpose of industrial-scale distribution and parallelism. 

Let's go into it! I'll probably be less detailed in this literature review, but the main themes that I've found will be discussed.  

## How powerful are graph neural networks? - Xu et al. 
This was a really interesting paper just for the fact that it ties the use of graph neural networks to an older technique used to determine whether two graphs are identical (isomorphic) or not. I think the most important findings here are the following:
* Graphs are limited by what they can express and this limit emerges from how aggregation functions interact with the graph structure overall. 
* The more popular graph neural network aggregation layers are actually limited in what they can express.
I got the impression when reading that the expressivity of a graph neural network is basically limited by the essential result here: the Weisfeiller-Lehman test, which by itself is a heuristic that is almost identical to the basic pattern of aggregation that GNNs exhibit. The main contribution of the paper is mostly an architecture that is a functional approximation of the Weisfeiller-Lehman test, which by structure is as maximally expressive as can be.

A good paper and one that ought to provide a very useful means of discussing the essence of what GNNs can be used for.

## [Stable-Baselines3: Reliable Reinforcement Learning Implementations](https://araffin.github.io/post/sb3/) - Antonin Raffin
Log post covering this [here](https://aravinthen.github.io/log/2026-02-10/). 

Overall, this is a very good starting point into RL. I think that `Stablebaselines3` will be my entry point into the algorithms that I'll be developing in my own time over the next few weeks. There are limitations to `Stablebaselines3` and I am earnestly trying to get to a point where my usecases are beyond it, but this is a good learning start.

## RLlib: Abstractions for Distributed Reinforcement Learning - Liang et al.
`RLlib` comes across as quite a complex library, *but not for RL*. The problem that `RLlib` tries to solve is that of *distribution*. 
This can can be seen in the basic structure of how `RLlib` algorithms are used within the software. I mentioned during my first pass of the paper that there's a difference in the actor and the learner within `RLlib` - this is an encouraging sign for complex deep learning architectures. However, I do note that there are quite a few algorithmic dependencies throughout the full framework. The software is by no means fully modular - as we'll find out in the next paper, an entirely new data structure had to be developed to handle this kind of modularity.

What is most apparent throughout even the writing of the paper is the level of detail that has gone into studying the parallelism by which reinforcement learning can be distributed. The `RLLib` paper is really in the field of distributed systems over reinforcement learning. I had a look at `RLlib` and saw that there is a massive library of ready-made RL algorithms available. 

As a result, it looks like `RLlib` is worth using for cases where heavy distribution is necessary to carry out advanced RL training loops. The place this library holds in the armory is one where the truly *massive* problems are to be handled - ginormous deep learning architectures, huge numbers of CPUs/GPUs, the like. It may be a while before I get to use this  library - I fear I haven't yet reached the understanding of distributed systems necessary to even utilize it yet. 

I'm glad I read this paper: it has unlocked a new path into the jungle of deep learning infrastructure for me. :) 

## Torch-RL: A Data Driven Decision-Making Library for PyTorch - Bou et al. 
I'm just going to say it: this library is absolutely amazing. It's worth noting that *an entire data structure* (the `TensorDict`) was invented as part of it - one that has now become widely used outside DRL applications. 
The library is built to be fully modular. It is for this purpose that `TensorDict` was even introduced: it's a means of generalizing operations over a *bunch* of tensors. 

Some things I noticed when looking into `TensorDict`:
* You can basically do anything you would do with `Tensor` with `TensorDict`, whilst combining it with the full functionality of `dict`.
* There's a common batch size introduced for all members of the `TensorDict`. This means that you can distribute operations to every member of the dictionary in a very elegant and easy way. *I wish I'd known about this beforehand!*
* `TensorDict` has been engineered to allow efficient distribution. Not even going to try and explain that one (yet).
Really, really cool stuff overall. 
`TorchRL` is a lower-level library than `RLlib` and `StableBaselines3`. It is almost certainly  harder to write *reinforcement learning* routines, because much of the functionality is based around truly modularizable code. On the plus side, I had a look and there quite a large number of training loops that have been built with `TorchRL` as part of the `Trainer` class. 
Of course, one of the huge strengths of the `TorchRL` comes from the interoperability with the `PyTorch` ecosystem: any architecture I can build with `Torch` can be chucked straight into `TorchRL`.

I think the basic workflow here that is emerging in my mind is
1. Use `Stablebaselines3` to try out an environment and attempt to maximise a reward based on the facilities provided by the gold-standard algorithms provided by the software. 
2. Build any algorithms from scratch using pure `PyTorch` in order to truly understand what the algorithm is actually doing and provide some level of direction into what can potentially be enhanced with modern functionalities. 
3. Move over to `TorchRL` to really tweak the architectures and algorithm parameters in a way where I don't have to worry about any performance issues. 
4. If scaling is assessed to be beneficial, move to `Ray` and `RLlib`. Distribute the problem, solve with high compute and assess whether performance improvements can been observed. 

My work for the next few weeks will be to test this generic framework and make adjustments when necessary.