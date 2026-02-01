---
title: Literature review - Week 4
category: literature
---
# Literature review - Week 4
A lighter reading week, although one that came in very handy in terms of solidifying some conceptual frameworks surrounding the field of deep learning and reinforcement learning as a whole. A reading week like this was very welcome given the fact that the week before was dedicated to some quite technically difficult material. That being said, I'll be thinking about the *Bitter Lesson* and the *First Law of Complexodynamics* for a good while after this. 

## The Hardware Lottery - Sara Hooker
This was a very useful essay for a variety of reasons. Weirdly enough, I used some of the arguments therein at my workplace last week (although not in the context of any secret projects, don't worry!). A must read for any algorithms researcher, for sure. 

Notes:
* The essay was written a while ago - ancient history in machine learning terms. Many of the predictions in the essay have come to pass or at the very least have begun coming to pass. 
* The first thing that I thought of when I came to know about this essay was the prevalence of the current form of personal computer over the LISP machine that was almost the lingua franca of artificial intelligence from the 60s to the 90s. I was encouraged that this distintion was mentioned within the paper, as well as an explanatory reason for why LISP is *no longer* the basic language of artificial intelligence. 
* The basic thesis in the essay is simple: ideas in software are adopted due to the availability of hardware. Some amazing, powerful ideas (like the "connectionist" approaches of neural networks) lay dormant over 'inferior' ideas (in this case, the symbolic approaches of logic programming) because the hardware of the time isn't capable of realising those ideas in an effective way.
* Hooker has a compelling argument for the evolution of AI techniques in an age where hardware becomes a flexible choice: it's likely that the transformer and the connectionist neural networks ideas *are not* the end architecture of deep learning. That architecture may in fact be obscured due to hardware constraints, only to emerge when the ability to actually realise the architecture emerges in hardware. 

Remarks:
* I've been looking closely into probabilistic computing lately. I have a slight feeling that one day, probably sooner than we expect, truly probabilistic approaches will be able to make dents into difficult optimisation problems that *could* be useful for AI. 
* On another note, is quantum hardware a potential next step for deep learning architectures? My honest opinion is no, not for a long time. This issue with a quantum computer is that *nobody really knows how to program the damn things yet*! 

## The Bitter Lesson - Richard Sutton
A classic argument for AI and an exceptionally lucid and important treatment of *why* AI matters as a problem solving technique. 
I was thinking about the bitter lesson today when solving some algebraic problems for fun. I realised that a *lot* of my problem solving strategy follows an aesthetic goal in mind: neat, well ordered arguments that can easily be read and written down. 
AI does not have these problems. It does not care how nicely the expressions are written down and the logical flow in which they are evaluated. The only thing AI cares about is whether or not the algebraic pattern approximates to some generalised encoding (learning) and whether or not it has enough compute to search for the solution (compute). That, alone, gives it the potential to become a much, much better mathematician than I am.

Notes:
* Sutton's essay neatly summarises the algebra argument I provided just now by explaining how human methods overcomplicate the problem in some way that makes them less suited to computation.
* In a sense, the bitter lesson is a general statement in the field of complex systems science. It says that theorizing about systems and develop general laws of complexiy is bound to lead to failure due to the inherent limitations of the human mind to grasp complexity in the first place. Instead of wasting time trying to build simple laws for generally intractable systems, it would be more appropriate to build *meta-laws* that can be used to discover those laws in the first place.
* As someone who was once hoping to build a career in theorizing about complex physical systems, I'm a little bitter at this realisation. :) 

## World Models - David and Schmidhuber.
This was an amazing paper. In general, it was lucidly written and the ideas and motivation behind the paper were the stuff of near-future science fiction. This is the kind of thing I imagined deep reinforcement learning to be like before I started working in the field!

Notes:
* The paper attempts to make use of the fact that humans don't *actually* experience the real world as it is, but a generalized model of the real world that acts to compress the enormous amount of information that the world itself provides. 
* In a sense, this paper demonstrates an approach to reducing the size of a neural network representing the agent by decoupling the part of the network that processes the environment from the part of the environment that determines actions. It's a very, very practical means of compartmentalizing the agent architecture. 
* There are three networks that are trained within the paper:
    * Vision - The compression module. This is merely a VAE (I say merely - I haven't studied VAEs yet. I should probably get onto that ASAP.)
    * Memory - Where the vision network compresses over "space", the memory network compresses over time. The goal of the memory network is to generate a predictive model of the future based on the latent spatial vector provided by the vision network.
    * Controller - This network takes in the the vision output and the memory output as a concatenated vector and generates the action of the agent. It is very, very simple compared to the other architectures at hand.
* The results of the paper were stunning. It beat a ton of other SotA approaches. It also provided a means of training against the "dream" state provided by vision network, which worked quite well for tasks that didn't require a particular termination condition from the environment. 
* The way in which the complex cognitive system of the agents was architected here was awesome. I'm looking forward to seeing how other people solved this problem or extended upon the idea of a world model. :)   

Remarks:
* I wonder what would happen if you used more complex models for the vision, memory and controller? A transformer for memory and a convolutional network for the controller (assuming some basic structure exists in the vision output)? 

## The First Law of Complexodynamics - Scott Aaronson
This one was from Ilya Sutskever's list. I wasn't particular sure why this was mentioned in the list at first. My confusion was compounded when I realised that the writer of the post was none other than Scott Aaronson, who I knew to be a mathematician working in quantum computing - nowhere near the field of deep learning. However, *suddenly* the reason for inclusion hit. :) 

My notes on this post can be found in my daily log over [here](https://aravinthen.github.io/log/2026-01-29/). 

## Neural Message Passing for Quantum Chemistry - Gilmer et al.
This paper introduced the basic framework behind the graph neural network. I would never have guessed that *this* was the paper that I shoudl read in order to truly understand the motivation behind a graph neural network, specifically because I thought that any application of graph neural networks to quantum chemistry would necessarily require huge input from the theory behind quantum chemistry. Luckily, it didn't!

I'll be honest, I mostly ignored the quantum chemistry aspects of this paper. It was interesting to read about the results, but frankly the value of quantum chemistry came from the fact that it presented readily available datasets of graph-based data, as well as a good list of tasks that had good correspondence to a variety of machine learning tasks. The actual content of the paper that was relevant towards graph neural networks was the framework introduced to describe general graph-based machine learning techniques.

Notes:
* The essence of GNNs (or message passing, as described in the paper) are the following processes, each represented by differentiable functions:
    * Message passing - a function that takes in a node and it's neighbours and provides a "message". 
    * Update - a function that takes in a message and updates the node with a new hidden states based on the previous hidden state.
    * Readout - a function that computes a feature vector for every node of the graph given the hidden state associated with it. 
* This is a very general framework for graph neural networks. In a sense, any graph neural network can be described using these three operations: a fundamental means of describing the basic structure of a graph neural network application. 

Remarks:
* I would write more, but my train is about to reach it's destination and I think I've written as much as is necessary about this paper. This isn't a copout, I promise!  