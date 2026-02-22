---
title: Literature review - Week 7
category: literature
---
# Literature review - Week 7
Back to studying deep learning architectures. I found the papers studied this week just a bit more difficult than usual - not just because they were particularly technical but more so because the content here was perhaps a little more abstract than what I'm used to. Extending the ideas of transformer attention to graph neural networks was also very challenging, but in a way was maybe one of the more useful endeavours of this week, mostly because doing so made transformers clearer to me as well.
This week also marked my introduction to diffusion models, a topic that is familiar to me due to my history in studying statistical physics but is completely new to me as a learning architecture. 

## Relational inductive biases, deep learning, and graph networks - Battaglia et al. 
A dense paper and something that I had to spend a bit of time digesting throughout the week. On the plus side, this was a very useful paper to read - it provides, like no other resource I've read so far, the clearest possible explanation of an inductive bias in the role of deep learnign models.
The graph network approach discussed in this paper is really more of a formalism or methodology over a pure GNN technique. In fact, it's the full generalization of the GNN approach of message passing, update and readout. There's levels to this: take the basic convolution over graphs approach that I looked into a few weeks ago. The very basic method, where machine learning isn't even applied, is the first level of using graphs as models. The GN approach is the very end of this spectrum, where everything in the process can be represented as a learnable function. 

I had a question of my first day of reading as to what the distinction between the message-passing approach of Gilmer/Kipf/Hamilton and the GN approach by Battaglia and Sanchez-Gonzales really is. I think I can answer that now in the message passing is simply an intermediary level in the spectrum that I just attempted to describe. 

My second question when reading this paper was the following: **Why don't the message passing networks discussed in Gilmer et al. exhibit the kind of combinatorial generalization discussed?** My answer to this is that the application discussed in Gilmer et al. is not as expressive as the GN framework, *purely* because the latter is capable of representing all of the graph as a learnable function - *even the edges*. A key ingredient to the GN approach is that edges can also be updated and participate in message passing - this is hard to do without learnable functions. 

I do not personally believe the claims of the paper that graph networks are capable of combinatorial generation, or at the very least I do not feel that combinatorial generalization is possible with GNs for anything above relatively simple systems. The results of Sanchez-Gonzales mode it clear that there's a point in system size where the graph networks deviate considerably for control and system identification. I feel like there's a limited amount of crossover that falls apart when the architecture handles physics and second/third/fourth order interactions that don't exist within its training data.

Regardless, excellent and incredibly instructive paper. Fully recommended for nailing down what graph learning is capable of. 

## Deep unsupervised learning using non-equilibrium thermodynamics - Sohl-Dickstein et al.
A very cool paper - the first of the efforts that led to the now ubiquitous diffusion models. I think the core ideas are quite simple to digest:
1. The aim of learning is to take a low entropy representation and drive it to a high entropy representation, whilst simultaneously learning how to drive it back to a lower entropy representation.
2. Take an image, apply forward diffusion until the image is pure noise. 
3. Following this, learn to reverse the diffusion so that the log likelihood of the posterior distribution of the forward diffusion kernel is minimised. This will give you the reverse diffusion operator. 

The diffusion model here is the reverse diffusion operator. It is quite literally an MPL trained to remove noise in a specific way.

The power of this approach comes from the very cool idea that high-entropy states are stores of possibility. I guess you could say that vanilla neural networks are do something similar: they start off with random weights and tuning those weights to a specific and useful representation for learning is on a basic level the same idea. However, diffusion models take this further by applying noise to the *data*. 

I'm looking forward to diving further into the essence of diffusion models. The results here are really impressive, but this is clearly the first stage. 

## A Generalization of Transformer Networks to Graphs - Dwivedi and Bresson
Understanding this paper was a two day effort. I needed to use an auxilliary resource in order to really figure this out.
I now understand however that transformers are actually significantly closer to graph neural networks than ordinary MLPs. Indeed, the MLP is a basic component of both approaches.

A graph transformer is a version of a transformer with significantly stronger biases that are included within a provided graph topology. Vanilla transformers are fully connected graphs that connect all tokens in a sequence together - there is *no* applied inductive bias because isn't particularly dense with structure. One benefit of using a graph transformer is that by applying the biases in the model directly, you reduce the search space necessary to train a transformer. 

I did not understand the positional encodings at first. In all honestly, the blog post in the next section makes this very clear - a graph transformer uses specific metrics (in the paper, the eigenvalues of the graph laplacian) to encode the relative position from one node to another. It should also be noted that there's a variant of the graph transformer that allows edge-level calculations to be made, improving the predictive capability in the same way that graph networks do.

I learned as much about graph transformers as I did about vanilla transformers after reading this paper. :) 

## [An introduction to Graph Transformers](https://kumo.ai/research/introduction-to-graph-transformers/) - Lopez, Fey and Leskovec.
Link to personal notes (https://aravinthen.github.io/log/2026-02-20/). This was a really good literature summary on graph transformers by itself!