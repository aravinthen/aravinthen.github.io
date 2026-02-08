---
title: Literature review - Week 5
category: literature
---
# Literature review - Week 5
A reasonably intense literature review over this week. I mostly read about architectures with a strong focus on graph neural networks. My interest here comes from the ability of these techniques to essentially reduce the search space of a training architecture by imposing specific inductive biases upon the network as it trains. 
There are some very cool results in these papers, but they are quite mathematical in nature. Graph convolutional networks especially seem to be simple, but have a somewhat higher level of mathematical sophistication than a lot of the other derivations that I've seen in the field of deep learning. However, it is in circumstances like these where using function approximations like neural networks become useful: whilst the graph convolutional network derives from a more complex mathematical expression, the `GraphSAGE` network is almost motivated in an entirely identical way, only with the heavy use of function approximators. 

A personal note, and maybe an encouragement to anybody who might read this: it's remarkable how much of a paper makes sense after you've read the papers that succeed it. I'll be honest, I completely missed the essence of Kipf and Welling the first time I read it, but now I *actually* understand it enough to think about it critically!

## Semi-supervised classification with graph convolutional networks - Kipf and Welling
This paper was difficult upon first reading, although now that I've read what comes after I realise that there's a straightforward and linear flow to how the ideas surrounding graph neural networks can be derived.
This paper is *not* the starting point for GNNs. I would probably go with *Convolutional neural networks on graphs with fast localized spectral filtering* by Defferard et al, although Kipf and Welling provides an entire section where the results of the paper are described.

On first reading, I realise that I got distracted from the main point of the paper by focusing a little too much on the *semi-supervised learning* aspect. This is a discipline of machine learning that I hadn't yet been exposed to. I won't be spending too much time on this, although in retrospect it's important: the semi-supervised learning discussed in this paper is a *transductive* process and the two papers that follow this one focus on generalizing to induction. 

Anyway, let's provide a quick recap of what is being done in this paper. It's worth starting off by stating that the essence of the graph neural network approach is actually exactly the same as that of the convolutional network approach: the goal is to impose biases on data. The convolutional neural network approach is excellent for grid structured data, but not so much for the unstructured relational data that are better modelled on graphs. 
How would one go about developing a convolution for graphs? This is introduced in Defferard et al. and fleshed out nicely in Kipf and Welling. There are purportedly multiple ways of doing this, but the way that is followed here is the spectral approach (where spectral refers to the fact that you're looking at the graph in Fourier space):
* You project the graph signal onto the eigenvectors of the Laplacian.
* Yoiu then multiply the signal by a filter in the spectral domain. The filter here represents the same thing as the convolution kernel does in the convolutional neural network approach.
* Carry out an inverse transform to move the result back to the spatial domain.

The approach of Defferard was to represent the filter as a polynomial expansion. This allows fast approximations by simply truncating the expansion (which is written in the form of Chebyshev polynomials). However, Kipf and Welling go further - they show that
* You don't need higher order polynomials. You only need a single expansion for your convolutional layer...
* ...as long as you stack your single approximation many times over!

This is an example of the basic idea of deep learning and convolutions in general. Stack your layers so that every part of your input is able to affect the output. The *very same* idea shows up in *Convolutional Sequence-to-Sequence Learning* by Gehring et al., where the use of convolutions is sufficient to spread attention over disparate parts of the sequence. 

All in all, a cool idea. I just had to spend a bit of time digesting the mathematics before I could even start to figure out what was going on. 

## Inductive Representation Learning on Large Graphs - Hamilton et al.  
One of the big confusion that emerged when I read Kipf and Welling was the fact that I wasn't quite aware what it really was that made the approach a neural network. This was cleared up when I realised that there are trainable weights present in the forward step of that paper. In a sense, you're training an embedding function for every node in the network - this is what makes Kipf and Welling's method a transductive approach.

*Inductive representations on Graphs*, however, makes it abundantly clear where the functional approximators are within the paper. `GraphSAGE` is a very nice technique: gone are the nasty convolutional steps where you have to appeal to graph Fourier transforms and spectral methods! Instead, the goal of aggregation - which the GCNs did previously - are folded into a set of aggregator functions that are trained to carry out the effective same step as what a GCN would do, but only in a far more flexible way. This paper solves the problem of transductive learning through the aggregator approach, which mean that it can generate embeddings for new, unseen nodes in dynamic graphs without retraining the model. There are a variety of ways that this could be useful in a range of important questions.

The method is meant to operator on large graphs, which is why one of the major additions of this approach is to employ a fixed-size set of neighbours that it can sample from. This is an assumption that is relaxed in the following paper...

## Graph Attention Networks - Velickovic et al.
The main contribution with this paper was an evolution in thought on how the neighbours of the system should influence the aggregation process. Essentially, all this is the extension of Bahdanau's attention mechanism to graphs inputs. There is a standard process here: the attention mechanism requires a linear layer to build a feature encoding of the nodes - this is in general a similar methodology the basic encoder structure that seq2seq models typically employ.

I should note that it looks like - just from the formulation - that GAT is meant to be employed over an entire graph. This is not a technique that should be used for massive graphs: `GraphSAGE` is probably preferable here. There is an important choice to be made here when modelling a system with inductive biases: choose `GAT` when the relative importance of neighbors matters and graph size is manageable and choose `GraphSAGE` when dealing with massive, evolving graphs.

On another note, machine learning is a *brutal* field. If you leave open a path for someone else to explore in a paper, you can damn well be sure that someone is going to have a paper about in a month. In my old field of computational soft matter physics, I had response to a method I produced for generating polymer configurations... a full two *years* after I wrote the paper!

## What is an equivariant neural network? -  Lim and Nelson
This was a very tough paper for me to digest, mostly because it used mathematical methods that I last encountered about six years ago in a class on representation theory. I'm afraid that it's difficult to go too deeply into the structure of this paper, but after about two hours of learning I think I extracted the basic essence behind what this paper is communicating:
* Basic neural networks are not equivariant functions - they do not consider symmetries in their inputs and those symmetries are typically *not* preserved over into the feature space. 
* Equivariant neural networks are a particular brand of neural network that exploit this symmetry to vastly reduce the search space when optimizing a network.
* Convolutional networks are a particular class of equivariant neural network: they support translational invariance in the way that their kernels are defined to operate over the input space and subsequent hidden layers.
* There are however a wider class of symmetries that can be exploited here: rotational invariance, for example.

I'll be honest, I think I'll only really understand when I engineer one myself. The mathematics here was *nasty* and it's going to take me a bit of concentrated effort before I can really figure out the power of this approach.

## [*Deep Reinforcement Learning Doesn't Work Yet*](https://www.alexirpan.com/2018/02/14/rl-hard.html) by Alex Irpan.
As usual with blog posts, a more thorough annotation of this paper can be found on the relevant log over [here](https://aravinthen.github.io/log/2026-02-06/).