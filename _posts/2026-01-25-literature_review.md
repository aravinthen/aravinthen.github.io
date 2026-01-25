---
title: Literature review - Week 3
category: literature
---
# Literature review - Week 3
This week was a journey into the origin and development of the attention mechanism. *What a journey it's been*!
For context, attention is fundamental to the large language models that as of writing are an essential focal point of artificial intelligence as a field. It's essentially a means of mimicking the attention that humans and other intelligent lifeforms exhibit: long story short, it's the ability to maintain context over large distances within a sequence (whether that sequence be words in a sentence or events in a timeline).

It should be noted that many of the papers described below are basically obsolete. The final paper, which introduces the ubiquitous transformer architecture, is technically speaking the only paper necessary to really understand what is actually going on in large language models. However, the journey has been really, really useful. Let's recap it. :) 

We're picking up from the introduction of sequence-to-sequence models by Sutskever et. al., which essentially uses long-short-term memory networks in order to encode an input sentence into a feature embedding and the decode that embedding into a target language. The problem here is the fact the given formation lacks the ability to capture context - this manifested as the inability for a sentence to relate distant parts of itself. The main problem that Sutskever et. al solved with the ability to encode one sequence and output another sequence that was a different length to it. Now it's a matter of solving this pesky problem of context...

## Neural Machine Translation by Jointly Learning to Align and Translate by Bahdenau et al.
The main innovation that Bahdenau et al. put forward from the model in Sutskever et. al. was the basic idea of the attention score, as well as the idea of generating *multiple* feature encodings over just one. This solves the problem of context in an information theoretic sense - the approach of Sutskever et. al just kind of chucked all of the input into a single vector. 

The terminology used in this paper is quite interesting in the way that I have barely seen it used in future papers. The feature encoding process is referred to as "annotation", whereas "attention" is referred to as alignment. 

In a sentence, the reason that attention is necessary in Bahdenau's formulation of the problem is that once you've generated a number of annotations, you have to "manually" build together the contextual associations. Attention is, in a sense, nothing more than a means of generating connections between the embeddings that *once* were associated by virtue of being represented in a single feature vector. 

In terms of architecture, two forms of recurrent neural network were used here. The encoding step was carried out with a bidirectional recurrent neural network, which essentially reads in the input sequence both front-to-back and back-to-front. The output of both of these orderings is concatenated and passed into the decoder, which was specified as a gated recurrent unit. This is a big, big problem - the biggest problem in the practical use of the model proposed. If the networks used are sequential, you cannot parallelize their training. 

## Effective approaches to attention-based neural machine translation by Luong et al.
The approach proposed in this paper was a *window* of localized attention. The overall framework is quite similar to Bahdenau, but the difference is that in the former paper the attention window is carried over the full sequence whilst the latter selectively assesses a window of hidden states. 

The important feature that was proven in this paper is that globalised attention (Bahdenau's approach) is not as effective as the simple local approach. I suppose that this would justify the development of selective attention mechanisms, proving that a "hierarchy" of attention (similar to the hierarchy that emerges from convolution neural networks) is ideal when handling context. 

## Convolutional Sequence-to-Sequence Learning by Gehring et. al. 
This paper further demonstrated the hierarchical structure of attention by demonstrating how convolutions could break attention into *layers* of context. This can most clearly be seen in the fact that each decoder layer has an attention module that can handle different layers of context.

It was also the first paper I read that moved away from the recurrent network approach to modelling sequences. This is a *huge* advance and one that allowed the field to eventually shift to *somewhat* more a parallelizable form of sequence processing. This is the real engineering achievement of the paper: it transforms a slow and impractical model into one that can be trained faster. A clear sign of this that was carried over into Vaswani et. al. was the need to encode positions in a convolutional neural network, as positional context isn't as easily available as it would be with recurrent neural networks.

Now that I think about it, the basic architecture (or user interface) for a transformer was really set through this paper first.  

I realised whilst reviewing my notes for this paper that I use the phrase "The essence of" too much. :( 

## Attention is all you need - Vaswani et al. 
A tour de force and also a paper that completely defeated me the first time I read it. I can now see that I got the basic idea of the paper right, but the deep mechanical aspects of how the full attention process was conducted was something I needed supplementary material for.

There are a number of big ideas in this paper. The first, of course, is the fact that this is the paper in which the transformer was introduced. As a summary, the advances are as follows: 
1. The basic idea of the Gehring et al., where many layers of context are built through stacking attention layers, is essentially mastered in this paper. This is effectively **the** solution for the problem of long range dependencies. 
2. Another big advance is the introduction of multi-headed attention. The use of multi-headed attention allows a transformer to focus on *multiple contexts at the same time*, which in general is probably the reason the transformer is especially powerful.
3. The convolutional and RNN-based architecture was thoroughly abandoned in this paper. The resulting architecture is *immensely* parallelizable. It's worth mentioning that the parallel design of the transformer also implicitly enables the use of multi-headed attention, too.
4. The positional encoding function of the transformer is conducted with sine and cosine functions, which is another remarkable cost-saving measure. 

All in all, a *magnficent* paper. The transformer will probably go down as one of the greatest inventions of this century.  

## The Illustrated Transformer - Jay Alammar
This was supplementary reading to the previous paper. I highly recommend going through the actual post, but my notes on this can be found in my daily log over [here](https://aravinthen.github.io/log/2026-01-25/). 