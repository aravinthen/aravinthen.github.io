---
title: Literature review - Week 2
category: literature
---
# Literature review - Week 2
Warmer than last week. The literature for this week had two basic themes: doing better deep reinforcement learning and refreshing deep learning architectures. Also notable for this week was progress made on [Ilya Sutskever's famous list of papers that cover 90% of "what matters" in modern AI research](https://github.com/dzyim/ilya-sutskever-recommended-reading). 
My first forays into deep learning were based around application - I wasn't too concerned with what the models actually did. As a physicist, I was mostly expected to treat these models as generalised black boxes: sitting down and actually parsing the details behind their design is proving to be quite instructive. 

### Papers
* Deep Reinforcement Learning that matters* by Henderson et al. 
* Sequence-to-sequence learning with Neural Networks* - Sutskever, Vinyals and Le.
  
### Blog posts  
* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy.
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Chris Olah.

## Deep Reinforcement Learning that Matters - Henderson et. al. 
A useful paper, notable mostly for illustrating both the serious pitfalls in the literature behind DRL as well as the quite limited empirical design that underlies the use of deep architectures in reinforcement learning. 

This paper was recommended as part of the Stablebaselines3 documentation. 

Key points:
* RL results are not consistent over algorithms, environments or even architectures.
* The nature of the environment is very, very important to assessing quality of results. Some environments are unstable by nature: these cannot provide reasonable experimental comparisons by merely taking averages of reward signals.
* Reward scaling is really important. This could also skew results and affect the learning process. 
* One very important if not rather unfortunate finding: the way people report reinforcement learning results can be quite sloppy, where top-N trials are selected. 
* Essentially, two main action points:
    1. One random seed is not enough to determine robust results. In order to run experimental tests, you need to run the experiment across a number of different seeds. 
    2. Bootstrapping and power analysis could *potentially* be used to detect statistically significant performance gains. 
* There's something interesting here though: the use of hyperparameter agnostic reinforcement algorithms. I wonder what that might look like?  

Notes:
* I'm curious about the consistency of results over architectures. I think we can quite easily poinpoint *why* results may be different: using a more expressive architecture for a simple problem would result in memorization, whereas using a simple architecture for a more complex problem would cause a bottleneck in the capability of the agent. 
* The paper is old. I'm almost certain that progress has been made since then... but now I need to find that progress!

## Sequence-to-sequence learning with Neural Networks - Sutskever, Vinyals and Le.
This was a cool paper. I'm trying to build up to a thorough understanding of transformers and this this paper was recommended as a prerequisite. 
Key points:
* The problem at this point in deep architecture research was the inability to handle sequence-to-sequence tasks.
    * The vanilla neural network maps an input to an output.
    * The RNN and the LSTM map a sequence to an output.
    * What do you do if you have a sequence of one size and you want to predict another sequence instead? You need a method that is agnostic when it comes to the output size.
    * You also need a way of encoding the input in a way that the necessary context necessary for the output is accessible to every part of the sequence. In French, "Please call me Aravinthen" would be written more commonly as "Appelle-moi Aravinthen, s'il te plaît". The contextual unit of the sentence that encodes to "please" occurs as the first word in the English and occurs as the last three words in French. How do you even begin to solve this?
* The architecture introduced by the authors uses one LSTM to read the input sequence to obtain a large fixed-dimensional vector representation. That fixed representation would then be used to extract the correct output via another LSTM.
* The temporal dependencies inherent in language make LSTMs entirely necessary for this task. However, *one* LSTM is nowhere near sufficient for the task, mostly because the single LSTM has no awareness of the full sentence, nor the temporal dependencies. 
* It's important to realise that the decoding process takes in one word at a time of the encoded representation. The real brain-jolt comes from understanding that the function that maps the translation of one word to another depends on *context*. The LSTM is a really good way of adding that very difficult context mapping as an input the function. 
* The training of the sequence-to-sequence model proposed took quite a lot of compute, which each layer of the LSTM existing within a different GPU.
    
Notes:
* I had eureka moment regarding attention when I came up with the French-to-English example. :)  
* It's still really weird that reversing the source sentence made a noticeable difference. The explanation offered in the paper proposes that the reversal allows for a structure that has better back-propagation dynamics...

## The Unreasonable Effectiveness of Recurrent Neural Networks - Andrej Karpathy
This is a blog post. I can't annotate blog posts, so I just carry out a full literature review as part of my log. This post was covered in [Less messy sequences](https://aravinthen.github.io/log/2026-01-14/).

Key points:
* I think the most important realisation I had here is was the understanding that an RNN is just a very deep neural network with the same weights for each layer.
* Having a grasp of the formula for the recurrent network architecture was also really useful. Basically, the input vector and the previous hidden state vector are used to update the hidden state vector. The hidden state vector is then used to produce the output. 
* Again, there are issues with this architecture: long-term dependencies break it and overfitting is rife.   

## Understanding LSTM Networks - Chris Olah
Post covered in [Short-short-term-memory](https://aravinthen.github.io/log/2026-01-15/).

Key points:
* All of the LSTM can be modelled as a means of maintaining a internal cell state and deciding which information to add to it and remove from it via the input. This is described within the linked log entry. 
* LSTMs solve the vanishing gradient problem that RNNs suffer from due to the cell state's capability of selectively remembering and forgetting information. This is a means of embedding *context*. 
* Also important to note is that LSTMs rely on additive gradient updates with respect to the cell state. This is a pretty important distinction between the multiplicative updates that RNNs rely on: as such, the context described previously is updated in a more stable way. 