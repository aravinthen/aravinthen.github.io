---
title: 9. The flight to PPO
category: literature
---
# Literature review - Week 9
This entire literature review was written during a flight from London to Mumbai. My basic goal for this flight was to work through every paper necessary to understand *proximal policy optimization*. I have nothing but time and as far as my battery permits (which, being on a five year old iPad, is a bit shit), I'll be working throughout. Let's get on with it!

Also, this flight doesn't have wifi. Let this be proof that all of the writing that I do on this website is my own and not AI. Granted, anybody who has read a few of my posts will probably discount AI involvement just from the number of spelling mistakes!

## Playing Atari with Deep Reinforcement Learning - Minh et al. 
This was the first paper that I've read on using parameterized models to represent actual-value networks: my understanding of function approximation applied to RL has before this been limited to application in policy-gradient techniques. 
The general model approach described in Minh et al. is quite simple. It's not too different from Q-learning by itself: the authors simply replace the Q-function with a neural network. 
What *is* interesting however is the extent of the modifications required to handle this quite dramatic shift in Q-function representation, as well as the actual issues that these modifications attempt to correct.

Let's go through them in order. 
1. When you operate with a neural network, you need a loss function in order to update it. The layman understanding of this process is that you chuck in data, then chuck in a target and then somehow magically things will learn. Loss functions are quite simple in standard machine learning, but how exactly do you think about them in a reinforcement learning sense? What is the *target* when you don't even know whether the data you're getting is even *accurate*? The loss function employed in DQ networks attempts to answer that like so
    * You have the prediction of an agent and a target reward that is obtained through the agent interactions. 
    * The target is the sum of the immediate reward obtained by the agent and the discounted maximum future value, the latter calculated using historical network weights.
    * You then optimize your neural network by *predictive* capability. You're trying figure out whether your guess of the next reward is accurate. Essentially, you're minimizing against the temporal difference error.
2. The use of the above loss function causes some level of issue. The first is the fact that if you change your target too dramatically you will completely destabilize the training. This is a boostrapping issue, where estimates are changing too quickly to provide a stable value estimate. In order to sort this out, the authors use a frozen target network to evaluate the estimate. This network is updated infrequently so that it can at the very least be used to tune to a stable Q-function before it's improved. *Oh, and there you have it!* That's value iteration, the first step of the RL training workflow.
3. The last basic tool employed here is the use of Experience Replay, which is a data structure that attempts to fix the fact that the networks will typically end up copying states that are highly correlated and thus cause the network to display unacceptable amount of bias. Experience replay operates by randomly saving states throughout the exploration steps and then randomly using those states in network training. 

All in all, a really nice paper and one that really opens eyes as to how bloody annoying it is to use function approximation when you're not working with a policy gradient method. Oh, policy gradient theorem, how you spoil us!

One issue that I spotted was the fact that the same issue that manifests in vanilla Q-learning would almost certainly pop up in this variant - you're using the same Q-function to explore as you are in updating. This coupling would cause the same kind of issues that double Q-learning was developed to tackle... and this is something that we'll see dealt with in future papers. 

## [RL102: From Tabular Q-Learning to Deep Q-Learning (DQN)](https://araffin.github.io/post/rl102/) - Raffin
This blog post is best read as supplemental material to the paper above - not because it's particularly supplementary, but because Antonin Raffin is really good at technical writing and explains the full model with exceptional clarity. 

As always, blog posts aren't covered in the literature review - you can find discussion of this post [here](https://aravinthen.github.io/log/2026-03-03/). I'd highly recommend just going through the original post!

## Human-level control through deep reinforcement learning - Mihn et al.
This paper is a short-form version of *Playing Atari with Deep Reinforcement Learning - Minh et al.* The paper itself is really useful as a discussion and evaluation piece, but I don't think it adds much in the way of new technique - everything discussed here has already been covered in the previous reviews. What is *striking*, however, is the fact that there's a very notable difference in the variance in the results depending on the game that is played. In general, looking at the results for this paper makes me think that the strengths of Q-learning emerge when you have exceptionally precise timing combined with a simple action space (given that Pinball has been solved most effectively whist Asteroids has not). 

Oooh. I'm on 34% battery (I started at 45%). It's also made complicated by the fact that the auntie sitting in front of me has fully dropped her seat and my wrists are cramped.

## Deep Reinforcement Learning with Double Q-learning - Hasselt, Guez and Silver.
And here we go - this is what I was referring to with my closing comment of the review in *Playing Atari with Deep Reinforcement Learning*. The overestimation bias is something that is carefully detailed here.

There are multiple contributions to this paper in the form of theoretical results. I'll details those in a bit, but fundamentally the model here isn't so different to the double-Q learning approach proposed to handle the tabular Q-learning case. In fact, it's a very minimal update (which the paper itself mentions). 

A quick recap of overestimation. This is a phenomenon that takes place in noisy environments where the Q-value of a state-action pair can, in circumstances, occasionally be overestimated to be *more* than the true optimum. In situations like this, the algorithm will naturally prefer states that have exhibited those overestimates, which in turn cause issues of bias and poor accuracy. Even worse, given that Q-learning is a temporal difference method that makes use of *bootstrapping*, overestimates usually bleed throughout the value function estimates of other states. 

There's an analogy here. You're a pirate and someone has told you that there's an island of gold to the west. As a result, you go to the West to where this purported island can be found. Your action space is to select an island to said to next - and one day, as you happen upon one island, your telescope glints in *just* the right way as to make the island you're looking at appear to be made of gold. You thus spend many years attempting to find gold on the island, which is in reality a rock with a few weeds growing on it. 

What would help avoid this situation, though? *Another pirate next to you*. "Cap'n", says the other pirate, "thar be no gold on that island. That's a rock with a few weeds on it". Now, as the captain (and the action-setting policy), you might not believe him at first. Chances are however you'll believe him later and, as a result, save many years of your life upon this search. This is essentially what the solution to the problem that double Q-learning presents: you have another network that is used as your assessment and you disentangle the network used to deciding the action with the network used to adjust your value function. 

It really should be noted (by me, at the very least - readers might have realised this before) that overestimation and overoptimism are *not the same*. Overestimation is where you are optimistic *after* your update. If you're simply optimistic at the very beginning of your training, you could potentially have a better time exploring your action space. 

In this paper the authors demonstrate some *hard truths*, that
* Q-learning can be overoptimistic in *large* environments, *even if they're deterministic*.
* The overestimations show up *a lot* and actually does screw with the training process. 

The rest of the paper shows that double Q-learning actually does reduce this harmful overoptimism and the double deep-Q learning is an appropriate extension to this process. 

## Continuous Control with Deep Reinforcement Learning by Lillicrap et al. 
DDQN was a very successful algorithm, but just by the nature of the technique it isn't well suited to continuous action spaces. This is the real of the policy gradient methods which, as they parameterize the *policy*, can represent continuous action via probability distributions.  Also I was correct in my previous assessment that DDQN and the Q-learning methods in general aren't so good at handling problems with large action spaces - policy gradient methods are much better suited to these. 

This paper is an extension of the deterministic policy gradient algorithm, a technique that was discussed in the literature review last week. As a summary, 
* DPG is an actor-critic algorithm. This is where you have one network that selected tha actions and another that evaluates the actions. As mentioned in other places in this blog, the goal of an actor within the policy-gradient update rule is to *surprise* the critic, which *functionally* is used to set the size of the gradient update for the actor. An actor that selects an action that critic *hasn't* got a good guess for will subsequently receive a large update for that particular action-state pair. At the same time, the critic itself is updated via one of the standard methods (typically as a temporal difference update). Note: the role of the critic is to estimate the value of the *next* action-state value. It is, in a sense, the policy-gradient version of a temporal difference method.
* The deterministic policy gradient (DPG) method is an adaptation of actor-critic to work with deterministic policies. In a stochastic policy, you need to weight the gradient of the action with the log probability of the action, whilst in DPG you merely have to weight by the gradient of the action-value function. The original DPG paper shows how a localized policy gradient update can be derived direclty from the policy gradient theorem. 

The statement of work here is a "model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces". The paper builds off the successes discussed at length in the previous sections, namely that
* The network is trained off-policy and uses an experience buffer to handle highly correlated observations,
* The network uses a target Q-network that allows for stable updates. 
These advanced are used in addition to *batch normalization*, a technique that was borrowed from deep learning. The resulting method is dubbed the deep deterministic policy gradient (DDPG) method. 

First, to note: the explanation of Q-learning in the paper is really lucid, being derived form the basic definition of $$Q(s,a)$$! This is a paper one can have at hand on reference as long as the prerequisites are properly understood. 

Second, the model proposed by the authors differs from standard DPG in the following ways:
1. They make use of a replay buffer, which is essentially the same as an experience buffer as described in DQN. The replay buffer is finite: the oldest updates are removed, but the store of "memory" is employed by randomly sampling the experiences within. As mentioned, at each timestep the actor and critic are updated by sampling a minibatch uniformly from the buffer.
2. Interestingly, the target network idea is used to "softly" update the actor-critic networks in very much the same way. I quite like the way that the authors describe this technique as a means of bringing the problem closer to a supervised learning problem: given how reinforcement learning is always described as a different paradigm to the other learning forms, I'm starting to see the deep reinforcement learning isn't so much a different paradigm altogether but something of a merger between two different paradigms. 
3. *Batch normalization* is employed, where the batched inputs to the networks are standardized to have unit mean and variance. The reason this is introduced is to mitigate the problem of dimensional units. It's so simple, such a clear addition... why have I never even thought about this before?

As in the standard DPG method, the whole technique is made off-policy by the introduction of Gaussian noise to the actor policy. Of course, this only really makes sense in the context of a continuous action space. The authors also specify that they sampled from a Ornstein-Uhlenbeck process to produce temporarily correlated noise - clever, as makes *much* more sense when your actions depend on eachother (as they do in robotics - no flailing arms due to total variance).

One important clarification to make that applies to the use of a replay buffer - immediate sample is stored in the replay buffer and then the whole replay buffer is sampled to build a training mini-batch. There's no actual guarantee that the final experience will be used to train the network: it's just likely that it will given the size of the replay buffer. I only realised this after reading the algorithm!

There's an interesting note in the conclusion: "As with most reinforcement learning algorithms, the use of non-linear function approximators nullifies any convergence guarantees; however, our experimental results demonstrate that stable learning without the need for any modifications between environments." I feel as though this is a fundamental consequence of using this approach of using an estimate of the action-value (the target networks) for as a learning target. You have a moving target - *of course* convergence becomes a matter of if rather than a matter of when!

We're at 18%. This is not looking so promising...

## Asynchronous Methods for Deep Reinforcement Learning - Mnih et al. 
This is an implementation paper over a theory paper, but it is very much an enabling function for developing a broader class of reinforcement learning techniques.
The main victim of this paper is the oft-mentioned replay buffer that has shown up in all of the promising examples of deep reinforcement learning so far. However, the paper describes some issues with experience replay techniques: 
* they use more memory and computation per real interaction and
* they are essentially fixed to off-policy learning algorithms.

The contribution here is an alternative way of representing the paper to allow for *on-policy* algorithms to shine. As the paper notes: "instead of experience replay, we asynchronously execute multiple agents in parallel, on multiple instances of the environment". 
Genius. **Genius!** I know I've seen this idea in concept, but looking at the historical path required to come up with this as solution is genuinely another experience altogether! I mean, *of course* you can decorrelate samples by *feeding them batches from different runs!* This opens my eyes somewhat to why minibatches are used in the first place when it comes to training modern RL agents. It's not to make the training more efficient or anything: it's _entirely_ based around dealing with correlated examples. 

So, instead of building an experience replay buffer sequentially, you basically build a fixed experience reply buffer in parallel for every time step. That's the key here. In practice, each thread independently computes a gradient with respect to a global set of parameters. Then, the parameters are asynchronously updated by each step, largely using the fact that the gradient update is a linear process.

The main takeaway from this technique is the fact the importance of architecture in this problem. Just by architectural changes and by making strong use of parallelism the authors have opened up a completely new vista of investigations. It is not *theory* that brought deep learning to on-policy algorithms: it is almost purely *architecture*. 

13%. Three more papers to go! At this rate, I will reach just short of my target paper... although honestly, I'd still be pleased with myself. :)  

Update: Just hit 1% whilst reading the next paper. The journey ends here! :(