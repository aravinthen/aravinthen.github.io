---
title: A concise review of world models
category: literature
math: True
---

# A concise review of world models
## Preamble
It's been a while since I've carried out a full literature review. I've still been reading papers, of course, but I haven't had a chance to compile them into the relevant thematic literature reviews that I made a habit of producing a few months ago.

What changed? Well, first of all, my previous topic of research was multi-agent reinforcement learning with a view to self-play. The self-play work became less relevant for me due to a colleague taking over the research direction at work. Besides, I eventually ended up moving harder into post-training. The line of questioning just sort of died out. 

I have, however, found myself in a variety of situations where understanding world models is of very high value. I've read a bunch of papers about this over the last few weeks, many of which are lurking within my logs and my own annotated notes. I've decided to spend a bit of time going through my resources and compiling a unified and concise view of what world models are and how they're represented within literature. I'm hoping this will be useful for others who are interested in the topic. 

## Why learn a model of a world?
Sutton and Barto [1] starts off with a formulation of reinforcement learning that explicitly uses the probabilities of transitions between states to quickly and accurately calculate the value of a given state. These methods (value iteration and policy iteration, which you can read about in [this post](https://aravinthen.github.io/2026/02/28/lessons_from_classical_rl/) are examples of model-based reinforcement learning.

They're great: classic applications of dynamic programming: they're elegant and exact in small tabular scenarios. They're also not really used very much because *we typically don't have accurate models given to us*:
* We often don't know the transition probabilities $$P(s'| s, a)$$, and 
* Explicitly representing them may be infeasible in large/continuous environments.

So what can one do? The next best step is to use model-free algorithms like PPO and many of the other algorithms I've investigated in my other posts. An agent can, in principle, learn which actions work purely from trial and error. This typically allows to to learn incredibly rich internal representations, but they don't explicitly learn and use a predictive dynamics model for planning

Thinking about this from a human perspective: what could one do if the only way to gain information was to just do things repeatedly and learn from trial and error? It's *definitely* a legitimate way of learning (my principle mode, in fact), but it is also extraordinarily inefficient. Interaction is an expensive resource! Environment experience is obtained primarily through
* Simulation, which is slow and expensive to run *whilst also being pretty bad at representing the world*
* Real life experience, which typically doesn't reward millions of attempts at trial and error.

What do humans typically do in these situations? They **imagine**. They play using an internal representation of the world around them. Right now, I'm in a cafe writing out this blog post. What might happen if I ripped off my clothes and started hooting at the top of my voice? I can somewhat imagine the consequences. I don't really need to try it to find out. 

That, my dear reader, is the essence of a **world model**. An internal predictive representation of how relevant aspects of the environments actually evolve. A tool for imagined counterfactual prediction, or analysis of scenarios of what could have happened versus what did happen. Humans have internalized world models that inform essentially everything we do. Some of us have incredibly well honed models that allow us to operate at extraordinarily high levels of physical activity. 

Now, this itself should provide some impetus as to the flaws of a the world modelling approach. I used to train gymnastics - there were plenty of times where my world model was so faulty that I'd end up smashing against whatever apparatus I was training on. One case is particularly interesting, where I landed on my head whilst doing a backflip. To this day, I have a serious aversion to doing backflips - a consequence of a faulty world model that has affected the way I perform. In mathematical terms, we can say that prediction introduced *model error*, which in turn can throw off actor model performance in interesting and complex ways.

World models demand answers of a set of specific and difficult questions. These are
1. Can we learn the underlying dynamical system of the problem we're interested in at all? Can an agent actually construct a useful predictive model?
2. What should the model even predict? If I'm trying to build a world model for, say, predicting football outcomes, I shouldn't really need to worry about modelling what the player ate for breakfast that morning. Or should I? How much detail is actually useful?
3. Presumably, a world model would learn a representation of the world within the compressed space that neural networks typically operate in (latent space). What can we actually do with such representations? Do they have to be translated to interpretable experiences before anything can be done with them?
4. Do we need to plan from scratch each time? Can we use imagined trajectories in their own right to improve internal understanding?
5. How faithful does a world model actually need to be before we get results?A easily interrogable means of making inferences about how the world might act.

These are leading questions. Let's get into the meat of the literature. :) 

## World models as an internal simulator
The first of the questions posed is the essential functionality of a world model. From my investigations, the landmark attempt at fully answering it comes from **World models** by Schmidhuber and Ha (2018) [2]. 

In classical model-based RL, the world and it's transitions are typically represented by a probability distribution,
$$
p(s_{t+1}, r_t | s_t, a_t),
$$
where the next state $$s_{t+1}$$ is accessible based on the action $$a_t$$ and the current state $$s_t$$. As mentioned, the difficult here is that the true state $$s_t$$ may be unavailable or extremely high dimensional: it might be too complicated to actually express. So, the agent must learn
1. A useful representation of the world state,
2. How that world state evolves.

Ha and Schmidhuber solve this problem by introducing three distinct modules that inform the agent - a very simple and damned cool way of representing the core functionalities of an agent. Seriously, I'm not sure whether the chills I'm feeling as I look over my notes on this paper are because the cafe air conditioner is blasting out or if the paper really is *that* magical. 

The three components of Schmidhuber and Ha's agent are
* A perception network $$V$$ which serves to compress what is observed via a variational autoencoder. The compression is then sent to 
* A memory model $$M$$, which predicts how the compressed world evolves. Memory is a strange term for this: "dynamics" model might be a little better, although I can see why the authors named this choice: the model used to represent the dynamics is an RNN. From this prediction,
* A controller network $$C$$ is used to generate an action based on that internal state.

After learning $$M$$, the agent can interact with predicted futures rather than only real ones. This is the killer feature here: an *imaginary* world that can drive decisions. This is the model - the approach belies a new form of model-based reinforcement learning. Excellent! To put some notation to the notes,
* The model-free approach relies on the observation, action, reward and next observation.
* The world model approach uses *the latent representation of the world*, a suggested action, a predicted reward and a prediction of the next latent representation of the world. 

Now, there are some reasons why this paper didn't take off as the end of reinforcement learning. The first is that the learned simulator can make mistakes: it is, after all, an approximation. There are some really interesting error cases and "hallucinations" that are discussed in the original paper, leading to compounding errors. There are also some model-specific errors like catastrophic forgetting and RNN capacity overflows that emerge when trying to handle complex scenarios. The most interesting error case (that I discussed in a previous literature review) is that of model exploitation, where the controller finds and hacks bugs/inaccuracies in the learned model rather than actually learning a robust real world strategy.

The essence of what Ha and Schmidhuber proved was that a world model *actually can* be represented as a learned latent simulator in which an agent can reason about it's immediate action space. There is however *no planning*... what if there was a way of using this latent simulator to plan directly, like how imagination actually operates? 

## Planning in latent space
The paper that answers this *Learning Latent Dynamics for Planning from Pixels* by Hafner et al. [3]. The premise here is that once an agent has learned an internal simulator, how can it actually use that simulator to choose actions over the long term?

The basic achievement of this paper is to enable sample-efficient reinforcement learning from raw pixels by training a recurrent state model to predict future states *entirely* within a lower dimensional latent space. This paper does something quite interesting that seems to have been carried over to modern day world model research: it uses a *recurrent state space model* to represent the dynamics of the world. 
RSSMs combine deterministic and stochastic components to model sequential data. It took me a bit of thinking to figure out how specifically this was done, but the basic idea is that there are *two* representations learned within the model:
* a deterministic state, whch itself is a RNN/GRU which acts as the memory of the system. This is computed deterministically from the previous state and action.
* A stochastic **latent** state, which captures the inherent uncertainty of the environment and allows the hidden state of the RNN to be augmented to handle a wider range of complex sequential data in an interpretable (or at least, learnable) way.

You know what this reminds me of? The *physics* approach to complexity. This is how physicists who work in stochastic thermodynamics and non-equilibrium statistical physicists reason about the world. In fact, Feynman's famous path integral formation actively separates the physics of system of interest from the effect of the "rest of the universe" [4]. I wouldn't go so far as to say that PlaNet does the same, but I *would* say that the same inductive step of carrying out a simplifying separation of the degrees of freedom is analogically equivalent to the approach: you have a standard deterministic latent variable which is augmented with a stochastic variable to represent complexity.

The RSSM compresses raw pixel inputs into a tiny, low-dimensional latent space. Because the transition model can run in this latent space without ever decoding the states back into actual images, we can now simulate thousands of future action sequences to choose the best path forward. Candidate actions can be evaluated by simply rolling the latent dynamics forward - just like imagination. 

How does this actually take place in the paper? 
1. At each decision point, PlaNet considers multiple candidate action sequences and simulates their consequences using the RSSM model. The method *does not use reinforcement learning*: it is an instance of model-predictive control.
2. The agent does not commit to the entire planned sequence. The general process flow here is that the agent first plans over a finite horizon, executes the first action, receives a new observation and that updates the latent state.

The basic approach is sort of reminiscient to what actual people do: they predict a little, act once (?!), observe reality and then replan. 

Doesn't this sound fantastic? It is... but there are limitations here.
* Planning at every environment step is quite expensive. See the blunder notation from the above point? It denotes an important fact: people don't carry out decision calculation after every step. They do a mix of the two: they plan, they carry out a set of steps derived from the planning, and then they plan again. No, they don't act over very long rollouts as in Schmidhuber and Ha's approach, but they also don't plan per step.
* Again, there are model-related issues. Errors in predicting future latent states accumulate quikcly over time, which is something reminiscient to "imagination drift". Again, recurrent neural networks are used here and representation collapse is still not solved in this paper.

## Dreamer: learning from imagination
Where PlaNet uses the model for online planning at every step, Dreamer attempts another purpose: let the imagined trajectories *themselves* be used for training a policy directly.
At this point I've been writing for two hours and the whirlwind of the progression from standard approaches to world models has melded together into a big lump in my mind. If you're easily overwhelmed like me, here are the steps in the ladder that we've climbed so far:
1. Knowing a full representation of the world for a very simple model.
2. Cast the world into the imagination using a world model (Schmidhuber and Ha),
3. Plan using imagination (PlaNet)
4. Learn using imagination (Dreamer). 

I've covered Dreamer somewhat extensively in an earlier literature review. In summary, Dreamer learns a latent world model from real experience, and then uses the model to generate imagined trajectories. As in, where PlaNet generates a ton of predictions of the future and then makes a single step, Dreamer uses those predictions to actually train a model.
The pipeline here is closer to true RL: real experience is generated, which is then encoded within a world model. This is then used to generate imagined latent rollouts, which then are used to inform an actor-critic algorithm to train an agent.

The power of this approach is that the extremely expensive computational step of the previous approach - generating rollouts in latent space - becomes a very valuable source of synthetic training experience.

On my second pass through this paper, I sort of realised that this is less of a means of generating "new" experience as it is a means of compressing repeated planning steps into a reusable mapping to actions. The essence of this approach is to make the *direct* link between the latent state encountered by a model in planning and the action that is best associated with that state: a *mapping* between the two, almost. 

Of course, given that you're now training policies to train acting agents and *also imagined versions of the world*, you're still going to require reasonably large computational budgets. The Dreamer models were trained on *a single GPU*, but took around 1 hour (DayDreamer) to 17 days (Dreamer V3).

Another issue that only really appears in retrospect is the consideration of trajectories that may never have occured in reality. Say you're training an agent to play chess against grandmasters. By virtue of playing against grandmasters, you're typically not going to run into many situations where a grandmaster moves a knight back and forth for the whole game. A world model that is capable of planning and learning in latent space will *likely* consider scenarios where this happens, which is fundamentally wasteful computation. Humans do it too, of course: we call it overthinking.

This made me wonder: if imagined latent trajectories are to be used for control, what exactly should those trajectories really preserve? In fact, one begs the question:

## Is world construction even necessary?
One cannot discuss model-based reinforcement learning without reference to MuZero, the model that 
* mastered some of the most challenging games in the world by simply playing from scratch,
* does so without even being told the rules,
* pretty much killed my love of abstract strategy games (and probably made me far more productive during my PhD).

I had a brief read of MuZero today to note any similarities to the world model approach. Roughly speaking,
1. The model uses a *representation network* to convert the current board state into latent space,
2. Monte-Carlo tree search is used to simulate future paths. At this point, the *dynamics* network is used to take the current hidden state and a potential action. It is this network that is the crux of the efficiency here: the model predicts the next hidden state as well as the reward, biasing specifically for high-reward states.
3. The prediction network is then used to actually evaluate a hidden state. The outputs of this network are a policy - or a set of moves to try - as well as a value, which is a proxy to a value function and provides the likelihood of winning from the position encoded in the hidden state. 

You can see that there are very distinct parallels to the other world model implementations: a means of encoding the world, using that encoding to generate imagined future paths and finally a means of assessing the states. The clever bit is, as mentioned, the prediction model... but the true crux of the approach is the training strategy. One of the key features of the AlphaZero papers is how the data used to inform the models are generated: this is done using self-play. This means that the model will gradually develop a very, very high quality set of games that can be used to encode some really high quality strategies. This is, obviously, quite difficult to learn how to do in the case of (say) Elden Ring, where even *human* players tend to suck at the game for many, many hours.

Why, however, is MuZero relevant to world modelling? It's a clear cut demonstration of why world modelling might not even be the relevant means of modelling decision making in a general way. *MuZero doesn't really seem to care about modelling the full game state* - it only seeks to train around quantities that are directly relevant to planning. Essentially, where other world model approaches generate observations, it seems apparent from this brief read of the MuZero paper that *this is unnecessary* when training full decision agents. 

## The pitfalls
I am a passable chess player: I can beat pretty much anybody who is less than semi-serious, but put me in a room with someone who played the game seriously and I'll be crushed more times than not.
Why, though? What makes those players better than me? Why is my understanding of the game enough to beat weak players but not enough to beat good ones?

The problem is that my internal representation of chess is not accurate enough with respect to a true reward function. My world model and my ability to generate imagined futures (calculation) is sufficient only up to a point.

This is *precisely* the issue with world models in general: a world model is only useful if its imagined futures are useful under the policy that employs them.

The common theme over my investigation was that of compounding errors. Traversing a decision tree is complex and even generalized search techniques like those mentioned can (and often will) make mistakes. However, this is only half the explanation for my chess limitations/world models. My current chess playing policy locks me into standard pattern of play. That means that diverging from this pattern of play is the surest way to beat me: a skilled player sees the way I play and will shift the game into an environment where my model isn't useful. In world model terms, this is called *distributional shift*: the agent is actively shunted into trajectories that are poorly represented by previous experience. 

In literature, these issues culminate in a phenomenon referred to as *world-model exploitation* [6]. I've only briefly read this paper because the cafe is about to close, but the general idea here is that optimization searches for trajectories that look good according to the *learned model*, not the actual environment. The results, then, are merely discoveries of systematic errors over genuine opportunities in the problem space at hand. 

Model failure is also a concern. Yes, RSSM based models are excellent and can be really strong as a means of learning generalized dynamical systems. They still suffer from trade-offs between recent events and oversmoothing. They are definitely prone to making wrong predictions with high confidence. Finally, an old enemy of reinforcement learning rears its ugly head: one might be able to reason with complex states, where an agent is solely uncertain about what happens next... but what happens when you have partial observability? *Partially observable states add another full layer of uncertainty*, where the agent is unsure about *what state the world is ucrrently in*. A very recent paper studies this compounding of partial observability and predictive uncertainty in depth [7].

## Current directions
World models are currently undergoing breakneck development, with the likes of Yann LeCun researching and deploying these techniques in fields that are more often than not focused on robotics. Very briefly, here are some of the main directions that world model research is currently undertaking:
### The foundation model approach 
This approach is based around using massive datasets and employing similar techniques to foundation model training to build world models with a very rich contextual basis to draw upon. [8]
The analogy here is pretty clear cut: foundation models are trained with massive textual databases to become extremely skilled at predicting the next most appropriate word. World models can carry out a similar approach by training on millions of hours of real world footage instead. The natural extension to this is...
### Post-training a world model
The analogue for post-training in world models is where specific robot data is used to support planning in the latent space. Investigation on this topic pending.
### Utilizing more of the imagination
Dreamer has a V4, which I was not aware of until I carried out this review. The basic idea behind Dreamer v4 is to use a scalable learned world model (based on a transformer) to eliminate unsafe/slow action selection.

## Conclusion
This is an overview of the historical progression of world model development, as well as the problems that were solved in order to get to this point. I feel like at right now, I'm pretty much ready to start reading frontier papers on this topic. Just to recap, we went from 
1. learning latent dynamics from an environment, 
2. using those dynamics to plan in imagination,
3. learning directly from the imagination and finally, 
4. model what matters for decisions.

The field has not yet reached it's ChatGPT moments. We are definitely moving closer, although the task is still formidable. The problem isn't just to learn to imagine the future, but learning what to imagine and when imagination tends to fail. These are problems that humans haven't yet mastered themselves...

Another conclusion: I really enjoyed writing out this blog post. It's so valuable to think about technical material in a personal way, which is something that writing really supports. *This* is the reason why one ought to do any kind of art or constructive activity even in the age of AI: the personal connection. I ought to keep doing these literature reviews, regardless of the connection of the writing to my real life applications. 

## References
1. Sutton and Barto
2. World models - Schmidhuber and Ha
3. Learning Latent Dynamics for Planning from Pixels - Hafner et al.
4. Statistical mechanics: A Set of Lectures - Feynman
5. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model - Schrittwieser et. al. 2020
6. Imperfect world models are exploitable - Bhamidipaty et al. 2026
7. How well do latent world models understand partially observable safety constraintsA
8. Genie: Generative Interactive Environments - Bruce et al. 2024
