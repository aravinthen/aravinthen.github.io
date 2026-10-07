---
title: The emergence of reasoning
category: literature
math: true
---
# The emergence of reasoning
I have now read an entire book on reasoning models and spent quite a bit of time thinking about them. I know about chains of thought, reinforcement learning and that the latter can create the emergence of reasoning in language models.
What I don't know yet is *how* reasoning actually emerges as a phenomenon. This is, essentially, the reason I read the book on this in the first place. However, some investigation on the matter makes it clear that reasoning in LLMs is something of a phenomenological distinction: reasoning could be thought of as search, and search is an important componet of exploration.

Exploration is a foundational mechanism in reinforcement learning, although in standard treatments it takes place within training. All of the algorithms I've written about use some kind of exploration mechanism alongside their natural exploitation, be it through a greedy strategy or high probabilities for given actions. However, I'd like to distinguish the kind of exploration I'm interested in here: the emergent exploration that serves as a learned strategy when the agent encounters scenarios of complexity or uncertainty. As such, I and some other researchers have gotten together to discuss the topic of exploration in general. My angle will be to study the problem from the reinforcement learning framework and find out 
1. how exploration emerges in classical reinforcement learning, 
2. how modern agentic systems manifest the ability to reason,
3. how deployed agents apply that reasoning capability online

I won't really be going into depth on 3, mostly because this part is being handled by another person in the group.

Why am I interested in this in the first place? Well, the way I see it, agents do most of their dumb stuff when they're exploring routes into a problem. Take the scenario of an agent breaking out of it's container to use the internet: it's doing that because it's in a situation where it doesn't have enough information to form a cogizent answer and must explore (even the most ridiculous routes) to get a suitable answer. My idea here is the following: if we can understand what exploration in agents is and how it is fostered, could we also control it? Let's dive in.

## What actually *is* exploration?
My first approach to the problem was to find techniques that slowly departed from the paradigm of simply adding a random element to allow for exploration. Just doing something random might be a useful strategy in aggregate, but it lacks the basic goal of exploration where *information* is gained about the problem or environment at hand. Essentially, I was trying to find forms of the exploration mechanic that allowed for useful experimenation over simply exploring via random walk.

This led me to the idea of variational information maximising exploration [1], which roughly works by 
1. initialising an uncertainty over the state of the transition function,
2. observing a transition and updating it's beliefs (and tweaking a specific uncertainty parameter),
3. classifying how interesting that transition is by measuring (using a KL divergence derived metric) how much the new observation changes the agent beliefs. 
4. rewarding the agent explicitly using a weighted KL divergence term to emphize good exploration.

This provides a sharp, elegant way of incorporating personal uncertainty, and the fact that this takes place in the transition function means that the uncertainty itself drives exploration. I should note that whilst the technical side is really rich (they approximate the Bayes posterior using a variational Bayes neural network), the genuinely useful part of the paper was the exposition of what exploration actually means: it's when you react to uncertainty by employing an *information seeking* action, which in turn leads to an update in your beliefs.

I highly doubt that this is how reasoning agents actually manifest exploration capabilities, but this is a useful framework: we have to find a means of classifying actions based on how much information they gain.

## Smarter random walks
One distinction that I realised from the previous situation is that a core marker of demonstrating exploration capabilities was the added feature of *efficiency*. What, precisely, is VIME doing? It's cutting down the search space of potentially actions *intelligently* by associating with prior information. 
It should be noted however that VIME isn't really capable of exploring yet. It's *still* a training focused algorithm: exploration is a mechanism form improving agent performance. Where does *learned* exploration come in?

The first paper I can find (and I searched quite a bit) that comes across a technique for building agents that learn how to learn is the $$RL^2$$ technique [2], which uses enables fast reinforcement learning via slow reinforcement learning.
Now, this is cool as hell. It uses the a variant bandit problem as a testbed (throwback to Sutton and Barto) and essentially approximates a learning algorithm via an RNN. 

This took me ages to figure out. Too much time. But the idea is genuinely quite profound.
1. You have *slow* learning, which is where you train the weights of your RNN across a bunch of bandit problems. 
    * This takes place during the training phase, where you play multiple episodes of a given problem and using the RNN. 
    * Following these episodes, you tweak the permanent weights so that the agent becomes better at remembering necessary queues. You're essentially developing a means of building in working memory into your agents.
2. In deployment, you freeze the weights of the RNN. You then
    * Check the agent into a new environment,
    * Take a set of actions and feed the history into the RNN. The network activations are remembered, so the agent itself now internalizes a key feature of the environment.
    * Following this, the agent uses and updates its history to both learn more about the task through exploration, store the results and then eventually master the task. 

This is the first example that I've seeon of Meta-RL - learning to learn. It's also a foundational result in that it proves that AI can independently discover it's own exploration/exploitation patterns just through learning. Your agent is taking its expreince and figuring out exactly how it should learn next.

Something interesting here comes from where the exploration strategy is stored: it exists within the weights of the RNN. However, the information about the task lives in the hidden activations. This is an interesting arrangement of parameters: you have a parameter set that provides information on how to explore, as well as a parameter set that encodes what it is that you've learned. 

A really, really smart paper... but not quite what I want to understand. Right now, we have working proof that *RNNs* are capable of learning how to learn. Are transformers much different? Is there a correspondent difference between exploration strategy weights and working memory activations?

## Exploration and reasoning in agents
The first paper I've seen that looks into exploration specifically agents is the DeepSeek-R1 paper [3], which is interesting but quite long and dense. The basic ideas that I get from this paper are the fact that *just by GPRO* and a very simple reward (accuracy and format), you get a range of behaviours that naturally map to reasoning. From the paper, you can see
* reasoning, 
* verification,
* self-reflection
emerge from just RL (in the R1-Zero model case) well as other capabilities. I didn't get much out of this paper in terms of the why's of exploration: it just says that the model naturally learns to allocate more thinking time. Like... what?! There are some cool features, where the the training process triggers *pausing*, *reconsidering* and switch tactics mid-thought. Very cool! ~~In fairness, a lot of this exploration is inherent in GRPO, which generates multiple responses~~ no, that's not right. Sure, they use multiple generated responses in training, but there's no impetus at all to do so via training. 

Another paper [4] makes it quite explicit that RL doesn't specifically instill reasoning capabilities in the paper, but mostly optimizes the sampling efficiency of trains present in the base model. I knew this already, but the language here is interesting: RL does not expand the "reasoning boundaries" of LLMs...
This paper is a dirct challenge to the idea of learned exploration in the first place. The conclusion of the papers is that *sampling* efficieny becomes a proxy for exploration, which makes the burden of proof much higher for actually demonstrating emergent reasoning.

## Expanding reasoning boundaries directly
We introduced the idea of a reasoning boundary in the last paper, which is effectively a shorthand for the limits of the specifically cognitive/problem-solving trajectories that a model can generate. It it perhaps possible that this boundary has some kind of flexibility depending on training?

The idea of *prolonged RL* [5] probes this specifically. The idea here is that sustained RL with mechanisms for preserving exploration might itself by a strong mechanism for discovering new behaviours. The routine in this paper is quite complex, but the most useful thing here that it seems evident that the model is capable of uncovering novel reasoning strategies that *are completely inaccessible to base models*. This flies contrary to the previous paper, although I find this somewhat more compelling.

## Improving reasoning without human assistance
An *incredibly* cool paper that I came across in the literature review was the one that introduces the Absolute Zero Reasoner [6]. The idea behind this is allow agents to develop reasoning without using human input/data at all. The innovation here is that the agents improve via an autonomous slef-play loop:
1. The model basically invents its own coding and maths tasks, testing three core logic types:
    * Deduction, which is predicting code output from input,
    * Abduction, inferring the input from an output,
    * Induction, building a program from scratch
2. Instead of guessing or asking a human, the model uses an execution environment as a referee. This essentially implements a reward model based on verification. 
3. The proposer *also* learns from the solver, making it capable of developing problems that are closer to the agent's current skill. 

This is a different kind of exploration, where previously exploration was defined out of just the environment. Absolute Zero, on the other hand, *constructs it's own learning environment*! Still an example of training exploration, but a powerful version. I really quite like this class of technique!

## Conclusion
To be added. Need to take some time to digest all of this first!

## Bibliography
1. [VIME: Variational Information Maximizing Exploration - Houthooft et. al. (2016)](https://arxiv.org/abs/1605.09674)
2. [RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning - Duan et. al. (2016)](https://arxiv.org/abs/1611.02779)
3. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - DeepSeekAI (2025)](https://arxiv.org/abs/2501.12948)
4. [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? - Yue et. al. (2025)](https://arxiv.org/abs/2504.13837)
5. [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models - Liu et. al. (2025)](https://arxiv.org/abs/2505.24864)
6. [Absolute Zero: Reinforced Self-play Reasoning with Zero Data - Zhao et. al. (2025)](https://arxiv.org/abs/2505.03335)

