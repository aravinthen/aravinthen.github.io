---
title: Literature review - Week 8
category: literature
---
# Literature review - Week 8
This week, I diverged from my previous focus on deep learning architectures to focus on the real mission at hand, which is reinforcement learning. This shift has been accompanied by a sharper focus in my research direction, where deep learning is studied less as a field unto itself and more as a means of augmenting autonomy and control in reinforcement learning. 

Something unusual within this literature review is the presence of a book chapter as well as the standard papers. It's a weird thing, "reviewing" a book chapter. Especially when the chapter itself is the culmination of quite a lot additional reading that is necessary to understand it in the first place! There are plenty of really cool books on reinforcement learning that I would be missing out on if I just read research papers. For the purposes of review, I think I'll just be summarising the broader points of what I learned from those book chapters and how the chapter ties in to the wider paper reading effort. 

Enough rambling - let's get on with it!

## *Chapter 13 - Policy Gradient Methods* by Sutton and Barto
I am very glad that I started with this chapter before I leapt into reading papers about the topic. I've noticed that with many early reinforcement learning papers the emphasis is on proving convergence. Whilst this *was* in some way a feature of Chapter 13, the authors treated it as a pedagogical exercise over anything particularly important. 

Anyway, Chapter 13 started by introducing policy parameterization. It also sought to motivate why a policy should be parameterized in the first place. The earlier chapters discussed the approximation of just the *value* function, but I skipped those chapters: I didn't miss much, as the basic failures of value function approximation are clear to read about in the subsequent papers. The real key here is the discussion of how to train a policy once that policy has been parameterized - the **policy gradient theorem** is the core of how this technique is possible and provides an analytical expression for the basic policy parameter update required.

Policy gradient methods typically operate with the assumption that the policy being learned is stochastic. This assumption is relaxed in one of the later papers discussed, but is still very important. I remember when discussing a policy gradient method at work a few months ago I was asked what the exploration parameter for the PPO algorithms is. This question represents a misconception, because there is no specific exploration parameter in a policy gradient algorithm like PPO: exploration is inherent to a stochastic policy, with each action having a probability associated with it based on the observation and environment state. Now, this doesn't mean that a parameterized policy need not eventually become deterministic: it might just be that this happens naturally during a course of parameter updates.

The policy gradient method is one half of the basic RL problem formulation discussed in the first few chapters: it represents policy improvement, whereas the previous methods discussed for function approximation in the books are focused on approximation the value estimation. 

The `REINFORCE` algorithm is a practical simplification of the original policy update rule. It attempts to sample using just *one* action over all possible actions. It achieves this by essentially weighting the policy gradient with the likelihood of the action being selected altogether: in this way, it avoids bias in action selection and can faithfully represent the expected value that the policy gradient theory details. 
It's worth mentioning that the policy gradient theorem is actually quite flexible, given that it's a statement on proportionality. As such, you can introduce a range of possible update rules, with one example being the `REINFORCE` algorithm with a baseline. The baseline is introduced to provide stable updates to the policy, which otherwise is susceptible to significant variance. 

It's worth mentioning that `REINFORCE` *is a Monte Carlo method*. The update itself is carried out after full episode runs. Actor-critic algorithms serves to extend the basic `REINFORCE` with baselines approach by providing an adaptive value function baseline. Normally, `REINFORCE` with baselines employs the value function for the first state as the baseline. Now, this is all well and good, but you sort of have to *have an amazing approximation of the value function in the first place*. Actor-critic algorithms relieve this step: you can effectively generate both the policy (actor) and the value function (critic) at the same time by allowing for bootstrapping. In a sense, the actor-critic approach is the temporal difference method equivalent of `REINFORCE`. 

There are some complications that emerge here. For one, the critic is a parameterized value function: you're using an estimate of the value to update an estimate of the policy. The critic is **subjective**. This in turn requires that you *do not optimize your actor too quickly*! If your actor no longer follows the value function, errors will begin to accumulate. 

That, roughly speaking, is the core of the field of policy parameterization in reinforcement learning. 

## Policy Gradient Methods for Reinforcement Learning with Function Approximation by Sutton et al. 
This is the paper that introduced the `REINFORCE` with baselines method. I feel as though a lot of the discussion of the previous paper covers the essence of this technique, but for completeness,
* Value approximation works really well when you have deterministic policies. They fall apart for stochastic policies!
* Without a baseline, it's likely that variance emerges just from the fact that small changes in the parameterization cause large swings in the action probabilities.
* The paper also formally introduces the idea of **advantage**. This is really important!
    1. Advantage is the difference between the action-value function and the state-value function for a particular state and action. 
    2. Given that the value function is defined as an average over the actions, taking the difference between the state-action value gives an indication of how much better the action is *on average*. 
    3. This is in fact what motivates the baseline in the first place. They isolate the effect of the action. 

## Actor-Critic Algorithms by Konda and Tsitsiklis
The results of this paper were also introduced in Chapter 13 of Sutton and Barto. As a recap of the basic ideas,
* It took me a bit of time to realise that the main point of the paper was the fact that the critic should *also* be parameterized rather than assume the highly complex dimensional form of the true value function. This is an essential point in the paper, but almost a standard assumption in Chapter 13 of Sutton and Barto. 
* This paper is very explicit about the actor-critic algorithm being a temporal difference based method. It does this by literally labelling the critic as a `TD(1)`` critic: it uses one step into the future to provide an estimate of the value function. 
* The paper goes on to demonstrate convergence properties - that the gradient of the policy will reach a local minimum. These aren't the strong convergence properties that I'm used to from physics... I'm wondering whether these convergence properties really emerge in practice or are just theoretical guarantees!

## Deterministic Policy Gradient Algorithms by Silver et al. 
This is the first modern paper that I've read regarding the policy gradient methods. It's an interesting one at that, being probably one of the first papers that DeepMind actually published!

Before this paper, the basic assumption was that policy parameterization could only really apply to stochastic policies. This was handily proven false by the work demonstrated here, which basically demonstrated a working example of a policy gradient update rule for deterministic policies and showed that the rule is as simple as the expected gradient of the action-value function. All you need to do to improve your policy is to move its parameters in the maximal direction suggested by the gradient of Q-value function.
One question I had was **why**? Why worry about this - what is the big issue what probabilistic algorithms in the first place? This was made clear by the fact that probabilistic algorithms have to pick actions, which in MC terms is effectively integrating over an entirely new variable. The new algorithm doesn't have to do this: it simply calculates over the action recommended by the deterministic policy. 

The paper also shows how the method is really well suited to off-policy learning, especially in the context of allowing a stochastic policy to explore the space whilst the results of that policy are used to iteratively improve the target policy. In a sense, this is *way* more intuitive to me than the off-policy target of greedy selection discussed in Sutton and Barto - here, it feels like you're actually learning a full on *policy* over a... greedy policy, which is still a policy but so simple that it might as well be a heuristic. I need to give that a bit of thought, eh...