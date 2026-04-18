---
title: Deep reinforcement learning experiments
category: projects
---

After spending a month getting to speed with reinforcement learning theory, it's time to finally get into the weeds of this field. The work here will slowly generate technical notes as I build algorithms and spend what I anticipate to be a great deal of my time fixing bugs and getting the processes to actually work as intended.

All of this work will be hosted in the [deep_rl_experiments](https://github.com/aravinthen/deep_rl_experiments) library. I'll be following a somewhat strict path of development that,
1. initially focuses on using high-level libraries in known environments to study the dynamics of the problem,
2. reproduces the algorithms used in Step 1 with custom implementation to get into the nitty-gritty of how the technique actually works,
3. implements a high-performance variant of the algorithm in `TorchRL`,
4. either adds a new feature to the existing implementations or develop the algorithms into distributed variants. 

## Technical notes
1. Lessons from classical reinforcement learning (in-progress)