---
title: On Multi-agent Reinforcement Learning - Game-theoretic prerequisites
category: literature
math: True
---
# Multi-agent Reinforcement Learning 1 - Game-theoretic prerequisites
## Introduction 
I normally carry out a literature review every weekend. I collect that papers that I've read over the week, collect the notes that I've written about them, give the paper a once-over and then write up my understanding of the papers as a whole. I usually follow a theme over the week, reading papers that are either based around an overarching concept or follow a direct trajectory of research. 

Last week, I read three papers on the foundations of multi-agent reinforcement learning (MARL). To say I actually *read* them was an understatement. I *vaguely* understood what they were about after spending most of my reading time struggling to parse the mathematics or the overall themes. When I sat down on Monday to write up my literature review I realised that *I still had no idea what to cohesively write*. I hadn't generated enough insight into the last of the papers that I was supposed to have read. Upon closer examination I didn't really understand the two papers that I'd struggled through before it. What on Earth was I supposed to do for my literature review?

I don't want to get into the practice of pushing out content for the sake of pushing out content. I thus abandoned the idea of writing a literature review and decided to spend a bit of time examining exactly *why* I didn't understand these papers at all. I've spent the last four months diving into the theory and mechanics of reinforcement learning, so what was I missing here? 

I went through the last paper on the reading list. "Extensive games", "Normal form", "equilibria"... I knew what those words *meant* because I'd read about them on Wikipedia or something. However, they conjured no intuitive images for me - they were just nouns, devoid of essence. I realised that my issue wasn't that the papers were particularly hard, it was that *I lacked the language and the intuition to handle this content in the first place*. I very suddenly realised that multi-agent reinforcement learning is almost a misnomer. **Multi-agent reinforcement learning is a subdiscipline of game theory**, and reinforcement learning is applied as a means of solving problems in that field. Trying to understand MARL without understanding game theory is reminiscient of trying to study chess without understanding what a checkmate is. 

I realised that if I wanted to make any progress in this field, I would have to undergo a shift from how I thought about reinforcement learning and optimization. The only way of carrying out this shift would be to bite the bullet and study game theory. 

## References
[1] *Introduction to Game Theory* - Osbourne (2000)
[2] 