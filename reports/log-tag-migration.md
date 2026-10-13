# Log tag migration report

Examined every one of the 285 Markdown entries, including each complete body, in chronological order before editing. The initial inventory contained 31 distinct tags. Most entries concern RL experiments; other sustained subjects include neural architectures, game theory/MARL, infrastructure/software, model post-training, and safety. Narrow algorithm/architecture tags, activity labels, variants such as `foundation-model`/`foundation-models`, and `AI safety`/`safety`/`ai-safety` made the original taxonomy inconsistent.

## Final taxonomy

Eight canonical tags; 160 entries modified, 125 unchanged. Entries receive one tag by default and two for substantial distinct subjects. No entry receives more than two.

- `reinforcement-learning`: classical/deep RL algorithms, control experiments, and model-based RL.
- `deep-learning`: neural architectures, representation learning, transformers/tokenizers, VAEs, and base-model pretraining. Architecture study is substantial enough to merit its own tag; language-model mentions alone do not imply post-training.
- `post-training`: fine-tuning, LoRA, preference learning, distillation, reasoning training and inference-time reasoning methods. Reasoning is folded here to avoid a redundant tag; the interface label makes its broader scope visible.
- `ai-safety`: alignment, monitorability, harmful transfer, safety evaluations and control. Ordinary RL is not tagged safety merely because it motivates later work.
- `multi-agent`: MARL, self-play, strategic games, regret dynamics and swarm-based oversight. These do not automatically also receive the generic RL tag.
- `systems`: software engineering, algorithms/data structures, operating systems, infrastructure and deployment.
- `mathematics`: standalone statistical/mathematical study, rather than equations used within another subject. The June 12 hierarchical-statistics entry justifies this small category.
- `general`: research administration/reflection and zero days with no substantial technical subject. Used for six entries.

## Frequencies

Counts refer to tag assignments; dual-tag entries count once in each category.

### Original

| Tag | Entries |
| --- | ---: |
| `AI safety` | 1 |
| `ai-safety` | 35 |
| `artificial-intelligence` | 2 |
| `back-tracking` | 1 |
| `backtracking` | 1 |
| `binary-search-trees` | 1 |
| `deep-learning` | 34 |
| `depth-first-search` | 1 |
| `diffusion-models` | 1 |
| `distributed-computing` | 1 |
| `equivariant-neural-networks` | 1 |
| `foundation-model` | 1 |
| `foundation-models` | 26 |
| `game-theory` | 2 |
| `graph-neural-networks` | 10 |
| `life-changes` | 1 |
| `literature-review` | 1 |
| `machine-learning` | 1 |
| `maintenance` | 1 |
| `multiprocessing` | 1 |
| `nothing` | 3 |
| `oof` | 1 |
| `reasoning` | 10 |
| `reinforcement-learning` | 150 |
| `safety` | 2 |
| `sequential-modelling` | 1 |
| `software` | 3 |
| `software-engineering` | 23 |
| `systems` | 30 |
| `website` | 1 |
| `world-models` | 3 |

### Canonical

| Tag | Entries |
| --- | ---: |
| `systems` | 65 |
| `reinforcement-learning` | 119 |
| `deep-learning` | 55 |
| `general` | 6 |
| `multi-agent` | 49 |
| `mathematics` | 1 |
| `post-training` | 19 |
| `ai-safety` | 39 |

## Subjective decisions and review boundaries

The complete per-entry original/final mapping is in `log-tag-migration.json`. These assignments are editorial judgments, not automatic keyword classifications. All dual-tag entries are listed below as review candidates.

Additional boundary decisions:

- January 17 and February 1: writing/review activity with limited technical detail; January 17 is general, while February 1 explicitly consolidates architecture study and is deep learning.
- February 2: RL study is mentioned but not substantially developed; graph-convolution study is deep learning only.
- February 5: equivariance is mathematics-heavy, but the actual subject is a neural architecture.
- March 16/18: model serving and infrastructure remain systems despite model-training terminology.
- May 23 and June 14/19/23: environment architecture and data-generator engineering are systems; June 15/20 focus on encoder inputs/training and are deep learning.
- July 8: a zero-day reflection with no developed technical discussion remains general; July 29 SPAR application planning is general.
- July 11/12: future foundation-model plans do not imply substantive post-training yet. July 11 remains RL and July 12 architecture/deep-learning planning.
- July 17/19: VAE work remains deep learning although the intended destination is a world model.
- July 25: implemented tokenizer work is deep learning; the post-training paper is only named.
- August 7/8: test-suite engineering is systems. August 10 safety-project selection is ai-safety, because the named project subjects are explicit; July 29 has no technical project details.
- August 14/29: inference-time reasoning is included under the intentionally broad post-training/reasoning umbrella. This is a practical browsing choice, not a claim that inference is training.
- August 27: revision of base-model prerequisites is deep learning; August 28 KV-cache/inference optimization is systems.
- August 30: safety appears in the original tags, but no substantial safety investigation is developed in the entry, so it is post-training only.
- September 9: software is developed in detail; safety is a brief remark, so systems only.
- September 25/26/28/29: safety through strategic interaction receives ai-safety plus multi-agent.
- September 27: personal change does not require a second general tag when safety work is the principal technical subject.
- October 3: physics analogies are used for interpretability; ai-safety only.
- October 4/6: RL exploration remains RL. October 5 explicitly discusses dangerous exploration and receives a safety second tag.
- October 10: undeveloped mentions of multi-agent/post-training plans do not justify extra tags.
- October 12: the substantial RLHF/post-training discussion receives post-training alongside the safety course work.

No entry is left unclassified. Boundary judgments above remain open to author review; no missing facts were invented. Legacy URLs use broad canonical aliases, so an old narrow filter now selects its broader category rather than exactly reproducing its previous set.

### Dual-subject entries

| Entry | Tags |
| --- | --- |
| 2026-01-02.md | systems, reinforcement-learning |
| 2026-01-05.md | deep-learning, reinforcement-learning |
| 2026-01-06.md | deep-learning, reinforcement-learning |
| 2026-01-12.md | deep-learning, reinforcement-learning |
| 2026-01-15.md | deep-learning, reinforcement-learning |
| 2026-01-19.md | deep-learning, reinforcement-learning |
| 2026-01-20.md | deep-learning, reinforcement-learning |
| 2026-01-21.md | deep-learning, reinforcement-learning |
| 2026-01-22.md | deep-learning, reinforcement-learning |
| 2026-01-23.md | deep-learning, reinforcement-learning |
| 2026-01-26.md | deep-learning, systems |
| 2026-01-29.md | deep-learning, reinforcement-learning |
| 2026-02-03.md | deep-learning, reinforcement-learning |
| 2026-02-04.md | deep-learning, reinforcement-learning |
| 2026-02-07.md | deep-learning, reinforcement-learning |
| 2026-02-08.md | deep-learning, reinforcement-learning |
| 2026-02-11.md | reinforcement-learning, systems |
| 2026-02-15.md | reinforcement-learning, systems |
| 2026-02-16.md | deep-learning, reinforcement-learning |
| 2026-02-18.md | deep-learning, reinforcement-learning |
| 2026-02-19.md | deep-learning, systems |
| 2026-02-28.md | reinforcement-learning, systems |
| 2026-03-21.md | reinforcement-learning, systems |
| 2026-04-03.md | reinforcement-learning, systems |
| 2026-04-05.md | reinforcement-learning, systems |
| 2026-04-07.md | multi-agent, reinforcement-learning |
| 2026-04-08.md | multi-agent, reinforcement-learning |
| 2026-04-09.md | multi-agent, reinforcement-learning |
| 2026-04-10.md | multi-agent, reinforcement-learning |
| 2026-04-11.md | reinforcement-learning, systems |
| 2026-04-16.md | multi-agent, reinforcement-learning |
| 2026-04-17.md | multi-agent, reinforcement-learning |
| 2026-04-18.md | multi-agent, reinforcement-learning |
| 2026-04-22.md | multi-agent, reinforcement-learning |
| 2026-04-23.md | multi-agent, reinforcement-learning |
| 2026-04-24.md | multi-agent, systems |
| 2026-04-25.md | multi-agent, systems |
| 2026-04-26.md | multi-agent, systems |
| 2026-04-27.md | multi-agent, reinforcement-learning |
| 2026-04-28.md | multi-agent, systems |
| 2026-04-29.md | multi-agent, reinforcement-learning |
| 2026-05-02.md | reinforcement-learning, systems |
| 2026-05-04.md | multi-agent, reinforcement-learning |
| 2026-05-06.md | multi-agent, reinforcement-learning |
| 2026-05-11.md | multi-agent, reinforcement-learning |
| 2026-05-28.md | multi-agent, systems |
| 2026-05-29.md | reinforcement-learning, systems |
| 2026-05-30.md | multi-agent, reinforcement-learning |
| 2026-05-31.md | multi-agent, reinforcement-learning |
| 2026-06-01.md | multi-agent, reinforcement-learning |
| 2026-06-02.md | multi-agent, systems |
| 2026-06-03.md | multi-agent, reinforcement-learning |
| 2026-06-29.md | deep-learning, reinforcement-learning |
| 2026-07-05.md | deep-learning, reinforcement-learning |
| 2026-07-22.md | post-training, deep-learning |
| 2026-07-28.md | post-training, deep-learning |
| 2026-08-16.md | reinforcement-learning, post-training |
| 2026-08-20.md | post-training, ai-safety |
| 2026-09-05.md | systems, ai-safety |
| 2026-09-07.md | systems, ai-safety |
| 2026-09-24.md | ai-safety, post-training |
| 2026-09-25.md | ai-safety, multi-agent |
| 2026-09-26.md | ai-safety, multi-agent |
| 2026-09-28.md | ai-safety, multi-agent |
| 2026-09-29.md | ai-safety, multi-agent |
| 2026-09-30.md | reinforcement-learning, ai-safety |
| 2026-10-05.md | reinforcement-learning, ai-safety |
| 2026-10-12.md | ai-safety, post-training |

## Preservation and infrastructure

Migration replaced only the single `tags:` front-matter line. Byte comparison with the baseline verifies the rest of every log file is identical. Article Markdown and the About page remain byte-identical. Counts, filters, related entries and timelines use the new tags. `_data/log_topics.yml` centralizes canonical topics and legacy aliases; the browser receives that same taxonomy as JSON. Full-text search and calendar data formats remain unchanged.
