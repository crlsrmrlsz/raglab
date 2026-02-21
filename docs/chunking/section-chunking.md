# Section Chunking (Baseline)

[← Chunking Overview](README.md) | [Home](../../README.md)

This is the baseline chunking strategy, splitting by token number but leveraging the structure authors have already created. It operates on a key assumption: **each section contains a single coherent subject**. By respecting this natural boundary, the chunker splits documents into chunks with a maximum 800-token size while maintaining sentence overlap for context continuity.

The algorithm uses these parameters:

- **Chunk size**:  800 tokens (max).  Upper limit balancing paragraph unity and retrieval performance
- **Overlap**:  2 sentences. Handles "As mentioned above..." references with minimal redundancy (~50-100 tokens)

### Chunk Size: Why 800-Token Limit?

Research indicates 512-1024 tokens is optimal for technical/analytical content. [NVIDIA's benchmark](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/) found this range best for complex queries, while [academic studies](https://arxiv.org/html/2505.21700v2) show smaller chunks (64-128) suit factoid queries but larger chunks dramatically improve technical retrieval (TechQA: 4.8% → 71.5% accuracy from 64 to 1024 tokens).

Analysis of this corpus:

![Chunk Size Distribution](../../assets/section-chunking-distribution.png)

The distribution reveals a **bimodal pattern** characteristic of fixed-limit chunking. Neuroscience shows a relatively uniform spread across 50-700 tokens (sections naturally fitting the limit) with a sharp spike at 800 tokens where the chunker forces splits. Philosophy exhibits an even more pronounced spike at exactly 800 tokens—longer essay sections repeatedly hit the ceiling, creating many maximum-size chunks. The mean-median gap (Neuroscience: 482 vs 492; Philosophy: 589 vs 746) indicates left-skew in neuroscience (many small sections) and right-skew in philosophy (chunks clustering at the limit). This "ceiling effect" is the key limitation of token-based chunking: it splits mid-argument regardless of semantic coherence.

<div align="center">

| Corpus | Avg Section Tokens | Single-Chunk Sections |
|--------|-------------------|----------------------|
| **Neuroscience** | 722 | 2,270 / 3,219 (71%) |
| **Philosophy** | 1,549 | 351 / 545 (64%) |

</div>

Neuroscience sections average 722 tokens—below the limit, preserving most conceptual units intact. Philosophy varies widely: aphoristic works (Tao Te Ching: 159, Art of Living: 238) fit single chunks, while essay collections (Seneca: 2,127, Schopenhauer: 2,300+) require splitting.

The 800-token limit balances these needs: within the research-backed 512-1024 optimal range, preserves paragraph unity (one idea per chunk), and keeps most neuroscience sections whole. For longer philosophy essays, 2-sentence overlap maintains continuity, though [Contextual Chunking](contextual-chunking.md) or [RAPTOR](raptor.md) may better handle extended arguments.

**Future work:** per-content-type tuning (shorter for factoid references, longer for essays).

### Algorithm

The input is a JSON file per book with NLP generated chunks, each one containing one paragraph and with the context included (book > section)
```
For each document:
  1. Load NLP-segmented paragraphs
  2. Initialize: current_chunk = [], current_context = None

  For each paragraph:
    If context changed (new section):
      1. Save current_chunk
      2. Start new chunk (no overlap across sections)

    For each sentence:
      If (current_chunk + sentence) <= MAX_TOKENS:
        Append sentence to chunk
      Else:
        1. Save current_chunk
        2. Start new chunk with last 2 sentences (overlap)
```




<details>
<summary><strong>Example Chunk</strong></summary>
<small>

```json
{
  "chunk_id": "Brain and behavior, a cognitive neuroscience perspective (David Eagleman, Jonathan Downar)::chunk_578",
  "book_id": "Brain and behavior, a cognitive neuroscience perspective (David Eagleman, Jonathan Downar)",
  "context": "Brain and behavior... > CHAPTER 14 Motivation and Reward > Why Motivation Matters",
  "section": "Why Motivation Matters",
  "text": "Staying alive is a balancing act. From the moment an animal opens its eyes in the morning, it is faced with a series of dilemmas. Should I spend my time foraging for food and building my energy supplies? Or is it more important to find a source of water? Is it too cold to go outside today, even if I am hungry? Am I safe from predators here, or do I need to find a better shelter? [...] It is worth mentioning here that the circuitry of what we usually mean by 'intelligence' is a relatively new addition to an ancient basic brain plan. [...] Motivation is more akin to judgment: the ability to make accurate predictions about what is most important in any given scenario. [...] What are these basic needs or drives? We can divide them into the drives arising from internal states or bodily functions and the drives arising from external sources or incentives. The internal drives include homeostatic drives such as energy balance, water balance, thermoregulation, circadian rhythms including sleep and wakefulness, and stress responses, as well as internal drives toward reproductive and defensive behavior. These drives are closely associated with the circuitry of the hypothalamus, as we'll see further below. External drives arise from sources outside the body itself. These include drives in response to threats, sexual or reproductive opportunities, parental attachment drives, social dominance, and affiliation.",
  "token_count": 782,
  "chunking_strategy": "section"
}
```

</small>
</details>

## Navigation

**Next:** [Semantic Chunking](semantic-chunking.md) — Embedding-based topic boundaries

**Related:**
- [Contextual Chunking](contextual-chunking.md) — LLM-generated context prepended
- [RAPTOR](raptor.md) — Hierarchical summarization alternative
- [Chunking Overview](README.md) — Strategy comparison
