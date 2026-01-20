# Contextual Chunking

[← Semantic Chunking](semantic-chunking.md) | [Home](../../README.md)

Contextual chunking prepends LLM-generated snippets that situate each chunk within the document, improving embedding quality by disambiguating entities and connecting isolated content to its broader context.

While section and semantic chunking optimize *where* to split text, contextual chunking addresses *what information is lost* after splitting—the document-level context that makes chunks meaningful.

Here [Anthropic Blog: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) approach is implemented as a **post-processing step on section chunks**, using these parameters:
- **Neighbor chunks**: 2 before + 2 after for local context
- **Max snippet tokens**: 100, brief disambiguation not a summary
- **Model**: gpt-4o-mini, cost-efficient for simple contextualization



## Research Background

Anthropic's research quantified how much document-level context is lost during chunking:

<div align="center">

| Approach | Retrieval Failure Rate | Improvement |
|----------|----------------------|-------------|
| Standard chunking | Baseline | — |
| Contextual chunking | **-35%** | Top-20 retrieval |
| + BM25 hybrid + reranking | **-67%** | Combined approach |

</div>

**Key insight:** The embedding model encodes *what words are in the chunk* but not *what the chunk is about*. Adding a contextual snippet that explicitly states the chunk's semantic role bridges this gap.




## Differences from Anthropic's Approach

<div align="center">

| Aspect | Anthropic | This Implementation |
|--------|-----------|---------------------|
| **Document context** | Full document in prompt | 2 neighbor chunks (token efficiency) |
| **Section metadata** | Not mentioned | Includes hierarchical path (Book > Chapter > Section) |
| **Original preservation** | Not mentioned | Stores `original_text` and `contextual_snippet` separately |

</div>



## Algorithm

The input is section chunks from `section/{book}.json`. The algorithm enriches each chunk with LLM-generated context.

```
For each document:
  Load existing chunks from section/ folder

  For each chunk:
    Gather neighboring chunks (2 before + 2 after)
    Build prompt with: book_name, section_path, neighbors, chunk_text
    Call LLM: "Give 2-3 sentences situating this chunk"
    Prepend snippet: "[{snippet}] {original_text}"
    Re-compute token count
    Save with original_text preserved
```



## Example: Section vs Contextual Chunking

The same content handled by each strategy:

<details>
<summary><strong>Section Chunking: 672 tokens</strong></summary>
<small>

**Original chunk** — No document-level context:
```json
{
  "chunk_id": "Brain and behavior...::chunk_549",
  "context": "Brain and behavior... > CHAPTER 13 Emotions > Ventral Striatum: Pleasure and Reward",
  "text": "In 1954, at McGill University in Montreal, Canada, the psychologists James Olds and Peter Milner implanted a pair of electrodes in the brain of a rat, hoping to study the effects of stimulation on its movements. However, the results were unexpected: the rat began returning again and again to the place in the cage where it received stimulation, as if strongly rewarded for doing so (Olds & Milner, 1954). Surprised to see this effect, Olds and Milner then tried providing the rat with a lever that would trigger stimulation. The rat soon began pressing this lever repeatedly, hundreds of times an hour, often to the exclusion of all other activities. The effects of the stimulation bore all the behavioral hallmarks of intense reward. X-rays and postmortem examinations eventually revealed that the electrode had missed its intended target and instead had reached a region known as the septal area, near the ventral striatum. In a series of experiments and later in televised demonstrations, Olds and Milner showed rats braving severe electric shocks to obtain stimulation and engaging in self-stimulation so fervently as to reach the point of starvation. As a result, this region, and its nearby connections through the medial forebrain bundle, soon became popularized as the so-called 'pleasure center of the brain' (Olds & Milner, 1954). Over the next two decades, studies provided evidence that these same regions have a similar function in human beings who underwent neurosurgical implantation of DBS electrodes for the treatment of psychiatric and neurological illnesses...",
  "token_count": 672,
  "chunking_strategy": "sequential_overlap_2"
}
```

</small>
</details>

<details>
<summary><strong>Contextual Chunking: 759 tokens (+87 from snippet)</strong></summary>
<small>

**Enriched chunk** — LLM-generated context prepended:
```json
{
  "chunk_id": "Brain and behavior...::chunk_549",
  "context": "Brain and behavior... > CHAPTER 13 Emotions > Ventral Striatum: Pleasure and Reward",
  "text": "[This chunk provides background on the discovery of the ventral striatum's role in pleasure and reward, which is a key focus of this chapter on the neuroscience of emotions. It describes early experiments on electrical stimulation of the ventral striatum in rats and how this led to the identification of this region as the \"pleasure center of the brain\", a finding that has also been observed in humans undergoing deep brain stimulation for psychiatric disorders.] In 1954, at McGill University in Montreal, Canada, the psychologists James Olds and Peter Milner implanted a pair of electrodes in the brain of a rat...",
  "token_count": 759,
  "chunking_strategy": "contextual",
  "original_text": "In 1954, at McGill University in Montreal...",
  "contextual_snippet": "This chunk provides background on the discovery of the ventral striatum's role in pleasure and reward..."
}
```

</small>
</details>


**Key difference:** The snippet explicitly names "ventral striatum," "pleasure and reward," "neuroscience of emotions," and "deep brain stimulation"—terms the embedding model can now use for disambiguation. A query about "brain reward mechanisms" will match this chunk more precisely than the original, which never explicitly states its topic.



## Navigation

**Next:** [RAPTOR](raptor.md) — Hierarchical summarization tree

**Related:**
- [Section Chunking](section-chunking.md) — Prerequisite (contextual builds on section chunks)
- [Semantic Chunking](semantic-chunking.md) — Embedding-based boundaries alternative
- [Chunking Overview](README.md) — Strategy comparison
