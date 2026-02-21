# Contextual Chunking

[← Semantic Chunking](semantic-chunking.md) | [Home](../../README.md)

Contextual chunking prepends LLM-generated snippets that situate each chunk within the document, improving embedding quality by disambiguating entities and connecting isolated content to its broader context.

While section and semantic chunking optimize *where* to split text, contextual chunking addresses *what information is lost* after splitting—the document-level context that makes chunks meaningful.

Here [Anthropic Blog: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) approach is implemented as a **post-processing step on semantic chunks (std=2)**, using these parameters:
- **Max snippet tokens**: 150, brief disambiguation with buffer for complete sentences
- **Model**: deepseek-v3.2, cost-efficient for simple contextualization



## Anthropic's Implementation


Anthropic's research quantified how much document-level context is lost during chunking:

<div align="center">

| Approach | Retrieval Failure Rate | Improvement |
|----------|----------------------|-------------|
| Standard chunking | Baseline | — |
| Contextual chunking | **-35%** | Top-20 retrieval |
| + BM25 hybrid + reranking | **-67%** | Combined approach |

</div>

**Key insight:** The embedding model encodes *what words are in the chunk* but not *what the chunk is about*. Adding a contextual snippet that explicitly states the chunk's semantic role bridges this gap.


Anthropic passes the **entire document** to the LLM for each chunk, using [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) to avoid re-processing the document for every chunk. The document is cached after the first chunk, making subsequent calls ~90% cheaper.

**Anthropic's exact prompt:**

```
<document>
{{WHOLE_DOCUMENT}}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>

Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

**Implementation details** (from [LlamaIndex's cookbook](https://developers.llamaindex.ai/python/examples/cookbooks/contextual_retrieval/)):
- **Model**: Claude 3 Haiku (fast, cheap)
- **Output**: 50-100 tokens per chunk
- **Cost with caching**: ~$1 per million document tokens
- **Document size assumption**: < 200K tokens (~500 pages)

The generated context is prepended directly to the chunk text before embedding.




## Differences from Anthropic's Approach

Anthropic tested on short documents (papers, articles) where the full document fits in context. This corpus contains **books** (300-800 pages, 150K-400K tokens) where full-document context is impractical:

1. **Cost**: A 600-page book × 700 chunks = massive token usage even with caching
2. **Noise**: 99% of a textbook is irrelevant to any given chunk (other chapters)
3. **Structure**: Books have deep hierarchy (chapters → sections) that papers lack

<div align="center">

| Aspect | Anthropic | This Implementation |
|--------|-----------|---------------------|
| **Document context** | Full document (~8K-50K tokens) | Section title (~10 tokens) |
| **Document size** | Papers, articles (<500 pages) | Books (300-800 pages) |
| **Context source** | Full document with prompt caching | **Book title** + section title |
| **Original preservation** | Not mentioned | Stores `original_text` and `contextual_snippet` separately |

</div>

**Trade-off**: This implementation uses **section title** instead of full document context. Section titles often contain the exact disambiguation terms needed (e.g., "Ventral Striatum: Pleasure and Reward") at a fraction of the token cost. The LLM's job is to connect the chunk's content to the section title's concepts.

**This implementation's prompt:**

```
<book>
{book_title}
</book>

<section>
{section_title}
</section>

<chunk>
{chunk_text}
</chunk>

Write a brief context (1-2 sentences, ~50-80 words) to situate this chunk within the book for search retrieval.
Use key terms from the section title. Ensure sentences are complete - do not end mid-sentence.
Answer only with the context, nothing else.
```

Key differences from Anthropic's prompt:
- **`<book>`** instead of `<document>`: Uses the **book title** (e.g., "Behave, The Biology of Humans at Our Best Worst (Robert M. Sapolsky)") which the LLM may recognize, providing implicit context about the work's themes and domain
- **`<section>`** instead of full document: Just the section title (~10 tokens vs ~200K tokens for a book)
- **"Use key terms from the section title"**: Explicit instruction to leverage section title terminology for disambiguation



## Algorithm

The input is semantic chunks (std=2) from `semantic_std2/{book}.json`. Semantic chunks are used instead of section chunks because they split at topic boundaries rather than token limits, avoiding orphan chunks and preserving argument coherence.

```
For each document:
  Load existing chunks

  For each chunk:
    1. Get book title and section title from chunk metadata
    2. Build prompt with: book_title, section_title, chunk_text
    3. Call LLM: "Situate this chunk using section title key terms"
    4. Prepend snippet: "[{snippet}] {original_text}"
    5. Re-compute token count
    6. Save with original_text preserved
```



## Example: Semantic vs Contextual Chunking

The same content handled by each strategy (same chunk as in [section-chunking.md](section-chunking.md)):

<details>
<summary><strong>Semantic Chunking (std=2): 649 tokens</strong></summary>
<small>

**Original chunk** — No document-level context:
```json
{
  "chunk_id": "Brain and behavior...::chunk_647",
  "context": "Brain and behavior... > CHAPTER 13 Emotions > Ventral Striatum: Pleasure and Reward",
  "section": "Ventral Striatum: Pleasure and Reward",
  "text": "In 1954, at McGill University in Montreal, Canada, the psychologists James Olds and Peter Milner implanted a pair of electrodes in the brain of a rat, hoping to study the effects of stimulation on its movements. However, the results were unexpected: the rat began returning again and again to the place in the cage where it received stimulation, as if strongly rewarded for doing so (Olds & Milner, 1954). Surprised to see this effect, Olds and Milner then tried providing the rat with a lever that would trigger stimulation. The rat soon began pressing this lever repeatedly, hundreds of times an hour, often to the exclusion of all other activities. The effects of the stimulation bore all the behavioral hallmarks of intense reward. X-rays and postmortem examinations eventually revealed that the electrode had missed its intended target and instead had reached a region known as the septal area, near the ventral striatum. In a series of experiments and later in televised demonstrations, Olds and Milner showed rats braving severe electric shocks to obtain stimulation and engaging in self-stimulation so fervently as to reach the point of starvation. As a result, this region, and its nearby connections through the medial forebrain bundle, soon became popularized as the so-called 'pleasure center of the brain' (Olds & Milner, 1954). Over the next two decades, studies provided evidence that these same regions have a similar function in human beings who underwent neurosurgical implantation of DBS electrodes for the treatment of psychiatric and neurological illnesses...",
  "token_count": 649,
  "chunking_strategy": "semantic_std2"
}
```

</small>
</details>

<details>
<summary><strong>Contextual Chunking: 709 tokens (+60 from snippet)</strong></summary>
<small>

**Enriched chunk** — LLM-generated context from book and section title:
```json
{
  "chunk_id": "Brain and behavior...::chunk_647",
  "context": "Brain and behavior... > CHAPTER 13 Emotions > Ventral Striatum: Pleasure and Reward",
  "section": "Ventral Striatum: Pleasure and Reward",
  "text": "[This chunk discusses the role of the ventral striatum in pleasure and reward, highlighting foundational experiments by Olds and Milner, the implications for human neuroimaging studies, and the potential therapeutic applications of deep brain stimulation (DBS) in treating severe depression and restoring the capacity for pleasure.] In 1954, at McGill University in Montreal, Canada, the psychologists James Olds and Peter Milner implanted a pair of electrodes in the brain of a rat...",
  "token_count": 709,
  "chunking_strategy": "contextual",
  "original_text": "In 1954, at McGill University in Montreal, Canada, the psychologists James Olds and Peter Milner implanted a pair of electrodes in the brain of a rat...",
  "contextual_snippet": "This chunk discusses the role of the ventral striatum in pleasure and reward, highlighting foundational experiments by Olds and Milner, the implications for human neuroimaging studies, and the potential therapeutic applications of deep brain stimulation (DBS) in treating severe depression and restoring the capacity for pleasure."
}
```

**Context provided to LLM:**
- **Book**: "Brain and behavior, a cognitive neuroscience perspective (David Eagleman, Jonathan Downar)"
- **Section**: "Ventral Striatum: Pleasure and Reward"

</small>
</details>


**Key difference:** The snippet explicitly names "ventral striatum," "pleasure and reward," "Olds and Milner," "deep brain stimulation (DBS)," and "treating severe depression"—terms derived from the section title and the chunk's content that the embedding model can now use for disambiguation. A query about "brain reward mechanisms" or "DBS for depression" will match this chunk more precisely than the original, which never explicitly states its topic.



## Navigation

**Next:** [RAPTOR](raptor.md) — Hierarchical summarization tree

**Related:**
- [Semantic Chunking](semantic-chunking.md) — Prerequisite (contextual builds on semantic std=2 chunks)
- [Section Chunking](section-chunking.md) — Token-based baseline alternative
- [Chunking Overview](README.md) — Strategy comparison
