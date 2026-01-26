# Optimizing RAG for dense philosophical and scientific books

> **Implementation Status (Dec 23, 2024)**: References to "step-back prompting" in this document have been superseded. After analysis, step-back prompting (arXiv:2310.06117) was determined to be a **reasoning technique**, not RAG-specific. It has been replaced with **HyDE (Hypothetical Document Embeddings)** (arXiv:2212.10496) which is a proper RAG research technique. Query decomposition (arXiv:2507.00355) remains the primary strategy for complex queries.

Your retrieval scores of **~0.5** stem from a fundamental architectural mismatch: naive chunking destroys the conceptual coherence that makes philosophical and neuroscience texts meaningful. The solution requires rethinking the entire pipeline, from how you segment text to how you process queries about abstract concepts like consciousness and altruism. Recent advances in 2024-2025—particularly **RAPTOR's hierarchical summarization**, **Anthropic's contextual retrieval**, **graph RAG approaches**, and **HyDE/query decomposition**—directly address the challenge of retrieving information about concepts that span many pages rather than residing in discrete passages.

The most impactful changes for your use case are: replacing naive chunking with RAPTOR or parent-child retrieval to preserve conceptual relationships; adding hybrid retrieval (dense + BM25) with cross-encoder reranking, which Anthropic found reduces retrieval failures by **67%**; implementing step-back prompting for abstract queries, which Google DeepMind showed improves multi-hop reasoning by **27%**; and considering graph RAG for questions requiring synthesis across an entire book.

---

## Why naive chunking fails for conceptual content

Your current setup—800 tokens with 2-sentence overlap—treats books like databases of independent facts. But when a philosopher develops an argument about consciousness across 40 pages, or when a neuroscientist traces the evolutionary origins of addiction through multiple chapters, the conceptual unity exists at a level above individual chunks. Standard embedding models cannot capture this because each chunk is embedded in isolation, losing the referential context that makes philosophical arguments coherent.

Consider your example query: "What is consciousness and where is it located?" This question implicitly requires understanding multiple philosophical positions (Global Workspace Theory, Integrated Information Theory, phenomenological approaches), their neural correlates, and how different theorists relate subjective experience to brain mechanisms. No single 800-token chunk contains all this information, and cosine similarity between your short query and any individual chunk will be low because the query is conceptually broad while each chunk is specific.

The fundamental insight driving recent RAG improvements is that **retrieval must operate at multiple levels of abstraction simultaneously**. You need fine-grained chunks for precise matching, but you also need hierarchical summaries that capture thematic coherence, and you need mechanisms to expand short queries into retrieval-friendly forms.

---

## Chunking strategies that preserve conceptual relationships

**RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)**, presented at ICLR 2024, represents a paradigm shift for book-length content. Rather than chunking linearly, RAPTOR builds a tree structure by: segmenting text into 100-token leaf nodes, embedding them, clustering by semantic similarity (not adjacency), generating LLM summaries of each cluster, then recursively repeating to build a hierarchy. The critical innovation is that **discussions of "consciousness" scattered across chapters cluster into the same tree branch**, even if they appear 100 pages apart in the original text. On the QuALITY benchmark for long-document comprehension, RAPTOR achieved a **20% improvement** over standard retrieval methods.

**Parent-child retrieval** offers a simpler but highly effective approach. You maintain two chunk sizes: small children (200-400 tokens) for precise embedding matching, and large parents (1500-2000 tokens) that contain the full argumentative context. When a query matches a child chunk, you return the parent. LangChain's `ParentDocumentRetriever` and LlamaIndex's sentence window approach implement this pattern. For philosophical texts, configure parents as major sections or subsections—units where a complete argument or idea is developed.

**Late chunking**, introduced by Jina AI in September 2024, addresses the problem of lost referential context. Traditional chunking creates embeddings for isolated text fragments, so when a philosopher writes "This concept relates to the previous argument," the embedding doesn't know what "this" or "previous" refers to. Late chunking processes entire documents (up to 8K tokens) through the transformer first, creating context-aware token embeddings, then chunks and pools afterward. Each resulting chunk embedding "knows" about the full document context. This requires Jina's embeddings-v2/v3 models but significantly improves retrieval for referentially dense philosophical writing.

**Anthropic's contextual retrieval** (September 2024) takes a different approach: before embedding each chunk, prepend a brief contextual summary generated by an LLM. A chunk originally reading "The author argues against reductionist accounts" becomes "This passage from Chapter 5 discusses Chalmers' critique of physicalist theories of consciousness. The author argues against reductionist accounts..." Anthropic reported **35% reduction in retrieval failures** with contextual embeddings alone, **49%** when combined with contextual BM25, and **67%** with reranking added. At $1.02 per million tokens using prompt caching, this is cost-effective for book-length content.

**Agentic chunking** uses LLMs to determine chunk boundaries dynamically by understanding when one argument ends and another begins. An LLM can recognize that "the author is still developing the same thesis" even when vocabulary shifts, preserving the unity of extended philosophical arguments. This is computationally expensive but particularly effective for texts where semantic similarity doesn't cleanly correlate with argument boundaries.

For your specific case, the recommended chunking architecture combines these approaches: use RAPTOR to build hierarchical summaries that group related concepts across the book, implement parent-child retrieval for fine-grained matching with broad context, apply contextual enrichment to help chunks "remember" their position in the larger argument, and consider late chunking if you adopt Jina embeddings.

---

## Embedding models and hybrid retrieval transform matching quality

Your choice of embedding model matters enormously for conceptual content. The current state-of-the-art includes **Voyage-3.5** (outperforms OpenAI by 8.26%, supports 32K context), **BGE-M3** (open-source leader supporting dense, sparse, and ColBERT retrieval simultaneously), **NV-Embed-v2** (NVIDIA's model with novel latent-attention pooling, MTEB score of 72.31), and **Qwen3-Embedding-8B** (top performer on multilingual MTEB, Apache 2.0 licensed). For philosophical texts requiring nuanced semantic understanding, Voyage-3.5 and BGE-M3 are the strongest choices as of December 2025.

**Instruction-tuned embeddings** require proper prefixes for optimal performance. With E5 models, prepend `query:` to questions and `passage:` to documents. BGE models use `Represent this sentence for retrieval:` as a query prefix. For philosophical queries, you can craft domain-specific instructions: "Represent this question for retrieving philosophical arguments about consciousness: [your query]." This seemingly minor change can improve retrieval quality by 5-15%.

**Matryoshka embeddings** (supported by Voyage-3.5, OpenAI text-embedding-3, Nomic-embed) train models so that early dimensions capture the most semantic information. You can truncate 2048-dimensional embeddings to 256 dimensions with minimal quality loss—enabling fast initial search followed by full-dimension refinement. For large book collections, this enables cost-effective multi-stage retrieval.

**Hybrid retrieval combining dense embeddings with BM25 is non-negotiable for philosophical text**. Dense embeddings capture semantic similarity (understanding that "consciousness" relates to "awareness"), while BM25 catches exact philosophical terminology that must match precisely ("qualia," "phenomenal consciousness," "intentionality"). The standard weighting is 60-70% dense, 30-40% BM25, combined via reciprocal rank fusion. Weaviate, Pinecone, and Milvus offer native hybrid search; LangChain's `EnsembleRetriever` enables custom implementations.

**Cross-encoder reranking** provides the largest single improvement for retrieval quality. Unlike bi-encoders that embed queries and documents separately, cross-encoders process query-document pairs together through a transformer, enabling much deeper semantic matching. Retrieve an initial top-50 or top-100 candidates with your hybrid retriever, then rerank with a cross-encoder to select the final 5-10 passages. The best rerankers as of late 2025 include **mxbai-rerank-large-v2** (open-source state-of-the-art at 57.49 on BEIR), **Cohere Rerank 3** (production-ready API), **BGE-reranker-v2-m3** (open-source, fine-tunable), and **Jina-ColBERT-v2** (8192 token context, late-interaction architecture). Cross-encoder reranking typically improves precision by **20-35%**.

**ColBERT-style late interaction models** offer a middle ground between bi-encoders and cross-encoders. Instead of single vectors per document, ColBERT maintains token-level embeddings and uses a MaxSim operation to find the best token matches between query and document. This captures fine-grained semantic relationships particularly well for philosophical terminology that may have subtle contextual variations. Jina-ColBERT-v2 supports 89 languages and 8192-token contexts. The trade-off is 10-100x more storage than single-vector embeddings, making it better suited for reranking than initial retrieval.

---

## Query processing makes abstract questions retrievable

Abstract philosophical queries like "What is consciousness?" fail at retrieval because they're semantically distant from any specific passage. The query is conceptually broad; individual chunks are specific. Several techniques bridge this gap.

**Step-back prompting**, from Google DeepMind's ICLR 2024 paper, is essential for philosophical questions. Before retrieval, generate a higher-level "step-back question" that establishes the conceptual domain. For "What is consciousness and where is it located?", the step-back might be "What are the major theories of consciousness in philosophy of mind and neuroscience?" You retrieve for both the original and step-back questions, providing broader conceptual grounding. The paper showed **27% improvement on TimeQA** and **7% improvement on multi-hop reasoning tasks** with this technique.

**Query decomposition** breaks complex questions into retrievable sub-questions. "What is the philosophical and neuroscientific meaning of altruism?" becomes: (1) "Philosophical definitions of altruism," (2) "Evolutionary biology explanations of altruistic behavior," (3) "Neural mechanisms underlying altruism," (4) "Relationship between moral philosophy and neuroscience of prosocial behavior." Each sub-question retrieves different relevant passages; synthesis provides comprehensive coverage. Research from July 2025 showed query decomposition combined with reranking improves MRR@10 by **36.7%**.

**HyDE (Hypothetical Document Embeddings)** works particularly well for philosophical questions. Instead of embedding your short query, prompt an LLM to generate a hypothetical answer, then use that answer's embedding for retrieval. The insight: question and answer embeddings may be semantically distant, but two answers to the same question are similar. For "What is consciousness?", generate: "Consciousness refers to the subjective experience of awareness and perception. Major theories include Global Workspace Theory proposed by Baars, Integrated Information Theory by Tononi, and Higher-Order Theories..." Search for documents similar to this hypothetical answer rather than the bare question.

**Multi-query retrieval (RAG-Fusion)** generates multiple query variations from different perspectives, retrieves for each, then applies reciprocal rank fusion to merge results. For "What is the evolutionary meaning of addiction?", generated queries might include: "Evolutionary psychology of addictive behaviors," "Neurobiological basis of addiction from evolutionary perspective," "How did reward systems evolve to create vulnerability to addiction?", "Adaptive functions of compulsive behavior." Each variant retrieves different relevant passages; RRF combines rankings robustly without requiring careful tuning.

The recommended query processing pipeline for your philosophical questions: (1) Apply step-back prompting to establish conceptual domain, (2) Decompose into sub-questions if the query has multiple facets, (3) Generate multi-query variations for each sub-question, (4) Optionally apply HyDE to one or more queries, (5) Retrieve using hybrid search for each processed query, (6) Combine via reciprocal rank fusion.

---

## Graph RAG enables reasoning about concepts across entire books

Standard vector retrieval treats documents as bags of chunks. For philosophical texts where concepts develop, relate to each other, and build arguments across chapters, graph-based approaches capture structural relationships that vector similarity cannot.

**Microsoft GraphRAG** (April 2024) builds a knowledge graph from extracted entities and relationships, then applies community detection (Leiden algorithm) to cluster related concepts. It generates natural language summaries for each community at multiple levels of abstraction. For a query like "What are the main themes across this book?", GraphRAG's global search retrieves community summaries and synthesizes them—something standard RAG cannot do at all. On comprehensiveness benchmarks, GraphRAG achieved **70-80% win rate** against baseline RAG.

The architecture has two search modes: **global search** uses community summaries for holistic questions, while **local search** fans out from specific entities to neighbors and associated concepts. For philosophical texts, this means you can ask both "How does Dennett define consciousness?" (local search) and "What are all the theories of consciousness discussed in this book?" (global search leveraging community summaries).

**LightRAG** (October 2024, EMNLP 2025) provides a simpler alternative with lower indexing overhead and crucially, support for incremental updates. While GraphRAG requires full re-indexing for new content, LightRAG can incorporate new chapters without rebuilding the entire graph. It uses dual-level retrieval: low-level for specific entities, high-level for broader themes, merged at query time.

For your neuroscience and philosophy books, GraphRAG is particularly valuable for questions about how concepts relate across the book. Consider the question "What is the relationship between consciousness and altruism?" Neither concept might appear in the same passage, but a graph representation captures how each relates to shared neighbors like "empathy," "theory of mind," or "neural correlates of moral reasoning."

**When to use Graph RAG versus Vector RAG**: Use vector search for questions about specific passages or localized content ("What does Chapter 3 say about memory?"). Use graph search for questions requiring synthesis across the book or about conceptual relationships ("How does the author's view of consciousness evolve throughout the book?"). The FalkorDB/Diffbot benchmark found GraphRAG achieved **3.4x accuracy improvement** over vector RAG for questions requiring relationship understanding, while vector RAG scored **0%** on questions about hierarchical or structural relationships.

Implementation complexity is higher for graph approaches. For a single book, parent-child retrieval with RAPTOR hierarchies may be sufficient. For a corpus of philosophical works where you want cross-book concept retrieval, invest in GraphRAG or a Neo4j-based hybrid architecture combining graph traversal with vector search.

---

## Post-retrieval optimization compounds improvements

Even with excellent retrieval, several post-processing steps further improve generation quality.

**Lost-in-the-middle mitigation** addresses a documented LLM attention pattern: models attend more strongly to tokens at the beginning and end of context, often ignoring middle content. Stanford research found performance degrades **30%+** when relevant information appears in the middle of the context window. The solution is strategic document ordering: place your most relevant retrieved passages at positions 1-2 and at the end of the context, with less critical passages in the middle. This simple reordering can improve answer quality by **15%** on question-answering benchmarks.

**Context compression** reduces costs and may improve generation by removing irrelevant content. **LongLLMLingua** (Microsoft) uses contrastive perplexity to identify query-relevant content, achieving **21.4% accuracy improvement** with only 25% of original tokens. **RECOMP** (ICLR 2024) offers both extractive compression (sentence selection) and abstractive compression (summary generation). For philosophical texts with dense argumentation, extractive compression preserves the author's precise formulations better than abstractive approaches.

**Voyage-context-3** (released 2025) automatically captures document-level context in chunk embeddings, reducing the need for explicit contextual enrichment. Voyage reports it outperforms separate contextual retrieval methods by **6.76%** and serves as a drop-in replacement for standard embedding calls.

---

## Advanced architectures adapt to query complexity

Beyond pipeline optimizations, several 2024-2025 papers propose fundamentally new RAG architectures suited for complex reasoning tasks.

**Self-RAG** (ICLR 2024 Oral, top 1% of submissions) trains a single language model to generate special reflection tokens that control retrieval and critique generation. The model learns to output `[Retrieve]` only when external knowledge is needed, `[IsRel]` to assess document relevance, `[IsSup]` to verify factual support, and `[IsUseful]` to rate overall quality. For philosophical questions requiring both factual grounding and conceptual synthesis, Self-RAG can decide when to retrieve scientific facts versus when to reason from retrieved philosophical principles. The model and code are available at the project repository.

**Corrective RAG (CRAG)** adds a retrieval evaluator that scores document relevance, triggering three pathways: if confidence is high, use refined documents; if low, discard retrieval and fall back to web search; if ambiguous, combine both sources. For questions about current neuroscience ("What does current research say about the neural correlates of consciousness?"), CRAG's web search fallback can supplement book content with recent findings. LangGraph provides a tutorial implementation.

**Adaptive RAG** (NAACL 2024) uses a classifier to route queries by complexity. Simple definitional questions ("What is qualia?") may not need retrieval at all—the LLM's parametric knowledge suffices. Moderate questions need single-step retrieval. Complex multi-hop questions ("Compare Chalmers' and Dennett's views on qualia and their implications for neural correlates") require iterative retrieval with intermediate reasoning. Training a classifier (typically FLAN-T5) on your query types enables optimal resource allocation—avoiding unnecessary retrieval for simple questions while applying full multi-hop reasoning for complex ones.

**Agentic RAG** incorporates autonomous agents that can plan multi-step retrieval, select among multiple knowledge sources, and self-correct based on intermediate results. Frameworks like LangGraph, CrewAI, and LlamaIndex's agent abstractions enable building agents that route philosophical questions to appropriate sections, retrieve iteratively until sufficient context is gathered, and validate answers against retrieved evidence. For a philosophical book corpus, you might configure specialized agents: a concepts agent for definitions and explanations, an arguments agent for claims and premises, and a cross-reference agent for relationships between chapters or books.

**DSPy** from Stanford NLP treats prompts as learnable parameters rather than hand-crafted text. You define signatures (input/output specifications) and modules (prompting strategies), then use optimizers like `MIPROv2` or `BootstrapFewShot` to automatically tune prompts for your specific domain. For philosophical texts with specialized vocabulary, DSPy optimization can learn effective few-shot examples and instruction phrasings that improve retrieval and generation quality.

---

## Evaluation must capture domain-specific requirements

Standard cosine similarity scores are insufficient for evaluating philosophical RAG systems. You need metrics that capture whether retrieved passages actually contain the relevant philosophical arguments, whether generated answers faithfully represent source positions, and whether the system handles the unique challenges of abstract conceptual content.

**RAGAS** provides reference-free evaluation using LLM judges: Context Precision (signal-to-noise of retrieved content), Context Recall (whether all relevant information was retrieved), Faithfulness (factual consistency with retrieved context), and Answer Relevancy (whether the response addresses the question). RAGAS works well for iterative development but can be opaque about failure modes.

**ARES** (Stanford, NAACL 2024) fine-tunes lightweight judge models on synthetic training data and uses Prediction-Powered Inference for statistical confidence intervals. It showed **59.3% improvement** over RAGAS on context relevance accuracy and handles domain shifts better—important if you're evaluating on philosophical content not seen during training.

**DeepEval's GEval** enables custom criteria in natural language. For philosophical texts, define metrics like: "Evaluate whether the response correctly represents the original philosopher's position without strawmanning" or "Assess whether all relevant premises are retrieved to support the identified claim." This captures domain-specific quality requirements that generic metrics miss.

For your conceptual questions, create an evaluation dataset that includes: factual questions with clear answers, interpretive questions about philosophical positions, comparative questions requiring synthesis, and questions about how concepts relate across the book. Measure not just retrieval metrics (Recall@K, NDCG) but also argument completeness (are all relevant premises retrieved for a claim?) and definitional consistency (does the answer use terms as defined in the source?).

---

## Domain-specific techniques for philosophical and scientific texts

**Handling polysemy in philosophical terminology** requires awareness that "consciousness" in phenomenology differs from "consciousness" in cognitive neuroscience. Solutions include sense-aware embeddings that create separate vectors for each meaning, ontology-based query expansion that adds domain-specific context, and explicit sense disambiguation in query processing. Build a domain glossary mapping terms to their senses, then use LLM classification to determine which sense a query intends.

**Tracking definitions that evolve through a book** is essential when an author refines their position across chapters. Index explicit definitions with page/chapter metadata and retrieve relevant definitions as background context. When a query asks about "consciousness" in the context of Chapter 10, include Chapter 1's foundational definition and note any refinements.

**Argument mining techniques** can preprocess philosophical texts to identify claims, premises, and argumentative relationships. Tag chunks as "claim," "supporting premise," "objection," or "response to objection." When retrieving for questions about an argument, fetch the full argument structure rather than isolated passages. This preserves the logical flow that makes philosophical arguments meaningful.

**Thematic coherence via topic modeling** (BERTopic achieves ~0.57 coherence score, 18% better than LDA) can identify chapter-level themes. Use these themes to boost retrieved passages from thematically relevant sections. When a query clearly relates to "theories of mind," passages from the chapters discussing various theories get ranked higher than tangentially related passages from methodology sections.

---

## Practical implementation priority for your system

Given your current setup and the challenges you've described, prioritize changes in this order:

**Immediate impact (implement first)**: Switch to hybrid retrieval (dense + BM25) with a cross-encoder reranker. This alone should substantially improve your retrieval scores. Use BGE-M3 for embeddings (supports hybrid natively) and BGE-reranker-v2-m3 for reranking. Increase chunk size to 1200-1500 tokens with 200-token overlap, and add chapter/section metadata to each chunk.

**High impact (implement second)**: Add step-back prompting and query decomposition for your abstract questions. These techniques directly address why "What is consciousness?" retrieves poorly. Implement parent-child retrieval to match on small chunks but return larger argumentative context. Apply Anthropic's contextual retrieval pattern—prepend contextual summaries to chunks before embedding.

**Substantial architectural improvement (implement third)**: Build RAPTOR hierarchies that cluster related concepts regardless of their position in the book. This transforms retrieval for thematic questions spanning multiple chapters. If you have multiple books, consider implementing LightRAG for cross-book conceptual search.

**Advanced optimization (implement as needed)**: Fine-tune embedding models on your philosophical/neuroscience corpus if base models still struggle with domain terminology. Implement Adaptive RAG to route queries by complexity. Explore agentic architectures if you need multi-source retrieval or iterative reasoning for particularly complex questions.

---

## Conclusion

The path from 0.5 similarity scores to effective retrieval for philosophical and neuroscience books requires abandoning the assumption that chunks are independent. Every technique that performs well on conceptual content—RAPTOR's hierarchical clustering, parent-child retrieval, contextual enrichment, graph RAG, step-back prompting—succeeds by preserving or reconstructing the relationships between ideas that naive chunking destroys.

The most underappreciated insight from recent research is that **query processing matters as much as indexing for abstract questions**. Step-back prompting and query decomposition transform vague philosophical questions into retrieval-friendly forms, while HyDE and multi-query approaches bridge the semantic gap between short questions and long document passages. These techniques stack multiplicatively with better chunking and hybrid retrieval.

For your specific domain, the combination of RAPTOR hierarchies (grouping concepts by meaning rather than position), contextual retrieval (preserving document-level context in chunks), hybrid search with reranking (capturing both semantic and lexical matches), and step-back prompting (handling abstract queries) provides a comprehensive solution. The expected improvement from implementing these techniques together is substantial—Anthropic documented 67% reduction in retrieval failures; RAPTOR showed 20% improvement on long-document comprehension; step-back prompting demonstrated 27% gains on multi-hop reasoning.

The key novel insight is that philosophical and neuroscience books operate at multiple levels of abstraction simultaneously—specific claims, supporting arguments, chapter-level themes, and book-level theses. Effective RAG for this content must retrieve at the appropriate level for each query type, which requires hierarchical indexing (RAPTOR or graph RAG), adaptive query processing (complexity routing), and query transformation (step-back prompting) working together. No single technique solves the problem; the solution is an integrated architecture that matches the multi-level structure of the content itself.

---

## Implementation Notes (Historical)

### Deprioritized Techniques

**Lost-in-the-Middle Mitigation** (Dec 22, 2024): Removed from active implementation plan. The technique (reordering chunks to place best content at start/end of context) was considered but deprioritized in favor of:
- Alpha tuning experiments (quick win, low effort)
- RAPTOR hierarchical summarization (higher impact for comprehension)
- GraphRAG (highest impact for comprehensiveness)

The research remains documented above (see "Post-retrieval optimization" section) for future reference if needed.