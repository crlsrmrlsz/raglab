# Content Preparation

[← Getting Started](../getting-started.md) | [Home](../../README.md)

This document covers the complete workflow to extract content from PDF books into clean markdown files ready for chunking. Content preparation follows three phases:

```
Phase 1: PDF Pre-Cleaning (manual)
    ↓
Phase 2: PDF to Markdown (Docling)
    ↓
Phase 3: Markdown Cleaning (automated + manual review)
    ↓
Phase 4: NLP Segmentation
```

Each phase addresses specific challenges encountered with complex academic texts, particularly neuroscience books with dense layouts. Philosophy books have a more regular structure in one column and chapters are easier to process.



## Phase 1: PDF Pre-Cleaning

Before any automated extraction, PDFs are manually cleaned using PDF editing tools ([PDF24](https://www.pdf24.org/)) to remove the pages that contain these elements:


<div align="center">

| Element | Why Remove |
|---------|------------|
| References/Bibliography | Dense citation blocks confuse layout detection |
| Glossary | Alphabetical lists extract poorly |
| Index | Page number lists are noise |
| Acknowledgments | Not core content |
| Notes sections | Often formatted as footnotes, sometimes at the end of chapters or end of book |
| Appendices | Supplementary material, separate handling needed |

</div>

This may seem unnecessary, but after facing the difficulties converting and cleaning downstream, I decided to simplify things from the start. 

During conversion not every section was correctly identified, sometimes a heading appeared in the middle of a paragraph or depending on book layout, some sections were not always detected, or had random errors, so cleaning the unwanted sections (complete pages) in advance was easier for me and ensures better quality in the data for next phases, although this obviously **won't scale for a bigger corpus**.

With expected future improvements in PDF text extraction models, this cleaning could be done during text extraction itself or afterwards over a properly structured markdown, removing unnecessary sections.



## Phase 2: PDF to Markdown Conversion

This phase took considerably more effort than expected.

First attempts used [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/), a lightweight PDF-to-markdown converter. For scientific books, however, it does not work well with multicolumn layouts, especially when images appear mid-paragraph. The [PyMuPDF-Layout](https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html) extension adds AI-based layout detection, which improves results significantly but still not enough. 


### Solution: Docling

[Docling](https://github.com/docling-project/docling) (IBM Research) uses AI vision models for layout understanding, solving many of the problems PyMuPDF4LLM couldn't handle. I didn't need tables or images, which are possibly the most difficult elements to extract, so I couldn't test it thoroughly but it worked quite well.

Most of the errors I got appeared with multicolumn layouts, in some specific pages like this one where images were mixed in the middle of a paragraph or there were two different column layouts on the same page. In these cases columns were mixed randomly and needed manual correction.

<div align="center">
  <img src="../../assets/page_columns.png" alt="Multi column page">
</div>




### Implementation

Docling identifies some elements semantically, like captions, tables, figures, headers or footers. It also allows you to directly remove them, so that simplifies the next cleaning phase.

Figures and tables were not extracted, as the complexity of extracting and parsing them correctly didn't pay off for the purpose of the project.

```python
# src/content_preparation/extraction.py

from docling.datamodel.document import InputFormat, DocItemLabel
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

def extract_pdf(pdf_path: Path) -> str:
    """Extract text from PDF to markdown."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False           # Skip OCR (text PDFs only)
    pipeline_options.do_table_structure = False  # Skip table parsing

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path)
    doc = result.document

    # Remove artifacts using native labels
    labels_to_remove = {
        DocItemLabel.CAPTION,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.TABLE
    }

    items_to_remove = [
        item for item, level in doc.iterate_items()
        if hasattr(item, "label") and item.label in labels_to_remove
    ]
    doc.delete_items(node_items=items_to_remove)

    # Also remove pictures and their children (diagram labels, etc.)
    # ... (see full implementation in extraction.py)

    return doc.export_to_markdown()
```


## Phase 3: Markdown Cleaning

After Docling extraction, there were still some errors, and perhaps too obsessed with the quality of the data, I performed two cleaning steps to refine the output.

I am not sure if it is worth the effort to get perfect data, each case would need to consider effort vs quality. Using 800 token chunks, the errors would affect about 3-5% of the chunks. That does not seem too much but those are non-recoverable concepts that will accumulate to the losses in next phases. Perhaps LLM could still extract some information from disordered text, and concepts will appear several times along the corpus, but with such specific knowledge I preferred to avoid as many errors as possible.

This phase outputs **clean and structured markdown divided in sections (headings)**. This structure will be leveraged downstream to avoid mixing content of different sections in chunking.

### Manual Review (Optional)

Location: `data/processed/02_manual_review/`

Purpose: Catch extraction errors before automated cleaning:
- Solve multicolumn errors (more than 40 pages from one specific book with this problem, a lot of manual work, not scalable)
- Verify heading hierarchy, some headings missing as headers used to have the fancier layouts in those books (all headers were second level markdown header, did not get proper hierarchy in any book)
- Fix obvious extraction failures
- Remove any remaining artifacts

### Automated Cleaning

After manual inspection, I identified common patterns suitable for automated cleaning using regex. Each book had different specific errors from conversion, so this was again a very manual task to identify them. 

These are some of the patterns that were removed:


<div align="center">

| Pattern | Example Match | Purpose |
|---------|---------------|---------|
| `FIGURE_TABLE_CAPTION` | "Figure 2. Model diagram" | Remove orphaned captions |
| `LEARNING_OBJECTIVE` | "LO 1.2" | Remove textbook learning objectives |
| `SINGLE_CHAR` | "a" (isolated line) | Remove OCR noise, diagram labels |
| `FIG_TABLE_REF` | "(Figure 2)" | Remove parenthetical figure references |
| `FOOTNOTE_MARKER` | "fn3" | Remove footnote markers mid-sentence |
| `/u2014` | `--` | PDF extraction em-dash artifact |

</div>



There is also other structural cleaning like incorrectly split paragraphs based on punctuation:
```
Input:  "The brain controls\n\nbehavior through"
Output: "The brain controls behavior through"
```


## Phase 4: NLP Segmentation

Converts cleaned markdown into structured **paragraphs with sentence-level granularity**, preparing text for chunking strategies that require sentence boundaries.



### Library Choice: scispaCy

The code uses [spaCy](https://spacy.io/) as the NLP framework, with a [scispaCy](https://allenai.github.io/scispacy/) model (`en_core_sci_sm`) trained on biomedical/scientific text. scispaCy models are spaCy-compatible — installed separately but loaded via standard `spacy.load()`.

For this neuroscience/philosophy corpus, `en_core_sci_sm` handles academic writing better than generic models:

<div align="center">

| Challenge | scispaCy Advantage |
|-----------|-------------------|
| Academic abbreviations | Doesn't split on "et al.", "Fig.", "i.e." |
| Long complex sentences | Trained on scientific writing patterns |
| Parenthetical citations | "(Smith, 2020)" doesn't create fragment |

</div>

Configuration from `src/config.py`:
```python
SPACY_MODEL = "en_core_sci_sm"
MIN_SENTENCE_WORDS = 2
VALID_ENDINGS = ('.', '?', '!', '"', '"', ')', ']')
```

### Implementation

The segmentation keeps section information as context metadata for future chunks. It also separates NLP chunks into paragraphs.

The segmenter uses a lazy singleton pattern to avoid reloading the spaCy model on each call:

```python
# src/content_preparation/segmentation.py

def segment_document(clean_text: str, book_name: str) -> list[dict]:
    """Segment document into structured chunks with context."""

    # Split by markdown headers, track chapter/section hierarchy
    sections = re.split(r'(^#+\s+.*$)', clean_text, flags=re.MULTILINE)

    for segment in sections:
        # Update context from headers (# = chapter, ## = section)
        if segment.startswith('#'):
            # Track current_chapter, current_section
            continue

        # Process paragraphs
        for paragraph in segment.split('\n\n'):
            sentences = _get_sentences(paragraph)  # spaCy
            valid_sentences, _ = _filter_sentences(sentences)

            # Build output with context path
            chunk_data = {
                "context": f"{book_name} > {chapter} > {section}",
                "text": " ".join(valid_sentences),
                "sentences": valid_sentences,
                "num_sentences": len(valid_sentences)
            }
```

### Sentence Filtering

spaCy identifies sentence boundaries but doesn't distinguish real sentences from PDF extraction artifacts. This filter removes noise that spaCy incorrectly treated as valid sentences:

```python
def _filter_sentences(sentences: list[str]) -> tuple[list[str], list[str]]:
    """Filter out noise sentences."""
    for sent in sentences:
        # Too short (fragments from diagram labels)
        if len(sent.split()) < MIN_SENTENCE_WORDS:
            removed.append(f"[Too short] {sent}")
        # Starts lowercase (mid-sentence artifacts)
        elif sent[0].islower():
            removed.append(f"[Starts lowercase] {sent}")
        # No terminal punctuation (incomplete extractions)
        elif not sent.endswith(VALID_ENDINGS):
            removed.append(f"[No terminal punctuation] {sent}")
        else:
            kept.append(sent)
```

### Output Format

Each book produces a JSON file with structured paragraphs:

```json
[
    {
        "context": "Behave > Chapter 5 > Us vs Them",
        "text": "The amygdala activates rapidly when viewing faces of other races. This response occurs within 50 milliseconds.",
        "sentences": [
            "The amygdala activates rapidly when viewing faces of other races.",
            "This response occurs within 50 milliseconds."
        ],
        "num_sentences": 2
    }
]
```

A human-readable markdown version is also saved for inspection, showing each paragraph with its context and sentence breakdown.



## Data Flow

Files are moved to different folders after each step:

```
data/raw/{corpus}/*.pdf (pre-cleaned manually)
    │
    ▼ Stage 1: extract_pdf()
data/processed/01_raw_extraction/{corpus}/*.md
    │
    ▼ Manual review (optional)
data/processed/02_manual_review/{corpus}/*.md
    │
    ▼ Stage 2: run_structural_cleaning()
data/processed/03_markdown_cleaning/{corpus}/*.md
    │
    ▼ Stage 3: NLP segmentation
data/processed/04_nlp_chunks/{corpus}/*.json
```

### Running the Stages

```bash
# Stage 1: PDF to Markdown
python -m src.stages.run_stage_1_extraction

# Stage 2: Automated cleaning (reads from 02_manual_review/)
python -m src.stages.run_stage_2_processing

# Stage 3: NLP Segmentation
python -m src.stages.run_stage_3_segmentation
```



## Key Files

<div align="center">

| File | Purpose |
|------|---------|
| `src/content_preparation/extraction.py` | Docling PDF extraction |
| `src/content_preparation/cleaning.py` | Regex cleaning orchestration |
| `src/content_preparation/segmentation.py` | spaCy sentence segmentation |
| `src/config.py` (lines 45-133) | Cleaning patterns, NLP settings |
| `src/stages/run_stage_1_extraction.py` | Stage 1 CLI runner |
| `src/stages/run_stage_2_processing.py` | Stage 2 CLI runner |
| `src/stages/run_stage_3_segmentation.py` | Stage 3 CLI runner |

</div>



## Lessons Learned

1. **Text extraction from PDF takes a significant amount of time and effort**. This was unexpected for me. Extracting text in reading order from PDF is not so easy when layouts are not standard one column and include images, tables or other random elements.

2. **Some pre/post cleaning is essential to get perfect texts**: It is difficult to rely completely on conversion tools to get perfect texts. Errors depend also on specific PDF layout, more variety in corpus layout means more cleaning patterns to identify. There is also a tradeoff between the quality of the text entering the chunking phase and the amount of effort dedicated. If I had to do this for a production project I would first measure the effect of these initial errors in the final quality to see how much effort is necessary.


3. **You need to find the right tools**. The 2025-2026 PDF extraction landscape offers several tiers of solutions, each with different tradeoffs:

    - **Rule-based (coordinate extraction)**: PyMuPDF parses the PDF file format directly, extracting text from internal objects (coordinates, fonts, text runs). No machine learning. Blazing fast (0.1s/page), CPU-only, zero cost, but no semantic understanding—struggles with multi-column layouts and mixed content.

    - **Modular ML pipelines**: MinerU, Marker, and Docling chain multiple specialized models: object detection (YOLO, RT-DETR) for layout regions, OCR engines (PaddleOCR, Surya) for text extraction, and optional models for tables or formulas. Good accuracy, GPU recommended. MinerU (48K stars, AGPL), Marker (GPL), Docling (MIT, ~2.5 pages/sec CPU, 97.9% table accuracy, native RAG framework integrations).

    - **End-to-end Document VLMs**: Single unified vision-language models trained to go directly from image to structured text. Granite-Docling-258M (IBM, Apache 2.0, 258M params, 0.97 TEDS on tables) and GOT-OCR 2.0 (580M params, Apache 2.0 on HuggingFace) consolidate layout, tables, equations, and code into one model—replacing entire pipelines.

    - **Commercial parsers**: LlamaParse ($0.003-0.09/page, LlamaIndex integration, ~99% accuracy), Reducto ($0.015/credit, bounding box citations for provenance). Major cloud options: Azure Document Intelligence (Read: $1.50/1K, Prebuilt: $10/1K, Formula add-on: $6/1K) and Google Document AI ($1.50/1K, 200+ languages, Gemini-powered). For equations: Mathpix ($0.005/page, $0.0035 at volume).

    - **Frontier VLMs**: Claude Opus 4.5, Gemini 3 Pro, and GPT-5.2 can achieve 90%+ precision with minimal post-processing. Near-perfect for simple layouts, but expensive ($0.01-0.10+/page for vision). **I would look at this [DocVQA leaderboard](https://llm-stats.com/benchmarks/docvqatest) if I had to do it professionally today.**

   For this project's scope—a small corpus of academic books excluding citations, equations, images and tables, and without access to a GPU and much time to dedicate—Docling was a good choice. For a company processing thousands of documents daily, paying for frontier model APIs often makes more economic sense than maintaining extraction infrastructure.




## References

**NLP Segmentation**
- [scispaCy](https://allenai.github.io/scispacy/) - spaCy models for biomedical/scientific text (Allen AI)
- [spaCy](https://spacy.io/) - Industrial-strength NLP library

**PDF Extraction (Open-source)**
- [PyMuPDF](https://pymupdf.readthedocs.io/) - Rule-based coordinate extraction, 0.1s/page, AGPL
- [MinerU](https://github.com/opendatalab/MinerU) - Modular pipeline (YOLO + PaddleOCR), AGPL
- [Marker](https://github.com/VikParuchuri/marker) - Modular pipeline (Surya OCR), GPL
- [Docling](https://github.com/docling-project/docling) - Modular pipeline (RT-DETR + TableFormer), MIT
- [Granite-Docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) - End-to-end VLM, 258M params, Apache 2.0
- [GOT-OCR 2.0](https://huggingface.co/stepfun-ai/GOT-OCR2_0) - End-to-end VLM, 580M params, Apache 2.0

**Commercial**
- [LlamaParse](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) - RAG-native, $0.003-0.09/page
- [Reducto](https://reducto.ai/) - Agentic OCR with provenance, $0.015/credit
- [Azure Document Intelligence](https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/) - Read $1.50/1K, Prebuilt $10/1K, Formula add-on $6/1K
- [Google Document AI](https://cloud.google.com/document-ai/pricing) - 200+ languages, Gemini-powered, $1.50/1K
- [Mathpix](https://mathpix.com/pricing/api) - Best for equations, $0.005/page ($0.0035 at 1M+ volume)


---

## Navigation

**Next:** [Chunking Strategies](../chunking/README.md) — How documents are split for embedding

**Related:**
- [Getting Started](../getting-started.md) — Installation and quick start
