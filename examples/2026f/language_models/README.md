# Using language models

These Python examples accompany the chapter's Peloton transcript case. Each file
contains its own setup and can be read independently. The examples illustrate
the methods; students may choose other software or providers.

| File | Chapter example | Optional dependencies |
| --- | --- | --- |
| [01_schema.py](01_schema.py) | Subclassing `BaseModel`, validating records, and checking evidence | pandas, Pydantic 2 |
| [02_huggingface.py](02_huggingface.py) | A basic hosted request and a schema-constrained annotation | pandas, Pydantic 2, huggingface_hub |
| [03_local_generation.py](03_local_generation.py) | Local text generation | pandas, transformers, torch |
| [04_openai.py](04_openai.py) | Structured output through the OpenAI Responses API | pandas, Pydantic 2, openai |
| [05_gemini.py](05_gemini.py) | Structured output through Gemini | pandas, Pydantic 2, google-genai |
| [06_evaluation.py](06_evaluation.py) | Zero-shot and one-example procedures, recorded failures, and evaluation | pandas, Pydantic 2, huggingface_hub, scikit-learn |
| [07_retrieval.py](07_retrieval.py) | Normalized embeddings and passage retrieval | pandas, sentence-transformers |
| [08_rag.py](08_rag.py) | A hosted answer grounded in retrieved passages | pandas, huggingface_hub, sentence-transformers |

The OpenAI and Gemini examples are independent alternatives. They do not need
a Hugging Face account or another script to run first. The `# %%` comments mark
the same conceptual steps used in the chapter; they are ordinary Python
comments, so the files also work as scripts.

## Separately shared class data

The instructor shares the Peloton corpus in one ZIP outside GitHub. It includes
the full transcripts, a sentence table, class metadata and links, and the
chapter's selected passages. The examples read these files from a `data/` folder
in the current working directory:

- `peloton-burton-excerpts.csv`: eleven passages, with identifiers and teaching roles.
- `peloton-burton-worked-example.json`: the worked annotation used by `01_schema.py`.
- `peloton-burton-reference-labels.csv`: the eight-row template used by `06_evaluation.py`.

The ZIP contains that folder structure. The working directory is the folder
containing `data/`, even when the Python source is stored elsewhere. The full
corpus and the worked example's quotations stay in this separate data bundle.
The worked annotation illustrates the schema; it is not a substitute for the
independent development labels.

## BERTopic comparison data

[wellness_theme_passages.csv](../../../data/wellness_theme_passages.csv) contains
the same 952 passages used in the Rand/Calvin classifier exercise, combining its
training, validation, and test splits for topic exploration. The columns are
`passage_id`, `text`, `label`, `source_file`, and `class_url`.

Labels are assistant annotations: Ayn Rand (163), Calvin (77), and neither (712).
The sample was selected partly using theme-related words; its counts do not
estimate the prevalence of the themes in the full corpus. BERTopic fits the text
without these labels, then compares the discovered topics with them.

## Hosted examples

The hosted examples read credentials and model choices from environment
variables. No credentials are stored in these files.

| Files | Variables |
| --- | --- |
| `02_huggingface.py`, `06_evaluation.py`, `08_rag.py` | `HF_TOKEN`; optional `HF_CHAT_MODEL` and `HF_PROVIDER` |
| `04_openai.py` | `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL` |
| `05_gemini.py` | `GEMINI_API_KEY`, `GEMINI_CHAT_MODEL` |

The Hugging Face defaults match the chapter. A chosen model/provider combination
needs to support the illustrated request, including JSON-schema output for
annotation. Model availability can change. The local generation and embedding
examples download their model files on first use unless those files are already
cached.

`02_huggingface.py` makes two hosted requests: one basic request and one structured
annotation. The OpenAI, Gemini, and RAG examples each make one hosted request.
`06_evaluation.py` attempts sixteen requests: two procedures applied to the eight
development passages. Its request loop preserves failed attempts in the output.

## Reference labels and evaluation

The reference-label template is deliberately blank. Complete and review those
judgments independently before running `06_evaluation.py` or inspecting model
predictions. Allowed labels are `present`, `absent`, and `unclear` for each of the
two themes. The validation code rejects missing or invalid labels.

The standalone script checks the reference file before making requests. This
moves the chapter's existing label checks earlier, so the blank template stops
execution before a hosted request is made. Once the reference labels pass those
checks, each run writes a new `annotations-<run-id>.jsonl` file in the working
directory.
That file contains passage text as well as predictions and errors. The local
`.gitignore` excludes the class data, ZIP files, generated annotation records,
and credential files from this example folder.

The evaluation includes failed requests in the overall accuracy denominator and
also reports accuracy among usable outputs. A schema-valid record and a matching
quotation can still encode a substantively wrong label.
