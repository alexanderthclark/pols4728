(evals_and_prompting)=
# Using LLMs

```{admonition} Reading
:class: seealso
- [Claude 4 Prompt Engineering Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
```

Large language models (LLMs) are practical tools for social-science tasks such as text coding,
stance classification, summarization, and weak labeling.

## Is using a pre-trained LLM machine learning?

Usually, yes. The learning happened during pretraining; your role is task adaptation.

- You define task and output schema.
- You provide instructions and examples.
- You evaluate quality with held-out labels and error analysis.

Even if you do not train model weights, you still need rigorous measurement.

## How LLMs work (operationally)

At use time, modern LLMs perform conditional next-token generation with internal reasoning
policies. For applied work, the practical consequence is:

- clear task instructions matter,
- output constraints matter,
- evaluation pipelines matter more than prompt folklore.

## Fancy autocomplete and bias

LLMs can reproduce artifacts and biases from training distributions. This makes evaluation on your
actual domain mandatory.

```{figure} ../assets/images/claudehaiku-refusal-20250603.png
:width: 100%
:align: center

Example refusal behavior from Claude Haiku.
```

```{figure} ../assets/images/claude-haiku35-20250603.png
:width: 60%
:align: center

Example completion from Claude Haiku.
```

## Prompting strategy now

Less useful than many older guides suggest:

- role-play priming as a primary optimization strategy,
- magical phrasing tricks,
- long prompt ornamentation without task signal.

Still useful:

- explicit label definitions,
- clear output schema,
- representative examples for ambiguous cases,
- deterministic post-processing and validation.

## Evaluating LLM classifications

For supervised tasks, evaluate LLM outputs exactly as any other classifier.

1. Define labels and a fixed scoring rubric.
2. Hold out test examples not shown in prompt examples.
3. Compare zero-shot vs few-shot prompts.
4. Compare model families/cost tiers.
5. Inspect confusion patterns, not only aggregate accuracy.

## Exercises

```{exercise-start}
:label: llm-classification-compare
```
Pick a labeled text dataset (<= 400 rows). Run:

- zero-shot and few-shot prompts,
- at least one frontier model and one low-cost model.

Compare performance, error patterns, and cost per 1,000 examples.

```{exercise-end}
```

```{exercise-start}
:label: llm-replication
```
Reproduce one published text-labeling workflow using an LLM pipeline and report where the model
fails relative to human-coded labels.

```{exercise-end}
```
