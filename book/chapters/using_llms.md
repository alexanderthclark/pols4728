(evals_and_prompting)=
# Using LLMs

```{admonition} Reading
:class: seealso
- [Claude 4 Prompt Engineering Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
```

LLMs are used in social science research for both classification (sentiment analysis, for example), document scaling, and topic modeling. The term "foundation models" is used to describe such general-purpose models. {cite}`haaland2025understanding` highlights the use of LLMs to code open-ended data, noting LLMs will "be the preferred choice over most existing text analysis methods for survey researchers." {cite}`ornstein2025train` notes where LLMs underperform, though the analysis is limited to GPT-4 and GPT-3. Other methods might be preferred when high-quality training data is available or in the case of very large data where LLMs are expensive. What does very large mean? Using GPT-5 through OpenAI's API, it would cost maybe \$5 to classify 10,000 text responses of about 100 words.[^llm-cost] These costs scale linearly, so classifying 1,000,000 responses would cost \$500. More complicated prompting could triple the cost.[^few-shot] One can save costs by switching to a cheaper model like GPT-5 nano, which is 20-25x cheaper than GPT-5, or using classical text analysis methods. 

The LLM cost would likely be dwarfed by the general administrative costs of conducting any survey. For example, {cite}`roberts2014structural` uses structural topic modeling to analyze open-ended survey responses from the American National Election Survey, which included 2,323 respondents. The LLM cost would be relatively trivial. 


[^llm-cost]: *Assumptions.* Model = GPT-5 API priced at \$1.25 per 1M **input** tokens and \$10.00 per 1M **output** tokens. Tokenization rule-of-thumb: **100 tokens ≈ 75 words** ⇒ 100 words ≈ ~130 input tokens. Output kept minimal (label + brief rationale) at ~30 tokens. No few-shot examples; reusable instructions negligible and **no prompt caching** assumed. Per-response cost ≈ \(130×1.25×10^{-6} + 30×1×10^{-5} = \$4.625×10^{-4}\). Totals: **10k** responses ≈ **\$4.63** (rounded to **\$5**); **1M** responses ≈ **\$462.5** (rounded to **\$500**). Costs scale linearly with tokens.


[^few-shot]: *Assumptions.* Few-shot, no caching. A prompt includes the 130 input tokens *and* a 200-token rubric with 500 tokens of examples. This modifies the previous calculation by now using 830 input tokens. We still assume 30 output tokens.  Totals: **10k** responses ≈ **\$13.38**; **1M** responses ≈ **\$1338.00**. 

While creating and fine-tuning LLMs is beyond our scope, we'll use them to:
- Ease into coding
- Explore performance measures for classification tasks
- Practice effective prompting

If an LLM solves your problem adequately, there's no need for more complex ML approaches unless the size of the data makes the LLM route too costly. 

## Is Using Pre-trained LLMs Machine Learning?

This raises an interesting philosophical question, which we might file next to questions like, "Is Katy Perry an Astronaut?"

**Traditional ML Classification:**
- You provide labeled data (experience E)
- Train a model on your specific task (T)
- Performance (P) improves with more of your data

**Using Pre-trained LLMs:**
- The model already learned from internet-scale text (someone else's E)
- You craft prompts to apply its knowledge to your task
- Model parameters do not change as you provide more data. 

So is it machine learning? Yes, by Mitchell's definition. The learning happened during pre-training, not when you use it. So, using an LLM doesn't make you a machine learning engineer the same way using a chess engine doesn't make you an AI researcher. There is one wrinkle: the model's performance does improve when you provide clearer instructions or examples. An LLM will adapt to your task without changing its underlying knowledge. For practical, research purposes, this distinction doesn't matter. You're still solving classification problems, and you still need rigorous evaluation. Whether you trained the model or someone else did, the scientific method remains the same: define your task, measure performance, validate results.


## How LLMs Work

How LLMs work is an overly ambitious title for this short section. They are next-token predictors, but that's not a useful description. Nobody really has a comprehensive answer to what LLMs are doing, at least [according to Stuart Ritchie of Anthropic](https://youtu.be/fGKNUvivvnc?si=6vHzbT3D7zb-maPl&t=13). But here are some reasonable points to be made.

**The "Old" Approach (GPT-3.5 Era):**
- "Like a fancy autocomplete,'' according to Peter Grabowski of Google 
- Statistical pattern matching from many examples
- Role prompts (e.g., "You are an MIT mathematician") shifted probability distributions to improve the chances of the autocomplete being correct
- Hypothesized sweet spot in detail and length of a prompt

**Modern Chain-of-Thought Models:**
- Built-in multi-step reasoning
- Hidden "thinking" processes before output
- Self-selected reasoning approaches
- Role-playing prompts less effective

In the video below, I ask ChatGPT's o3 how to get to Carnegie Hall. You can see the thinking process (at 8x speed), where o3 reasons through the intent of my question based on what it already knows about me. ChatGPT 5 Thinking will behave like o3. 

<div style="width: 99%; margin: auto;">
  <iframe src="https://www.youtube.com/embed/_tGDS6g63So?si=JU-cZFBjiaZ8bad"
          style="width: 100%; aspect-ratio: 16 / 9; border: 0;"
          allow="accelerometer; autoplay; clipboard-write;
                 encrypted-media; gyroscope; picture-in-picture; web-share"
          allowfullscreen>
  </iframe>
</div>

Claude Haiku or ChatGPT 4.5 are more likely to answer "Practice, practice, practice." This answer is seemingly the best for a sophisticated autocomplete. 

### Fancy Autocomplete and Bias

Fancy autocompletes have their drawbacks and these garner a lot of attention. Namely, LLMs can forward whatever bias is in their training data (like any other model). As noted in {cite}`atari2023humans`, LLMs are biased toward the pyschology of people from WEIRD (Western, Educated, Industrialized, Rich, and Democratic) societies. LLMs exhibit human-like cognitive biases when trying to generate random sequences {cite}`van2024random`. If you ask AI for a random number, some models disproportionately choose 42. Claude is especially bad at this, in my own experience. 42 is a salient number because of Douglas Adams's "The Hitchhiker's Guide to the Galaxy." Its fans are overrepresented on the Internet and thus in training data. Similarly, LLMs can reproduce stereotypes. Companies devote enormous resources to mitigating these biases, but it is not a solved problem.

```{figure} ../assets/images/claudehaiku-refusal-20250603.png
:width: 100%
:align: center

Response from Claude Haiku 3.5
```

```{figure} ../assets/images/claude-haiku35-20250603.png
:width: 60%
:align: center

Response from Claude Haiku 3.5
```

## Prompt Engineering Evolution

Prompt engineering has evolved, meaning many of self-annointed gurus on LinkedIn should probably be ignored. And instead of falling into the trap of trying to hit a moving target by stating best practices in the era of GPT-5, we'll only take the time to mention that prompting strategies, like role priming, that attempted to shift the model to a better probability distribution are no longer important. 

**Less effective now:**
- Role priming ("You are an expert...")
- Explicit "think step-by-step" instructions
- Confidence boosting
- Mystical incantations

**Still valuable:**
- Clear problem specification
- Relevant context and constraints
- Examples of desired output format
- Domain-specific information
- Being smarter than the LLM because they still hallucinate

It used to be emphasized that LLM performance improves if the prompt included an example with the desired output, instead of merely giving instruction. A prompt with one example corresponds to a one-shot prompting strategy and mutatis mutandis for zero-shot and multi-shot. Something I haven't addressed above is if newer chain-of-thought models still improve with multi-shot prompting strategies. We'll test this for ourselves shortly.

# Evaluating LLM Classifications for Supervised Tasks

{cite}`gilardi2023chatgpt` showed that ChatGPT can outperform crowd workers (Mechanical Turk) on text annotation tasks. {cite}`tornberg2024large` shows that GPT-4 outperforms even experts in annotating political social media messages. But how do we know this? How do we measure performance? And how do we find the best prompt? Instead of approaching this through prompt engineering principles, we'll tackle this in a data-driven way by *trying stuff and seeing what works best*. 

## Zero vs. Few Shots: What Does the Evidence Say?

| Study | Model(s) | Task family | Main result |
|-------|----------|-------------|-------------|
| Brown et al. 2020 | GPT‑3 | 42 NLP benchmarks | 0‑shot < 1‑shot < 16‑shot (classic few‑shot curve) |
| Kojima et al. 2022 | GPT‑3 | Reasoning (GSM‑8K) | Adding *"Let's think step by step"* closes 60 % of gap to few‑shot. |
| Wei et al. 2022 | PaLM‑540B | Math, commonsense | 8‑shot CoT boosts accuracy by up to 36 pp over plain few‑shot. |
| Liyanage et al. 2024 | GPT‑4 | Twitter stance | Zero‑shot CoT matched 4‑shot accuracy (≈ 93 %). |
| Chen et al. 2024 | GPT‑4 | 6 tasks | Extra demos show diminishing returns after coverage of each label. |

Including examples should be less important in more modern models. You will test this in your homework. 


## Exercises

```{exercise-start}
:label: LLM
```
Find a labeled text data set of interest ([here is one](https://github.com/Ravihari123/GPT-for-Twitter-Stance-Labeling/blob/main/final_annotated_dataset_355%20records.csv) from {cite}`liyanage2024gpt`) with no more than 400 observations. Write two prompts for classification: one with two examples and one with no examples. Classify the entire data set using Gemini 2.5 Pro and then Gemini 2.5 Flash-Lite for a total of four "models." Compare accuracy across each. You will be provided with an API key and some sample code. 

```{exercise-end}
```

```{exercise-start}
:label: LLM-grimmer
```
Replicate {cite}`grimmer2010bayesian` using an LLM. Find the data [here](https://github.com/lintool/GrimmerSenatePressReleases).

```{exercise-end}
```