# Using LLMs

```{admonition} Reading
:class: seealso
- [Claude 4 Prompt Engineering Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
```

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

So is it machine learning? By Mitchell's definition—yes. The learning happened during pre-training, not when you use it. So, using an LLM doesn't make you a machine learning engineer the same way using a chess engine doesn't make you an AI researcher. There is one wrinkle: the model's performance does improve when you provide clearer instructions or examples. An LLM will adapt to your task without changing its underlying knowledge. For practical, research purposes, this distinction doesn't matter. You're still solving classification problems, and you still need rigorous evaluation. Whether you trained the model or someone else did, the scientific method remains the same: define your task, measure performance, validate results.


## How LLMs Work

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

Prompt engineering has evolved, meaning many of self-annointed gurus on LinkedIn should probably be ignored. And instead of falling into the trap of trying to hit a moving target by stating best practices in the era of GPT 5, we'll only take the time to mention that prompting strategies, like role priming, that attempted to shift the model to a better probability distribution are no longer important. 

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

Something I haven't addressed above is if models do well with zero-shot vs one-shot vs multi-shot prompts. We'll find out for ourselves shortly.

# Evaluating LLM Classifications for Supervised Tasks

{cite}`gilardi2023chatgpt` showed that ChatGPT can outperform crowd workers (Mechanical Turk) on text annotation tasks. {cite}`tornberg2024large` shows that GPT-4 outperforms even experts in annotating political social media messages. But how do we know this? How do we measure performance? And how do we find the best prompt? Instead of approaching this through prompt engineering principles, we'll tackle this in a data-driven way by *trying stuff and seeing what works best*. 

## 1 Evaluation Mindset ≈ Test‑Driven ML

**Key idea:** treat a *prompt + model* pair as a *hypothesis* you must test against a labelled gold set.

Consider Mitchell's framework: a system learns from experience E with respect to task T and performance measure P. With pre-trained LLMs, we're in a remarkable situation—we define T through our prompts, but E (the training on internet-scale text) was done by someone else. We still need to rigorously measure P to know if our prompts work.


### 1.1 Micro‑workflow

1. **Gold set** – 20‑200 hand‑labelled examples (`review`, `sentiment`).
2. **Metric** – accuracy, F‑1, or task‑specific cost (see *Learning* §Performance Measures).
3. **Batch run** – execute the prompt on every row.
4. **Record** – score, token cost, failure cases → iterate.

> You now have a **unit test for prompts.**

---

## 2 Evaluating Prompts: Think A/B Testing

When social scientists evaluate survey questions, we test different wordings to see which performs better. Evaluating LLM prompts works the same way—it's just A/B testing with different prompt templates.

Here's a minimal evaluation setup using the same tools from your assignments:

```python
import google.generativeai as genai
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

# Setup (same as your homework)
genai.configure(api_key=YOUR_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def evaluate_prompt(test_data, prompt_template):
    """Run a prompt on test data and measure performance"""
    predictions = []
    
    for text in test_data['text']:
        # Insert text into your prompt template
        prompt = prompt_template.format(text=text)
        response = model.generate_content(prompt)
        predictions.append(response.text.strip())
    
    # Calculate metrics
    accuracy = accuracy_score(test_data['true_label'], predictions)
    cm = confusion_matrix(test_data['true_label'], predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'confusion_matrix': cm
    }

# Define prompt templates
ZERO_SHOT_PROMPT = """Classify the sentiment of this review as 'positive' or 'negative'.
Review: {text}
Answer:"""

FEW_SHOT_PROMPT = """Classify sentiment as 'positive' or 'negative'.

Review: "Terrible product, waste of money"
Answer: negative

Review: "Excellent quality, highly recommend!"
Answer: positive

Review: {text}
Answer:"""

# A/B test two prompts
zero_shot_results = evaluate_prompt(test_data, ZERO_SHOT_PROMPT)
few_shot_results = evaluate_prompt(test_data, FEW_SHOT_PROMPT)

print(f"Zero-shot accuracy: {zero_shot_results['accuracy']:.2%}")
print(f"Few-shot accuracy: {few_shot_results['accuracy']:.2%}")
```

This is exactly how {cite}`gilardi2023chatgpt` determined that ChatGPT beats crowd workers—they tested prompts systematically on labeled data, not by eyeballing a few examples.

The key insight: treat each prompt variant as an experimental condition. Your "treatment" is the prompt wording, and your outcome is classification accuracy.

---

## 3 Prompt Patterns That Survive 2025

Modern chain-of-thought models have moved past the verbose "you are an expert" prompts from 2022. Today's models want clear, structured instructions. Here are three patterns that consistently deliver:

### 3.1 Zero‑Shot CoT

```jinja
Classify the sentiment of the review between <<< >>>.
Respond with **Positive** or **Negative** only.

Review: <<<{{input}}>>>

Let's think step‑by‑step.
```

* Pros – 1‑line template, no demos, often ≥ 80 % accuracy on vanilla sentiment.
* Cons – may hallucinate labels; longer reasoning increases cost.

### 3.2 One‑Shot (format anchor)

```jinja
You are a sentiment analyst.  
Return `Positive` or `Negative`.

Example  
Review: "Worst purchase ever."  
Answer: Negative  
---  
Review: "{{input}}"  
Answer:
```

*Use when zero‑shot misunderstands output schema.*

### 3.3 Few‑Shot + Structured CoT

```jinja
Task: Sentiment classification (Positive/Negative)  
Output schema:  
  - chain_of_thought: string  
  - label: Positive|Negative  

Review: "Loved it. Five stars!"
chain_of_thought: Uses strong positive verbs ("Loved"), 5‑star rating → Positive
label: Positive
---
Review: "Terrible, broke day 2."
chain_of_thought: Breakage and complaint → Negative
label: Negative
---
Review: "{{input}}"
chain_of_thought:
```

* Best for nuanced or domain‑specific tasks (e.g., sarcasm, political stance).  
* Showcases *explanation* → easier error analysis.

---

## 4 Zero vs. Few Shots: What Does the Evidence Say?

| Study | Model(s) | Task family | Main result |
|-------|----------|-------------|-------------|
| Brown et al. 2020 | GPT‑3 | 42 NLP benchmarks | 0‑shot < 1‑shot < 16‑shot (classic few‑shot curve) |
| Kojima et al. 2022 | GPT‑3 | Reasoning (GSM‑8K) | Adding *"Let's think step by step"* closes 60 % of gap to few‑shot. |
| Wei et al. 2022 | PaLM‑540B | Math, commonsense | 8‑shot CoT boosts accuracy by up to 36 pp over plain few‑shot. |
| Liyanage et al. 2024 | GPT‑4 | Twitter stance | **Zero‑shot CoT matched 4‑shot accuracy** (≈ 93 %). |
| Chen et al. 2024 | GPT‑4 | 6 tasks | Extra demos show diminishing returns after coverage of each label. |


The kicker? {cite}`gilardi2023chatgpt` found that ChatGPT beat crowd workers on political annotation tasks—even with zero-shot prompts. No fancy engineering required. But here's the catch: they only knew this because they measured properly.

