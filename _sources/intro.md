# Welcome


```{important}
These notes are not entirely self-contained. They are intended to highlight important topics and should be read in conjunction with other course readings.
```

These are the course notes for *POLS 4728: Machine Learning and Artificial Intelligence for the Social Sciences*. 

## Course Design Choices

- **Forward-looking Topic Selection**. I sometimes reference a "canon" of machine learning techniques, but that canon is not frozen. LLMs outperform classical natural language processing techniques and LLMs are practical for most academic use cases, so we don't spend as much time on text analysis as we might have a few years ago. Similarly, we skip recurrent neural networks in favor of transformers. Topic selection is informed by my industry experience, surveys like {cite}`athey2019machine` and {cite}`de2024use`, and my best guess regarding the future.
- **Applied**. The intended audience is researchers who will *use* machine learning. This course will not prepare you to make significant methodological contributions. Instead, you will understand the applicability of different models and be able to implement them. 
- **Survey of Fundamentals**. We prefer deep, applied understanding of the simple case for the important methods to (1) even greater breadth that sacrifices understanding or (2) premature optimization where we cover extensions to panel data, time series data, etc. With foundations in place, you the researcher will be prepared to learn necessary extensions in the future.
- **AI Maximalist**. We embrace AI for writing code and as a study partner. 

### Topic Selection

{cite}`de2024use` analyzed 339 articles to identify the most frequently used machine learning methods in political science. Topic modeling emerges as the most common approach, followed by random forest and support vector machines. Individual articles may employ multiple methods.

```{table} Most frequently used ML methods in political science (n=339)
:name: tbl-ml-methods-de-slegte-2024

| Machine Learning method or method class | Paradigm |  N | Percentage |
|----------------------------------------|----------|---:|-----------:|
| Topic Modeling Methods                 | UML      | 84 |       24.8 |
| Random Forest                          | SML      | 55 |       16.2 |
| Support Vector Machine                 | SML      | 55 |       16.2 |
| Natural Language Processing Methods    | UML      | 44 |       13.0 |
| Neural Networks                        | SML      | 42 |       12.4 |
| Other Tree-based Methods               | SML      | 30 |        8.8 |
| Regularization Methods                 | SML      | 28 |        8.2 |
| Naive Bayes                            | SML      | 26 |        7.7 |
| Other Text Mining Methods              | UML      | 23 |        6.8 |
| Logistic Regression                    | SML      | 20 |        5.9 |
| Bayesian Machine Learning Methods      | SML      | 13 |        3.8 |
| Clustering Methods                     | UML      | 12 |        3.5 |
| Kernel Regression Methods              | SML      |  7 |        2.1 |
| Causal Forest                          | SML      |  5 |        1.5 |
| Supervised Machine Learning Methods    | SML      |  6 |        1.8 |
| Semi-supervised Machine Learning Methods | SML    |  1 |        0.3 |
| Other Machine Learning Methods         | SML/UML  | 28 |        8.3 |
```

This analysis helps guide our topic selection, but I deemphasize NLP methods and some old-fashioned methods like support vector machines that I have never seen used in data-rich settings. 

### AI Maximalism

It is worth clarifying that AI maximalism does not equate to using AI always and everywhere. Nor does it necessarily equate to techno-optimism. I could easily substitute the phrase "AI realism," meaning that powerful AI models are useful enough and here-to-stay enough that it is silly to ignore them in 2025. 

This isn't necessarily advice to do the same, but here is how I use AI currently. I use both ChatGPT and Claude. I use [Claude Code](https://www.anthropic.com/claude-code) extensively (see the [Claude Code in Action course](https://anthropic.skilljar.com/claude-code-in-action)). And I think I use all of these a little too much. For coding, my best work is probably done by working *homo solus* and then copying and pasting into conversation with the best model that is available for periodic feedback or to overcome obstacles. This is how I made progress in unfamiliar territory like creating Jupyter books or Python packages. Claude Code and other tools reduce some tedium but with the risk of promoting inattention. I do not use Copilot and find pure vibe coding usually leads to a spiral of misery. My caution against vibe coding is therefore part practical and part encouragement to seek out so-called desirable difficulties.

As graduate students and researchers, your job is to work on the frontier of some particular subject. This is impossible if you can only learn from YouTube videos and not from reading a book or research article cold. *That should be obvious*. Similarly, real fluency in coding or machine learning will require being able to work directly with primary documentation and texts instead of solely through the so-far-imperfect AI models we have.  



