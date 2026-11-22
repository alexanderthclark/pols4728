# A small trained PFN

`mini_pfn.py` trains an educational network on a deliberately restricted two-world prior. This is not TabPFN. It is a compact example of learning a context-to-prediction mapping across generated tasks.

```sh
python -m pip install torch==2.8.0
python examples/tabular-foundation-models/mini_pfn.py
```

Python 3.11 and CPU execution with two threads were used for the saved run. The default is 4,000 Adam updates, 256 newly generated tasks per update, learning rate 0.002, and width 32. Every task chooses one hidden world with probability one half. That world governs four independently generated labeled context rows and one query. Raw binary bear type, response, and context outcomes are the only model inputs. The query outcome is used solely as the loss target. No match feature, world indicator, Bayesian posterior, or analytic prediction is supplied to the network.

The network separately embeds each context row and the query through small MLPs. One scaled-dot-product cross-attention head computes a weighted mixture of context value vectors. A decoder receives that mixture and the query representation and predicts query survival. All parameters are shared across tasks and updated during pretraining. A new task changes the context, not the trained parameters.

The saved JSON and corresponding `mini-pfn-results.mjs` include training checkpoints, the actual first Adam update on one minibatch, aggregate evaluation on 10,000 independent tasks, learned parameters, and two complete forward-pass traces. The traces contain actual queries, keys, values, attention weights, weighted values, their sum, decoder logits, and predicted probabilities. They use the same Brown + Lie down query and flip only the context labels. These weights are not posterior probabilities over worlds or feature-importance scores.

Evaluation compares the learned predictor with exact Bayesian prediction under the specified prior and with a predictor that always returns 0.5. Observed-label log loss uses newly sampled query outcomes. Expected excess log loss scores both predictors against the analytic conditional outcome distribution, which removes the extra randomness from the sampled test labels. The analytic comparator is used only for evaluation.

The tiny prior has exactly two binary input features, two possible worlds, and four context rows. There are no continuous distractors here, and no result establishes performance on other datasets, longer contexts, or actual TabPFN checkpoints. The model uses a fixed training schedule and the final weights, with separate pseudorandom generators for training, validation, and test tasks.
