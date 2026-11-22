import {miniPfnResults as run} from './mini-pfn-results.mjs';
const A=run.examples.A, B=run.examples.B;
const pct=x=>`${(100*x).toFixed(1)}%`, f=x=>x.toFixed(3);
const training='Learning to use a table', inside='An actual prediction';
export const mechanismScenes=[
  {
    chapter:training, kind:'mechanism', mechanismStage:0,
    title:'One rule within a table. Different rules across tables.',
    body:'Our opening simulation uses rule A. For pretraining, choose A or its reverse, B, once per table. Every encounter in that table shares the chosen rule.',
    equation:'P(\\phi_t=A)=P(\\phi_t=B)=\\tfrac12',
    note:'The network is not told which rule was chosen. This deliberately small task family makes the learning problem visible.',
    reading:'We built and trained a small educational prior-data fitted network for this two-world family. It is not TabPFN. World A is the opening bear rule; B reverses all four survival probabilities. Each task chooses its world with equal probability, then draws four labeled context encounters and one query encounter. Bear type and strategy are independent balanced binary features; outcomes are conditionally independent Bernoulli draws. Crucially, the world is shared within a table. If we instead drew a fresh world for every row, context outcomes would tell us nothing about the query. Real tabular foundation models train on much broader families of tasks.'
  },
  {
    chapter:training, kind:'mechanism', mechanismStage:1,
    title:'Turn one table into a prediction exercise.',
    body:'Give the network four labeled encounters and the next encounter’s features. Keep its survival outcome outside the input.',
    equation:'\\underbrace{(D,x_*)}_{\\text{input}}\\longrightarrow q_\\theta(Y_*=1\\mid x_*,D)',
    note:'Columns are 0/1 indicators: brown bear, lie down, survived. The hidden answer and generating rule stay outside the input.',
    reading:'The displayed rows form one possible context under the generator, selected to make the contrast easy to follow. The query is Brown + Lie down. Each labeled row supplies raw numeric bear, strategy, and survival indicators; the query supplies only the two feature indicators. The neural network must turn that whole prediction exercise into one survival probability. In the actual training run, contexts and query outcomes are sampled afresh for each task; they are not always this particularly clear pattern.'
  },
  {
    chapter:training, kind:'mechanism', mechanismStage:2,
    title:'The hidden answer supplies the training signal.',
    body:`If the query outcome is survival, a prediction of ${pct(A.probability)} incurs log loss ${f(-Math.log(A.probability))}. Backpropagate the loss to update the shared network weights.`,
    equation:`\\ell=-\\log(${A.probability.toFixed(4)})\\approx${f(-Math.log(A.probability))}`,
    note:'One loss calculation using the trained teaching model. The answer enters the loss, never the context.',
    reading:`This calculation uses our trained model’s actual prediction for the displayed context. Take a possible held-out draw Y*=1. Binary cross-entropy is −[Y* log(q) + (1 − Y*) log(1 − q)], so the loss here is ${(-Math.log(A.probability)).toFixed(6)}. If z is the output logit, its derivative is q − Y*, here ${f(A.probability-1)}. Backpropagation passes that signal through the decoder, attention projections, and encoders. Training averages the losses of 256 separate tasks before each Adam update. An individual contradictory outcome can push the other way; optimizing across noisy tasks rewards useful probabilities, not certainty. The diagram illustrates the update operation without inventing a numerical post-update prediction.`
  },
  {
    chapter:training, kind:'mechanism', mechanismStage:3,
    title:'The next table can reverse the evidence.',
    body:'Keep the same bear types, strategies, and query. Reverse the context outcomes. Those labels now support the opposite survival rule.',
    equation:'x_*\\text{ unchanged},\\qquad D_A\\ne D_B',
    note:'One memorized bear rule cannot solve both tasks. The labeled rows reveal which pattern this table supports.',
    reading:`The two displayed contexts are equally valid possible inputs. Each of the first table’s four outcomes favors A; each reversed outcome favors B. The query Brown + Lie down is identical. Our trained network predicts ${pct(A.probability)} from the first context and ${pct(B.probability)} from the second, using exactly the same saved weights for both forward passes. These are measured outputs of the teaching model, not probabilities copied from the simulator or outputs from TabPFN.`
  },
  {
    chapter:training, kind:'mechanism', mechanismStage:4,
    title:'Using the context lowers prediction loss.',
    body:'Without context, equally common opposite rules average to 50% survival for every query. The labeled encounters contain information that this pooled prediction discards.',
    equation:'P(Y_*=1\\mid x_*)=\\tfrac12',
    note:'Measured on 10,000 fresh tasks. Lower log loss is better; the Bayesian calculation is exact for this two-world prior.',
    reading:`The context-blind benchmark predicts 0.5, the best possible prediction when the two worlds are equally likely and the context is ignored. Its test log loss is ${f(run.evaluation.ignore_context_log_loss)}. The trained miniature network achieves ${f(run.evaluation.model_log_loss)}; exact posterior predictive inference under the known generator achieves ${f(run.evaluation.bayes_log_loss)}. All three scores use the same independent test tasks and sampled query outcomes. This is evidence that this small network learned to extract useful information from its context. It does not measure TabPFN performance or establish how a production model handles continuous distractors.`
  },
  {
    chapter:training, kind:'mechanism', mechanismStage:5,
    title:'Train one network across all those exercises.',
    body:`We made ${run.metadata.steps.toLocaleString()} updates using ${run.metadata.batch_size} new tasks per update. The encoders, attention projections, and decoder learned together from hidden-outcome errors.`,
    equation:'\\theta\\leftarrow\\operatorname{Adam}\\!\\left(\\theta,\\nabla_\\theta\\frac1{256}\\sum_t\\ell_t\\right)',
    note:'A real training run of a 7,617-parameter teaching model with one attention head. The task generator supplies labels, not Bayesian answers.',
    reading:'The model starts with randomly initialized neural-network weights. Its inputs contain raw binary bear, strategy, and context-label columns. Neither world identity, the matching-pair indicator, the exact posterior, nor a Bayesian prediction is supplied as an input or training target. A context-row MLP and query MLP feed a single cross-attention head; an MLP decoder produces a survival logit. Adam updates all parameters through the sampled query loss. The final checkpoint is fixed by the training schedule, without selecting it against test results. The plotted curve reports observed log loss on the same 2,048 independent validation tasks at each checkpoint, not the training minibatch loss. The saved source and numerical trace make the calculation reproducible.'
  },
  {
    chapter:inside, kind:'mechanism', mechanismStage:6,
    title:'Freeze the weights. Supply a new context.',
    body:`The same trained network returns ${pct(A.probability)} for context A and ${pct(B.probability)} for context B. Its intermediate calculations change because the input changes.`,
    equation:'\\theta\\text{ fixed};\\qquad q_\\theta(y_*\\mid x_*,D_A)\\ne q_\\theta(y_*\\mid x_*,D_B)',
    note:'Two forward passes through one checkpoint. No gradients or weight updates occur between them.',
    reading:'This is in-context learning. The function computed by the trained network is fixed, but it is a function of both the query and the labeled context. Replacing the context changes the row vectors, attention computations, pooled representation, and final prediction. In ordinary task-specific neural-network training, fitting a new dataset changes weights; here, pretraining has already learned a way to use the dataset as input. These particular predictions are imperfect: their exact Bayesian targets are approximately 72.3% and 27.7%. Learning the inference procedure does not guarantee exact inference or calibration.'
  },
  {
    chapter:inside, kind:'mechanism', mechanismStage:7,
    title:'First, turn each labeled row into a vector.',
    body:'In our teaching model, an MLP maps bear type, strategy, and survival into 32 numbers. A separate MLP encodes the query’s two known features.',
    equation:'h_i=g_\\theta(B_i,L_i,Y_i)\\in\\mathbb R^{32}',
    note:'Actual stored activations; only the first two coordinates are shown. The query encoder receives no survival label.',
    reading:'The displayed row is Black + Fight + Survived, encoded as (0,0,1). The same context encoder is applied to every labeled row. Its nonlinear transformations allow the resulting vector to depend jointly on features and outcome; the coordinates have no assigned human-readable meanings. This small model encodes whole rows. Molnar’s nanoTabPFN explanation instead embeds cells and alternates attention across rows and columns. The details differ, while both designs provide learned representations through which the context can affect a query.'
  },
  {
    chapter:inside, kind:'mechanism', mechanismStage:8,
    title:'Attention combines the context vectors.',
    body:'The query produces a query vector; each labeled row produces a key and a value vector. Softmax scores determine the weighted sum of those values.',
    equation:'c=\\sum_{i=1}^{4}a_i v_i,\\qquad\\sum_i a_i=1',
    note:'Actual attention weights and value coordinates from this teaching model. The values are learned vectors, not raw survival labels.',
    reading:'Linear projections of the encoded query and context produce Q, K, and V. With width 32, the scores are Q·K_i/√32; softmax converts them into nonnegative weights a_i summing to one. The pooled representation is c = Σ a_i V_i. The diagram shows the first two coordinates of each actual 32-dimensional value vector and verifies their weighted sum. The decoder also retains the query representation. The weights in this example are fairly even: useful information is spread across the context, so a sharply focused lookup is not required. These attention weights describe a computation in this particular model; they are not posterior probabilities over worlds or an explanation of feature importance.'
  },
  {
    chapter:inside, kind:'mechanism', mechanismStage:9,
    title:'Decode that representation into a probability.',
    body:`The decoder combines the query vector with the attention output. Its logit ${A.logit.toFixed(3)} becomes a survival probability of ${pct(A.probability)} through the sigmoid function.`,
    equation:`\\sigma(${A.logit.toFixed(3)})\\approx ${A.probability.toFixed(3)}`,
    note:'The entire input-to-probability calculation uses fixed learned weights. The next section derives its ideal Bayesian target.',
    reading:'Concatenating the 32-dimensional query representation and the 32-dimensional attention output gives the decoder 64 inputs. A hidden ReLU layer followed by a scalar output produces the logit; sigmoid maps it into (0,1). During pretraining, query-label error could flow backward through this full computation. During inference, there is only the forward calculation. The network was trained to predict outcomes; it was not trained to reproduce explicit posterior weights. Under the task generator, the log-loss-optimal answer is nevertheless the posterior predictive probability.'
  }
];
