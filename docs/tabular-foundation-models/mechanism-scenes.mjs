import {miniPfnResults as run} from './mini-pfn-results.mjs';
const A=run.examples.A, B=run.examples.B;
const update=run.first_update;
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
    note:'Actual first task in the first training minibatch. The hidden query answer and generating rule stay outside the input.',
    reading:'These are the recorded context and query from the first task in the first training minibatch, when the network still has its initial random weights. The query is Brown + Lie down. Each labeled row supplies raw numeric bear, strategy, and survival indicators; the query supplies only the two feature indicators. Repeated bear–strategy pairs can have different outcomes because survival is noisy. The neural network must turn the whole prediction exercise into one survival probability. The displayed query outcome is held outside the network input and used only to score its prediction.'
  },
  {
    chapter:training, kind:'mechanism', mechanismStage:2,
    title:'Average the losses. Update the weights.',
    body:`The first sampled query survived. After one Adam step using ${update.batch_size} tasks, its prediction changes from ${pct(update.first_query_probability_before)} to ${pct(update.first_query_probability_after)}.`,
    equation:`\\overline\\ell=\\frac1{256}\\sum_t\\ell_t:\\quad ${update.batch_log_loss_before.toFixed(4)}\\longrightarrow${update.batch_log_loss_after.toFixed(4)}`,
    note:'Recorded before/after values for one update. The mean losses use the same 256 tasks; the displayed probability belongs to one task.',
    reading:`Before the first update, the first query has predicted survival probability ${update.first_query_probability_before.toFixed(6)} and observed outcome Y*=1. Its individual log loss is −log(q) ≈ ${f(-Math.log(update.first_query_probability_before))}. Training averages the losses of all ${update.batch_size} tasks in this minibatch, giving ${update.batch_log_loss_before.toFixed(6)}. Backpropagation differentiates that mean through the decoder, attention projections, and encoders, and Adam updates their shared parameters. Re-evaluating the same minibatch after the update gives mean loss ${update.batch_log_loss_after.toFixed(6)}; the first query’s probability is then ${update.first_query_probability_after.toFixed(6)}. Both quantities are recorded measurements. A single task’s probability is not the minibatch objective, and improving the average need not improve every individual prediction. Scroll switches between the measured before/after states; it does not interpolate fictitious training steps.`
  },
  {
    chapter:training, kind:'mechanism', mechanismStage:3,
    title:'Each task brings another table.',
    body:'Here is the second sampled task from that same minibatch. Its context and query differ. Each task contributes one loss to the shared update.',
    equation:'(D_t,x_{*,t},Y_{*,t})\\sim\\text{task generator}',
    note:'Two of the 256 tasks in one minibatch. A new world is drawn once per task; its identity is never an input.',
    reading:'The second table is reproduced from the same seeded generator call as the first: training seed 4729, batch size 256, task index 1. Its query is Black + Lie down and its sampled query outcome is survival. Context labels are available to the model; the query answer remains outside the input. The two displayed tasks enter the same minibatch update, not separate weight updates. Every task in the minibatch has its own independently drawn world, shared by all rows within that task. Pretraining repeatedly draws fresh minibatches. The final trained checkpoint and its predictions appear only after this training sequence.'
  },
  {
    chapter:inside, kind:'mechanism', mechanismStage:4,
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
    body:`Stack the 32-number query vector and the 32-number attention output into 64 inputs. The decoder’s logit ${A.logit.toFixed(3)} becomes ${pct(A.probability)} survival through sigmoid.`,
    equation:'z=\\operatorname{MLP}_{\\theta}([h_*;c]),\\qquad q=\\sigma(z)',
    note:'Concatenation, not addition: [query vector; pooled context] has 64 coordinates. The next section derives the ideal Bayesian target.',
    reading:'The saved model concatenates the 32-dimensional query representation h* first and the 32-dimensional attention output c second. Thus [h*;c] is a 64-dimensional vector; the two vectors are not added coordinate by coordinate. A hidden ReLU layer followed by a scalar output produces the logit; sigmoid maps it into (0,1). During pretraining, query-label error could flow backward through this full computation. During inference, there is only the forward calculation. The network was trained to predict outcomes; it was not trained to reproduce explicit posterior weights. Under the task generator, the log-loss-optimal answer is nevertheless the posterior predictive probability.'
  }
];

// Keep renderer stage IDs stable while placing the fixed-checkpoint evaluation
// after pretraining and the explicit freeze transition.
const chronologicalOrder=[0,1,3,2,5,6,4,7,8,9];
mechanismScenes.sort((a,b)=>chronologicalOrder.indexOf(a.mechanismStage)-chronologicalOrder.indexOf(b.mechanismStage));
for(const scene of mechanismScenes){
  const phase=scene.chapter===training
    ? 'Pretraining: shared weights change after minibatch updates.'
    : 'Inference: weights stay fixed as the context changes.';
  scene.note=`${phase} ${scene.note}`;
}
