const chapter = 'Why the interaction matters';
export const bearScenes = [
  {
    chapter, kind:'bear', bearStage:0,
    title:'Now add irrelevant columns.',
    body:'Our miniature PFN varied the rule but always used the same two features. Return to world A and add three irrelevant columns. Bear type and strategy still have no marginal signal.',
    equation:'P(Y=1\\mid B)=P(Y=1\\mid L)=\\tfrac12',
    note:'Changing which rule applies and discovering which columns matter are different tasks. The miniature PFN demonstrated the first.',
    reading:'The miniature PFN was trained to distinguish two opposed rules using two fixed, relevant feature columns. It was not trained or tested on finding those columns among distractors. We now generate data from world A with both relevant features independently Bernoulli(0.5). The target is Y = 1[−1 + 2·1(B = L) + ε > 0], with independent standard logistic noise ε. Three independent Uniform(0,1) columns contain no information about survival. The random outcome noise and the irrelevant input columns play different roles: even a predictor that knows the correct rule cannot predict every outcome. This XGBoost experiment introduces the additional difficulty of finding an interaction among irrelevant columns.'
  },
  {
    chapter, kind:'bear', bearStage:1,
    title:'The combination reveals the pattern.',
    body:'Survival is more likely when the response matches the bear type. The direction of the strategy effect reverses across bear types.',
    equation:'P(Y=1\\mid B,L)=\\begin{cases}0.7311 & B=L\\\\0.2689 & B\\ne L\\end{cases}',
    note:'These are population probabilities from the simulation, not fitted predictions. The animation reveals them from the marginal 50%.',
    reading:'The four equally likely feature combinations form a noisy equality rule, or XNOR. Brown-and-lie and black-and-fight have survival probability logistic(1) ≈ 0.7311. The other two combinations have probability logistic(−1) ≈ 0.2689. Averaging either column or row gives exactly 0.5. The two predictors are jointly informative even though each is individually independent of survival.'
  },
  {
    chapter, kind:'bear', bearStage:2,
    title:'A small tree can express the rule.',
    body:'Split on bear type, then on strategy. The second split reveals the signal, but the first split alone leaves both groups at 50%.',
    equation:'\\Delta_{\\mathrm{first}}=0,\\qquad\\Delta_{\\mathrm{joint}}>0',
    note:'A representable tree is not necessarily the tree found by greedy training. Sample imbalance can provide a useful first split.',
    reading:'A depth-two tree can represent the four conditional probabilities exactly. XGBoost constructs its trees through greedy split selection, which evaluates immediate improvement rather than the benefit of a subsequent pair of splits. At a constant initial prediction, neither signal feature offers a population-level gain. This is a difficulty of discovering the interaction, not a claim that trees cannot represent interactions. Sampling variation, deeper trees, and later boosting rounds can help.'
  },
  {
    chapter, kind:'bear', bearStage:3,
    title:'Noise offers more places to look.',
    body:'A binary feature has one possible cut. A continuous distractor offers many thresholds, giving the split search more opportunities to find an accidental training improvement.',
    equation:'\\max_{j,t}\\;\\operatorname{Gain}(x_j\\le t)',
    note:'Threshold illustration is schematic. XGBoost’s histogram method limits candidate thresholds through max_bin.',
    reading:'All irrelevant features have zero population predictive value, but their empirical split gains need not be zero. Maximizing over many candidate cutpoints can select a chance association. The useful binary variables have only one possible partition each. This combines with the interaction’s absent marginal signal to make continuous distractors particularly troublesome. The diagram does not show fitted split gains or attention weights.'
  },
  {
    chapter, kind:'bear', bearStage:4,
    title:'The average hides unstable fits.',
    body:'With continuous distractors, some runs approach chance while others approach the oracle. Reveal the individual seeds first, then their means. The repairs make performance much more consistent.',
    equation:'\\operatorname{AUC}_{\\rm oracle}=\\operatorname{logistic}(1)\\approx0.7311',
    note:'12 seeds; paired lines join continuous and binary noise on the same split. Depth 2, 200 trees, learning rate 0.1; 8,000 train and 2,000 test rows.',
    reading:'Each dot is one measured test AUC from XGBoost 3.0.5. The same seed uses identical sampled bear types, strategies, survival outcomes, and train–test splits across all five conditions. Continuous-noise AUC ranges from about 0.488 to 0.751, so its mean of 0.651 conceals near-chance and near-oracle fits. Light lines connect each continuous-noise result to its binary-noise counterpart. Mean markers appear after the individual results. Binarizing distractors at 0.5, reducing max_bin from 256 to 2, or adding the product B·L alongside the original inputs makes performance much more consistent. The five means are approximately 0.730, 0.651, 0.730, 0.729, and 0.728. The dashed line is the population AUC of the oracle that ranks matching strategies above mismatching strategies: p² + p(1−p) = p, where p = logistic(1). It is not an upper bound on finite-test-sample AUC; sample results can lie above it. These are fixed-setting diagnostics with no tuning against the test set and no TabPFN evaluation.'
  },
  {
    chapter, kind:'bear', bearStage:5,
    title:'Does TabPFN solve the harder task?',
    body:'Our miniature PFN learned which rule applies to two fixed columns. Finding an interaction among irrelevant columns requires another capability. A matched TabPFN comparison must establish that advantage.',
    equation:'q_\\theta(Y_*\\mid B_*,L_*,Z_*,D)',
    note:'The diagram is a hypothesis about joint-pattern inference. Neither a TabPFN advantage nor its internal mechanism is measured here.',
    reading:'The XGBoost results diagnose unstable split search and show that particular changes repair it. The miniature PFN experiment demonstrated a different but narrower skill: recognizing which of two rules fits a table with the same two relevant columns every time. Its success does not establish robustness to irrelevant columns. Molnar’s account of varied synthetic task priors gives a reason to investigate whether TabPFN can infer useful interactions while discounting distractors. Test a specified TabPFN checkpoint and XGBoost on the same datasets, splits, and metrics before claiming an advantage. If an advantage is observed, joint-pattern inference is one possible explanation; these experiments and attention weights alone would not identify its internal cause.'
  },
  {
    chapter, kind:'bear', bearStage:6,
    title:'Remove one more clue.',
    body:'Here, both useful columns are binary and every distractor is continuous. Compare TabPFN on both versions, including binary distractors, so feature type no longer identifies the useful columns.',
    equation:'Z_j\\sim U(0,1)\\quad\\longrightarrow\\quad\\mathbf{1}[Z_j>0.5]',
    note:'The XGBoost binary-noise result is measured. Repeating the comparison with TabPFN is a proposed next test.',
    reading:'Feature type is an additional cue in the original design. When all columns are independent balanced binary variables, being binary no longer singles out the two useful predictors. This control changes both the available split thresholds and the datatype cue, so it cannot by itself isolate a single internal mechanism. Compare the same TabPFN checkpoint and XGBoost on the same splits, with both continuous and binary distractors and several sample sizes. The current evidence establishes the XGBoost recovery; it does not establish the TabPFN result for this control.'
  }
];
