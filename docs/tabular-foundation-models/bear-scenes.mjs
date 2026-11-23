import {bearResults} from './bear-results.mjs?v=20261123-context-ten-noise';
const diagnostics=bearResults.cases.filter(row=>row.depth===2);
const continuous=diagnostics.find(row=>row.id==='continuous_noise');
const continuousAucs=bearResults.records.filter(row=>row.depth===2&&row.id==='continuous_noise').map(row=>row.auc);
const diagnosticOrder=['signal_only','continuous_noise','binary_noise','two_bins','interaction_feature'];
const reportedMeans=diagnosticOrder.map(id=>diagnostics.find(row=>row.id===id).mean_auc.toFixed(3)).join(', ');
const reportedSDs=diagnosticOrder.map(id=>diagnostics.find(row=>row.id===id).sd_auc.toFixed(3)).join(', ');
const chapter = 'Why the interaction matters';
export const bearScenes = [
  {
    chapter, kind:'bear', bearStage:0,
    title:'Return to the distractor columns.',
    body:'Our miniature PFN varied the rule but always used the same two features. Return to world A and the ten distractors from our opening comparison. Bear type and strategy still have no marginal signal.',
    equation:'P(Y=1\\mid B)=P(Y=1\\mid L)=\\tfrac12',
    note:'Changing which rule applies and discovering which columns matter are different tasks. The miniature PFN demonstrated the first.',
    reading:'The miniature PFN was trained to distinguish two opposed rules using two fixed, relevant feature columns. It was not trained or tested on finding those columns among distractors. We now generate data from world A with both relevant features independently Bernoulli(0.5). The target is Y = 1[−1 + 2·1(B = L) + ε > 0], with independent standard logistic noise ε. Ten independent Uniform(0,1) columns contain no information about survival. The random outcome noise and the irrelevant input columns play different roles: even a predictor that knows the correct rule cannot predict every outcome. The opening tree, forest and TabICLv2 comparison already measured this difficulty. This separate XGBoost experiment now investigates its split-search mechanism.'
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
    title:'Compare the fitted runs.',
    body:'Reveal the individual seeds first, then their means. Compare the ten continuous distractors with binary distractors, coarser histogram bins, and an added interaction feature.',
    equation:'\\operatorname{AUC}_{\\rm oracle}=\\operatorname{logistic}(1)\\approx0.7311',
    note:'12 seeds; paired lines join continuous and binary noise on the same split. Depth 2, 200 trees, learning rate 0.1; 8,000 train and 2,000 test rows.',
    reading:`Each dot is one measured test AUC from XGBoost ${bearResults.metadata.versions.xgboost}. The same seed uses identical sampled bear types, strategies, survival outcomes, and train–test splits across all five conditions. With ten continuous distractors, test AUC ranges from ${Math.min(...continuousAucs).toFixed(3)} to ${Math.max(...continuousAucs).toFixed(3)}, with mean ${continuous.mean_auc.toFixed(3)}. Light lines connect each continuous-noise result to its binary-noise counterpart. Mean markers appear after the individual results. The comparisons binarize all ten distractors at 0.5, reduce max_bin from 256 to 2, or add the product B·L alongside the original inputs. The five means, ordered as signal only, continuous noise, binary noise, two bins, and added product, are ${reportedMeans}; their between-seed standard deviations are ${reportedSDs}. The dashed line is the population AUC of the oracle that ranks matching strategies above mismatching strategies: p² + p(1−p) = p, where p = logistic(1). It is not an upper bound on finite-test-sample AUC; sample results can lie above it. These are fixed-setting diagnostics with no tuning against the test set and no TabPFN evaluation. The Nature 2025 inference pipeline also offers optional products of up to 50 randomly selected feature pairs, disabled by default. This connects to our B·L repair, but this optional feature construction is not part of the paper’s default benchmark. Feature grouping into tokens and explicit product construction are different operations.`, references:[{title:'Nature 2025 · Inference details: optional second-order features',url:'https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/'}]
  },
  {
    chapter, kind:'bear', bearStage:5,
    title:'Does TabPFN solve the harder task?',
    body:'Our opening test ran TabICLv2. The Nature paper studies TabPFN; its training and shuffled-feature experiments motivate testing that checkpoint too.',
    equation:'q_\\theta(Y_*\\mid B_*,L_*,Z_*,D)',
    note:'Both useful bear features also have zero marginal signal. Discarding columns by marginal association would discard the solution.',
    reading:'The opening comparison measured TabICLv2, which is a different pretrained model from TabPFN. The XGBoost results diagnose unstable split search and show that particular changes repair it. The miniature PFN only learned to distinguish two rules on fixed relevant columns. The Nature 2025 paper supplies a more specific reason to test the harder problem: its synthetic generator can contain disconnected graph components that produce uninformative features, and Fig. 5 evaluates robustness by adding randomly shuffled copies of original features. Our independent Uniform(0,1) distractors are a different intervention. In the bear task, B and L each have no marginal association with Y, while their combination is predictive. Handling irrelevant columns must therefore allow useful combinations to survive; univariate screening would miss the signal. Pretraining and table attention make joint-pattern inference plausible, but they do not establish that a particular checkpoint succeeds here or identify its internal algorithm. Run that checkpoint and XGBoost on identical datasets, splits, and metrics. The diagram gives the predictor all columns without labeling which ones matter.', references:[{title:'Nature 2025 · Fig. 5 and Methods: Graph structure sampling',url:'https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/'}]
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
