import {bearComparisonResults as results} from './bear-comparison-results.mjs?v=20261123-context-ten-noise';

const chapter='Try the bear task';
const noiseCount=results.metadata.n_noise;
const featureCount=noiseCount+2;
const protocol=`Same ${results.metadata.n_train.toLocaleString('en-US')} training and ${results.metadata.n_test.toLocaleString('en-US')} test rows per seed; ${results.metadata.seeds.length} simulation seeds. Small marks show individual runs. Dashed: population oracle AUC .731.`;
const common=`Bear type and response are independent balanced binary variables. Outcomes follow the depth-two rule just shown, with survival probabilities logistic(1) and logistic(−1). Every input table also contains ${noiseCount} independent Uniform(0,1) columns that carry no information about survival. Within each seed, all three learners share the same outcomes, ${featureCount} feature columns and train–test split. Models and settings are fixed before evaluating test labels. Small marks show each of the 12 measured test ROC AUCs in this continuous-noise condition; the large marker and number show their mean. The dashed reference is population oracle AUC logistic(1), not a ceiling for every finite test set. `;

export const bearComparisonScenes=[
  {
    chapter,kind:'bearComparison',comparisonStage:0,
    title:'Can a fitted depth-two tree find the rule?',
    body:`The true rule fits in two levels. Fit a decision tree using bear type, response, and ${noiseCount} continuous noise columns.`,
    equation:'\\texttt{DecisionTreeClassifier(max\\_depth=2)}',
    note:protocol,
    reading:common+'A single CART tree uses greedy split selection: at each node it chooses a split with the largest immediate impurity reduction. Neither signal variable has a population marginal association with survival, even though the pair predicts survival. The correct interaction can therefore be representable at depth two while still being difficult for greedy fitting to discover among continuous distractors.'
  },
  {
    chapter,kind:'bearComparison',comparisonStage:1,
    title:'Now fit a random forest.',
    body:`Average 500 fully grown trees. At each split, randomly consider three of the ${featureCount} columns.`,
    equation:'\\texttt{RandomForestClassifier(n\\_estimators=500)}',
    note:protocol,
    reading:common+'The random forest uses 500 trees, bootstrap sampling, square-root feature subsampling at each split (three of twelve candidate columns), no maximum depth, and a minimum leaf size of one. These are fixed settings, not an optimized forest. Averaging and feature subsampling give it a different route to the interaction from the single depth-two tree. Its empirical performance must be checked rather than inferred from the data-generating tree.'
  },
  {
    chapter,kind:'bearComparison',comparisonStage:2,
    title:'Now try a tabular foundation model.',
    body:'TabICLv2 gets the same training encounters and the same distractors. These are measured predictions from a pretrained model.',
    equation:'\\texttt{fit}(X_{\\rm train},y_{\\rm train})\\quad\\texttt{predict\\_proba}(X_{\\rm test})',
    note:protocol,
    reading:common+'TabICLv2 is an open-source pretrained tabular foundation model, distinct from the miniature teaching PFN and the TabPFN model in the Nature paper. This benchmark runs the recorded checkpoint using four preprocessing ensemble members and fixed settings. Its pretrained network weights are held fixed; the labeled training encounters supply context. These measurements establish what this fitted procedure does on this simulation. They do not measure its attention or establish a general advantage across datasets.'
  }
];
