import {miniPfnResults as run} from './mini-pfn-results.mjs';
import {contextState, observedContext, reversedContext, MATCH_SURVIVAL} from './bayes.mjs';
const A=contextState(observedContext),B=contextState(reversedContext);
const pct=x=>`${(100*x).toFixed(1)}%`;
const chapter='The exact Bayesian target';

export const bayesScenes=[
  {
    chapter,kind:'bayes',bayesStage:0,
    title:'Return to the same four encounters.',
    body:'The network used this table without being told its world. Now compute the ideal prediction: start with equal prior weight on A and B, then incorporate these same outcomes.',
    equation:'p(A)=p(B)=\\tfrac12,\\qquad p=\\operatorname{logistic}(1)',
    note:'A assigns 73.1% survival to Black + Fight and Brown + Lie down. B reverses all four probabilities.',
    reading:'We return to exactly the A-pattern context used in the neural calculation: Black + Fight + Survived; Black + Lie down + Did not survive; Brown + Fight + Did not survive; Brown + Lie down + Survived. The query remains Brown + Lie down. The displayed first state masks the outcomes while showing each world’s conditional survival probabilities. Both worlds initially have probability one half. The feature distribution is identical under the two worlds, so the feature values alone do not change those prior odds. Outcomes are independent conditional on the fixed features and the shared world.'
  },
  {
    chapter,kind:'bayes',bayesStage:1,
    title:'The first survival favors A.',
    body:'The first encounter is Black + Fight + Survived. That observed outcome has probability 73.1% under A and 26.9% under B. The likelihood ratio is e.',
    equation:'p(A\\mid D_1)=\\frac{p}{p+(1-p)}\\approx0.7311',
    note:'Update the probability of the world using the probability of the observed outcome.',
    reading:'For this matching pair, world A predicts survival with probability p and B with probability 1 − p. Their ratio is p/(1 − p) = exp(1). Equal prior odds therefore become e to one, and the posterior probability of A is logistic(1). The posterior concerns which rule generated the whole table. It is not yet the survival probability of the query.'
  },
  {
    chapter,kind:'bayes',bayesStage:2,
    title:'The two deaths favor A too.',
    body:'A predicts low survival for Black + Lie down and Brown + Fight. Their observed deaths therefore support A. All four observed outcomes are more likely under A.',
    equation:'\\begin{gathered}p(D\\mid A)/p(D\\mid B)=[p/(1-p)]^4=e^4\\\\p(A\\mid D)\\approx0.9820\\end{gathered}',
    note:'Use P(the observed outcome | world): for a death, this is one minus that world’s survival probability.',
    reading:'Under A, each displayed outcome has probability p: survival in the two matching pairs, and death in the two mismatching pairs. Under B, each has probability 1 − p. Conditional independence gives likelihoods p⁴ and (1 − p)⁴. With equal prior odds, posterior odds are exp(4), about 54.6 to one; the posterior probability of A is logistic(4), about 98.2%. A death is not automatically evidence for B. Its evidential direction depends on the encounter’s features and what each world predicts.'
  },
  {
    chapter,kind:'bayes',bayesStage:3,
    title:'A world weight is not a survival probability.',
    body:'Use the same posterior weights to predict either query. Brown + Lie down gives 72.3% survival; Brown + Fight gives 27.7%. Changing the query does not change the evidence already observed.',
    equation:'p(Y_*=1\\mid x_*,D)=\\sum_{\\phi\\in\\{A,B\\}}p(Y_*=1\\mid x_*,\\phi)\\,p(\\phi\\mid D)',
    note:'Weight each world’s conditional prediction, then add. Even knowing the world would leave outcome noise.',
    reading:`The exact posterior probability of A is ${A.posterior[0].toFixed(6)}. For Brown + Lie down, multiply that weight by p and the remaining weight by 1 − p, yielding ${A.prediction.toFixed(6)}. For Brown + Fight, each world’s conditional prediction reverses, so the result is ${(1-A.prediction).toFixed(6)}. The world probabilities stay fixed because only the query features changed. More certainty about A would move these probabilities toward p ≈ ${MATCH_SURVIVAL.toFixed(4)} and 1 − p, not toward one and zero. This separates uncertainty about the rule from randomness within the rule.`
  },
  {
    chapter,kind:'bayes',bayesStage:4,
    title:'Compare the learned answer with its exact target.',
    body:`For the two contexts, exact Bayesian prediction gives ${pct(A.prediction)} and ${pct(B.prediction)}. Our trained Mini-PFN gives ${pct(run.examples.A.probability)} and ${pct(run.examples.B.probability)}. It learned an approximation.`,
    equation:'q_\\theta(Y_*=1\\mid x_*,D)\\approx p(Y_*=1\\mid x_*,D)',
    note:'Measured neural outputs versus exact calculations under the same prior. These are not TabPFN benchmark results.',
    reading:`The A-pattern context is [1,0,0,1] in the displayed feature-row order; reversing those labels gives the B-pattern context [0,1,1,0]. The query is Brown + Lie down in both. Exact inference gives ${A.prediction.toFixed(6)} and ${B.prediction.toFixed(6)}, while the one fixed trained checkpoint gives ${run.examples.A.probability.toFixed(6)} and ${run.examples.B.probability.toFixed(6)}. An exact mixture of these worlds must lie between 26.9% and 73.1%. The neural network is not constrained to that interval: its 74.0% and 23.6% predictions are approximation errors, not a removal of outcome noise or an additional possible world. The weights were optimized to predict sampled outcomes, not supplied with Bayesian posterior weights.`
  },
  {
    chapter,kind:'bayes',bayesStage:5,
    title:'Why does log loss favor the Bayesian answer?',
    body:'For this context and query, the next outcome is Bernoulli(r), with r = 72.3%. Expected log loss is minimized by predicting q = r. Pretraining learns this mapping across its task distribution.',
    equation:'\\begin{aligned}r&=p(Y_*=1\\mid x_*,D)\\\\L(q)&=-r\\log q-(1-r)\\log(1-q)\\\\q^*&=r\\end{aligned}',
    note:'This is the population target under the task prior. Finite capacity, finite training, and optimization error can prevent the network from reaching it.',
    reading:'Let r = p(Y*=1 | x*,D), the posterior predictive probability induced by the task generator. Conditional expected binary log loss is −r log(q) − (1 − r) log(1 − q). Its derivative is (q − r)/(q(1 − q)), which is negative below r and positive above it. Equivalently, the loss is the binary entropy H(r) plus KL(Bernoulli(r) || Bernoulli(q)); the extra term is minimized at zero when q = r. For the displayed A-pattern context, r is approximately 0.7227468. The moving square is a candidate prediction q, not a recorded neural training trajectory. A single observed survival would favor q near one, but averaging the training objective over random query outcomes favors r. Averaging again over tasks makes the desired function the posterior predictive mapping for that prior. This characterization does not guarantee that a finite network or optimizer finds the population optimum, or that its pretraining prior matches a new real dataset.'
  }
];
