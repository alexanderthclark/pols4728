const paper='https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/';
const tabicl='https://arxiv.org/html/2602.11139v1#S5';
const tabdpt='https://proceedings.neurips.cc/paper_files/paper/2025/file/fc0e3f908a2116ba529ad0a1530a3675-Paper-Conference.pdf';
const chapter='Broad pretraining, then a new task';
export const pretrainingScenes=[
  {
    chapter,kind:'pretraining',pretrainingStage:0,
    title:'Beyond our two-rule teaching model.',
    body:'Structural causal models (SCMs) generate many different synthetic tables. Now we move beyond the teaching model; bears become a new prediction task.',
    equation:'\\text{many mechanisms}\\longrightarrow\\text{many tables}',
    note:'Illustrative SCMs. TabPFN 2025 and TabICLv2 use synthetic generators; TabDPT uses curated real tables, including OpenML datasets.',
    reading:'Our miniature PFN practiced only two bear rules. Production models use much broader task distributions. The 2025 TabPFN generates synthetic tables from sampled structural causal models, varying graph structure, functions, noise, and observed feature and target variables. TabICLv2 also pretrains on synthetic tables, with its own broader generator and different noise choices. These four small SCMs illustrate the recipe; they are not recovered samples from either checkpoint’s pretraining run. Synthetic data is not the only source: the TabDPT paper forms prediction tasks from curated real OpenML tables, including tasks that select a different column as the target. Each model has its own data recipe; the source strip does not imply they all combine both sources.',
    references:[{title:'TabPFN 2025 · Synthetic data based on causal models',url:paper},{title:'TabICLv2 · Synthetic data prior',url:tabicl},{title:'TabDPT · Real-data pretraining, §§3.2–4.1',url:tabdpt}]
  },
  {
    chapter,kind:'pretraining',pretrainingStage:1,
    title:'A table becomes a prediction exercise.',
    body:'Choose features and a target. Supply labeled context rows and query features; keep the query answers outside the network’s input.',
    equation:'q_\\theta(y_Q\\mid X_Q,X_C,y_C)',
    note:'C = context rows; Q = query rows. qθ is the model’s predictive distribution. The generator knows query answers; the model does not see them.',
    reading:'First generate a table, or select a real table. Select observed feature columns and a target, then divide rows into labeled context C and queries Q. The network receives X_C, y_C, and X_Q. It predicts the withheld targets y_Q. Here qθ denotes its predicted conditional distribution; for binary labels, qθ(Y=1 | inputs) is its predicted probability of class 1. The true query targets are kept outside the input and used only to score predictions during pretraining. One mechanism is shared by the rows in a task, so its labeled context contains relevant evidence about its queries. Across tasks, schemas, functions, and targets can differ. Class code 1 need not mean survival, and the SCM graph is not supplied to the predictor.',
    references:[{title:'TabPFN 2025 · Pretraining tasks and target generation',url:paper},{title:'TabDPT · Self-supervised context/query tasks',url:tabdpt}]
  },
  {
    chapter,kind:'pretraining',pretrainingStage:2,
    title:'Repeat across tables. Learn shared weights.',
    body:'Compare predictions with withheld answers, combine errors across tasks, and update one network. Repeat with fresh tables from the pretraining distribution.',
    equation:'\\text{predict}\\longrightarrow\\text{check}\\longrightarrow\\text{update }\\theta',
    note:'q₁, q₂, q₃ are predicted probabilities of class 1. Withheld targets score those predictions; pretraining updates the shared weights θ.',
    reading:'Each prediction exercise contributes a loss. For classification, cross-entropy rewards high predicted probability for the observed query label. Training combines losses across tasks and uses gradients to update the shared network parameters. Repeated tasks teach a reusable prediction procedure, rather than one set of coefficients for a single pooled table. Each new task brings its own labeled context and queries. The choice of generators or real-table collection determines what relationships the network practices. The diagram shows this workflow, not another measured training run: no numerical model prediction, loss reduction, or checkpoint is invented for these illustrative SCMs.',
    references:[{title:'TabPFN 2025 · Training details',url:paper},{title:'TabICLv2 · Pretraining setup',url:'https://arxiv.org/html/2602.11139v1#S4.SS1'}]
  },
  {
    chapter,kind:'pretraining',pretrainingStage:3,
    title:'Now the bear table arrives as context.',
    body:'The public TabICLv2 checkpoint was pretrained on a broad synthetic prior. It now predicts our new bear task using labeled encounters as context.',
    equation:'\\theta\\;\\text{fixed},\\qquad q_\\theta(Y_*=1\\mid x_*,D_{\\rm bear})',
    note:'qθ is predicted survival probability. AUC summarizes the measured test-set ranking; it is not a probability for this one query.',
    reading:'This is the model measured in the opening comparison, with unchanged pretrained weights. Our independently simulated bear dataset is its new context, not the corpus on which we pretrained that checkpoint. Each run supplies 8,000 labeled encounters with bear type, response, and ten distractors, then evaluates predictions on 2,000 held-out encounters. The displayed mean ROC AUC comes from the saved twelve-seed benchmark; the diagram does not invent a probability for its illustrative new encounter. Broad synthetic pretraining can therefore be evaluated on a new task whose variables carry a domain meaning absent from the anonymous pretraining tables. Next we return explicitly to the earlier, restricted Mini-PFN checkpoint to inspect an actual numerical forward calculation; that small model was not retrained on these broad SCMs.',
    references:[{title:'Measured bear comparison · Reproduction and checkpoint provenance',url:'https://github.com/alexanderthclark/pols4728/blob/main/examples/tabular-foundation-models/README-bear-comparison.md'},{title:'TabICLv2 · Pretraining distribution',url:tabicl}]
  }
];
