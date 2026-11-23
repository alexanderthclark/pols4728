const paper = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/';
const chapter = 'From our Mini-PFN to TabPFN';

export const natureScenes = [
  {
    chapter, kind:'nature', natureStage:0,
    title:'Sample a rule, then sample a whole table.',
    body:'One rule generates a whole table. Here, B and L jointly determine survival probabilities; disconnected Z columns add irrelevant variation. TabPFN repeats this procedure with many different graphs and functions.',
    equation:'Y=\\mathbf{1}\\{U<p(B,L)\\},\\qquad U\\sim\\mathrm{Uniform}(0,1)',
    note:'Original teaching analogue of the paper’s generator, not a task extracted from TabPFN pretraining. B = Brown; L = Lie down; Y = survived.',
    reading:'The Nature paper samples structural causal models: directed graphs with random functions and noise. Its generator includes small neural functions, decision-tree mappings and discretization; disconnected graph components can supply irrelevant features. One sampled mechanism generates multiple rows before another task is drawn. Our diagram substitutes the familiar bear rule to make that hierarchy visible. B and L affect Y jointly, although each alone has no population association with Y in our balanced example. All ten independent Z columns remain irrelevant even jointly; the diagram shows the first and last, omitting eight. U supplies the outcome randomness; the displayed binary labels are illustrative draws, not survival probabilities. Our Mini-PFN varied only which of two rules applied to two fixed relevant features. It did not learn to find those features among distractors.',
    references:[
      {title:'Nature 2025 · Fig. 2: synthetic task generation',url:`${paper}figure/Fig2/`},
      {title:'Nature 2025 · Methods: generating synthetic datasets',url:paper}
    ]
  },
  {
    chapter, kind:'nature', natureStage:1,
    title:'Combine features, then compare encounters.',
    body:'Feature attention combines columns within each encounter. Sample attention then compares corresponding representations across encounters. The query reads labeled context while its own survival outcome stays hidden.',
    equation:'\\text{feature attention}\\;\\longrightarrow\\;\\text{sample attention}',
    note:'Schematic information flow, not measured attention. The paper groups two features per token by default; columns are separated here for clarity, with eight noise columns omitted.',
    reading:'Figure 1 describes a two-dimensional table transformer. Feature attention mixes the representations within an example, and sample attention exchanges information across examples for the same feature or feature group. Training rows attend to training rows. Test queries read training context; they do not send their unknown labels into it or attend to other test queries. The diagram traces one query’s incoming context paths, with its target withheld throughout. These operations recur across transformer layers. This differs from our educational Mini-PFN, which first embeds each entire row with an MLP and then uses one cross-attention operation. The real architecture has a route for combining columns and comparing examples, but this diagram alone does not establish how a trained TabPFN handles the bear distractors.',
    references:[
      {title:'Nature 2025 · Fig. 1: table transformer',url:`${paper}figure/Fig1/`},
      {title:'Nature 2025 · Methods: architecture and masking',url:paper}
    ]
  },
  {
    chapter:'Evidence from the paper', kind:'nature', natureStage:2,
    title:'The paper finds a concrete small-data advantage.',
    body:'In Figure 5a, TabPFN reaches roughly the same aggregate predictive performance with half the training rows as default CatBoost uses with all of them. This compares sample efficiency across benchmark datasets.',
    equation:'\\begin{gathered}\\text{similar aggregate performance}\\\\\\text{with about half the rows}\\end{gathered}',
    note:'Approximate reading of Fig. 5a, not a raw AUC comparison or a guarantee for every dataset. The baseline here is default CatBoost, not XGBoost.',
    reading:'Figure 5a varies the training-data fraction and aggregates predictive scores normalized separately within each dataset. Its reported half-data comparison is against default CatBoost. The main benchmark contains 29 classification and 28 regression datasets, with at most 10,000 rows and 500 features; classification has at most 10 classes. These results motivate testing TabPFN when observations are scarce. They do not establish the size of any advantage on our bear problem, or show that every dataset needs exactly half as many labels. A matched empirical TabPFN–XGBoost comparison on the bear simulation remains necessary.',
    references:[
      {title:'Nature 2025 · Fig. 5a: sample efficiency',url:`${paper}figure/Fig5/`},
      {title:'Nature 2025 · Benchmark design and evaluation',url:paper}
    ]
  }
];
