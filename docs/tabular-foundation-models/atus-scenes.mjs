const chapter = 'Try it on our data';

export const atusScenes = [
  {
    chapter, kind:'atus', atusStage:0,
    title:'Pretraining does not guarantee a win.',
    body:'AutoGluon wins for Homework 6',
    equation:'\\text{test prevalence}=\\frac{526}{44\\,885}\\approx0.0117',
    note:'Unequal training access: AutoGluon fits 136,146 rows; each transformer receives a 6,468-row context. All scores are official-test AP.',
    reading:'The target is qualifying solo recreation beginning at or after noon, predicted from morning activities and earlier Current Population Survey background characteristics. The official test set contains 44,885 respondents and 526 positives. AutoGluon selects its ensemble using training out-of-fold predictions, then scores the official test set. Each transformer receives the same 6,468-row context: all 1,617 training positives plus 4,851 negatives. Constructing that context consults all 136,146 training labels. The chart displays all four prespecified transformer variants: TabICLv2 and TabDPT 1.3, each with the original 611 inputs or 100 training-selected inputs. TabDPT applies its native PCA reduction to the 611-input representation, retaining 128 components. These are comparisons of the fitted procedures under their stated training access, not an isolated causal effect of model architecture.',
    references:[]
  },
  {
    chapter, kind:'atus', atusStage:1,
    title:'Give every learner the same 200 cases.',
    body:'Three of the four foundation-model variants have higher observed AP than AutoGluon’s ensemble, whose weights are selected using only the 200 training rows.',
    equation:'n_{\\mathrm{train}}=100_{\\mathrm{positive}}+100_{\\mathrm{negative}}=200',
    note:'Both size and balance change. TabDPT 100’s AP difference from AutoGluon is +0.0021; its unadjusted paired 95% interval [−0.0020, +0.0086] includes zero.',
    reading:'All five procedures use the same fixed 200-row training sample, while evaluation remains on the same official test set. Learner preprocessing, category levels and feature selection use only those 200 rows. The AutoGluon ensemble is selected by its training out-of-fold predictions in this regime. The sample is balanced by consulting labels in the larger training pool, so this is not a 200-label acquisition budget. All four transformer variants were prespecified and remain visible. TabDPT with 100 selected features has the highest observed AP among the four displayed variants: 0.02294 versus AutoGluon’s 0.02085. The paired-bootstrap 95% interval for the AP difference is [−0.00205, +0.00864], which includes zero and is unadjusted for selecting that reported maximum or for multiple comparisons. One fixed sample cannot establish a general small-data advantage. Training size and class balance both differ from the preceding scene, so the transition is not a controlled learning curve.',
    references:[]
  }
];
