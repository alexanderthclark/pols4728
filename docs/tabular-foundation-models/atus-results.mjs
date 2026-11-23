// Frozen official-test results. Selection and preprocessing used training data;
// these five rows are displayed without selecting a winner on the test set.
export const atusResults = {
  task: 'Predict qualifying solo recreation beginning at or after noon from morning activities and earlier background characteristics.',
  metric: 'Average precision',
  test: {rows:44885, positives:526, prevalence:0.011718837027960343},
  axis: {min:0, max:0.055},
  regimes: {
    larger: {
      label:'Larger training set',
      autogluon_fit_rows:136146,
      transformer_context_rows:6468,
      transformer_context_positives:1617,
      transformer_context_negatives:4851,
      sampling_labels_consulted:136146,
      matched_training_access:false
    },
    small: {
      label:'Same 200 training rows',
      training_rows:200,
      positives:100,
      negatives:100,
      fixed_sample:true,
      sampling_labels_consulted:136146,
      learner_training_labels:200,
      sample_sha256:'ec02442ca128b9c4e467877f5fd371e814fdfd2cd4fb490eae84aeba3632803f'
    }
  },
  models: [
    {id:'autogluon', family:'AutoGluon', variant:'611 inputs', color:'#234e70',
      larger:0.045068472116863864, small:0.020853589696533933,
      selection:'Ensemble selected by training out-of-fold predictions, separately in each regime.'},
    {id:'tabicl_611', family:'TabICLv2', variant:'611 inputs', color:'#007f73',
      larger:0.03676066381032611, small:0.021710492099350547},
    {id:'tabicl_100', family:'TabICLv2', variant:'100 selected', color:'#007f73',
      larger:0.03638768169067709, small:0.021466909514714354},
    {id:'tabdpt_611', family:'TabDPT 1.3', variant:'611 → 128 PCA', color:'#a54f32',
      larger:0.028308725341052034, small:0.016696251652385828},
    {id:'tabdpt_100', family:'TabDPT 1.3', variant:'100 selected', color:'#a54f32',
      larger:0.034257776077229365, small:0.022940608498178}
  ],
  small_sample_comparison: {
    contrast:'TabDPT, 100 selected features minus AutoGluon',
    ap_difference:0.0020870188016440666,
    paired_bootstrap_95_interval:[-0.002047520612924495, 0.008636855629215578],
    includes_zero:true
  },
  protocol: {
    official_test_shared:true,
    metric_definition:'Unweighted noninterpolated average precision',
    model_seed:4728,
    small_sample_seed:4728200,
    paired_bootstrap_repetitions:2000,
    paired_bootstrap_seed:4728205,
    transformer_variants_prespecified:4,
    native_transformer_ensemble_members:4,
    feature_selection:'100-feature variants use training-only selection, recomputed within each regime.',
    caveat:'Both training size and class balance change between regimes; this is not a controlled learning curve.',
    fixed_thirds_ensemble_displayed:false
  }
};
