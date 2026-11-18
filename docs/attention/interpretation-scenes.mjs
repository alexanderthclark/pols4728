import {attentionInterpretation, deletionExperiment} from './interpretation-model.mjs';

// Each draw function is a pure function of scroll progress. The two discrete
// comparisons therefore work identically when scrolling forward or backward.
export function createInterpretationScenes({m, line, rect, C}) {
  const negative = attentionInterpretation();
  const positive = attentionInterpretation({tone: 'great'});
  const deletion = deletionExperiment();
  const {blue, rust, teal, ink, muted, rule} = C;
  const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  // Explicit inline font-size keeps small-screen generic SVG text rules from
  // making long labels overflow. All labels remain real SVG text.
  const label = (value, x, y, size = 24, color = ink, anchor = 'middle', extra = '') =>
    `<text class="interpretation-label" x="${x}" y="${y}" text-anchor="${anchor}" style="font-size:${size}px;fill:${color}" ${extra}>${esc(value)}</text>`;
  const fixed = (value, digits = 2) => (Math.abs(value) < .000001 ? 0 : value).toFixed(digits).replace('-', '−');
  const signed = value => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(2)}`;
  const texNumber = value => (Math.abs(value) < .000001 ? 0 : value).toFixed(2);
  const texVector = values => `\\left[${values.map(texNumber).join(',\\;')}\\right]`;
  const percent = value => `${Math.round(value * 100)}%`;
  const clamp = value => Math.max(0, Math.min(1, value));
  const ease = value => { const t = clamp(value); return t * t * (3 - 2 * t); };
  const lerp = (start, end, progress) => start + (end - start) * progress;
  const math = (value, x, y, width = 200, height = 58, size = 27, color = ink) => m(value, x, y, width, height, size, color);
  const arrow = (x1, y1, x2, y2, color = muted) => line(x1, y1, x2, y2, color, 1.7, 'marker-end="url(#arrow)"');
  const panel = (x, y, w, h, fill = '#f5f6f5', stroke = 'none') => rect(x, y, w, h, fill, stroke, 9);
  function packet(value, x, y, width = 170, color = teal) {
    return panel(x - width / 2, y - 25, width, 50, '#fff', rule)
      + math(value, x, y, width - 8, 48, 24, color);
  }
  function tokens(model, y = 115) {
    return model.tokens.map((token, j) => {
      const x = [130, 320, 510][j];
      const color = j === 2 ? (token === 'great' ? teal : rust) : blue;
      return panel(x - 77, y - 39, 154, 64, '#fff', rule) + label(token, x, y, 30, color);
    }).join('');
  }
  function probabilityBar(model, y = 425) {
    return math(`P(\\mathrm{positive})=${(model.probability * 100).toFixed(1)}\\%`, 320, y - 31, 530, 64, 31, model.logit < 0 ? rust : teal)
      + panel(95, y, 450, 24, '#eeeeeb')
      + panel(95, y, 450 * model.probability, 24, model.logit < 0 ? rust : teal)
      + line(320, y - 3, 320, y + 30, muted, 1.3)
      + label('0%', 95, y + 55, 20, muted) + label('50%', 320, y + 55, 20, muted) + label('100%', 545, y + 55, 20, muted);
  }
  function outputVector(t) {
    const total = clamp(t) * 3;
    const active = Math.min(2, Math.floor(total));
    const progress = ease(total - Math.floor(total));
    let out = label('A small review', 320, 47, 25, muted) + tokens(negative, 123);
    [130, 320, 510].forEach(x => { out += arrow(x, 171, 320, 265); });
    out += panel(93, 282, 454, 104, '#edf3f6')
      + label('attention output', 320, 315, 23, blue)
      + math(total < 3 ? 'h=\\sum_j\\alpha_jv_j' : `h=${texVector(negative.output)}`, 320, 356, 432, 62, 35, blue)
      + label('One vector for the summary destination.', 320, 444, 25, ink)
      + label('The classifier still has to read it.', 320, 493, 26, muted);
    // Each weighted value travels from its source toward the receiving vector.
    // The result is revealed only after all three inputs have arrived; numeric
    // labels never interpolate into an invented intermediate model output.
    if (total < 3) {
      const startX = [130, 320, 510][active];
      const x = lerp(startX, 320, progress);
      const y = lerp(189, 267, progress);
      const carried = negative.values[active].map(coordinate => coordinate * negative.weights[active]);
      out += packet(texVector(carried), x, y, 160, active === 2 ? rust : teal);
    }
    return out;
  }
  function roles() {
    let out = panel(28, 32, 584, 247, '#edf3f6')
      + label('Queries and keys: match the summary to sources', 320, 75, 24, blue)
      + label('This query gives the following mixing weights.', 320, 114, 22, muted);
    negative.tokens.forEach((token, j) => {
      const x = [130, 320, 510][j];
      out += label(percent(negative.weights[j]), x, 181, 39, blue)
        + label(token, x, 231, 27, j === 2 ? rust : ink);
    });
    out += panel(28, 312, 584, 166, '#f5f6f5')
      + label('Values: the information each source supplies', 320, 352, 25, teal);
    negative.values.forEach((value, j) => {
      out += math(texVector(value), [130, 320, 510][j], 408, 190, 62, 27, teal);
    });
    out += label('Matching and carrying are different jobs.', 320, 519, 24, muted);
    return out;
  }
  function mixture(t) {
    const total = clamp(t) * 3;
    const completed = Math.min(3, Math.floor(total));
    const active = Math.min(completed, 2);
    const progress = ease(total - completed);
    const weightedValues = negative.values.map((value, j) => value.map(coordinate => coordinate * negative.weights[j]));
    const partial = [0, 1].map(coordinate => weightedValues.slice(0, completed).reduce((sum, value) => sum + value[coordinate], 0));
    let out = '';
    negative.tokens.forEach((token, j) => {
      const x = [130, 320, 510][j];
      out += label(token, x, 46, 27, j === 2 ? rust : ink)
        + math(`v_${j+1}=${texVector(negative.values[j])}`, x, 109, 202, 62, 24, blue)
        + math(`${texNumber(negative.weights[j])}\\times v_${j+1}`, x, 169, 190, 58, 26, blue)
        + math(texVector(weightedValues[j]), x, 229, 190, 60, 25, teal)
        + arrow(x, 264, 320, 377, j === active && completed < 3 ? teal : rule);
    });
    out += panel(81, 400, 478, 101, '#edf3f6')
      + label(completed === 3 ? 'The complete sum' : `${completed} of 3 weighted vectors added`, 320, 429, 23, blue)
      + math(`${completed === 3 ? 'h' : 'h_{\\mathrm{partial}}'}=${texVector(partial)}`, 320, 472, 460, 62, 32, blue)
      + label('Each arriving vector adds to both coordinates.', 320, 532, 22, muted);
    if (completed < 3) {
      out += packet(texVector(weightedValues[active]), lerp([130, 320, 510][active], 320, progress), lerp(276, 373, progress), 181, active === 2 ? rust : teal);
    }
    return out;
  }
  function readout(t) {
    const total = clamp(t) * 3;
    const phase = Math.min(2, Math.floor(total));
    const progress = t >= 1 ? 1 : ease(total - Math.floor(total));
    const zReady = total >= 2;
    let out = label('Read the two features with different coefficients.', 320, 38, 24, muted)
      + panel(84, 64, 472, 73, '#edf3f6')
      + math(`h=${texVector(negative.output)}`, 320, 101, 450, 66, 33, blue)
      + math('\\beta=[1,0]^{\\mathsf T}', 320, 165, 350, 60, 28, teal)
      + math('\\times1', 130, 213, 140, 54, 28, teal)
      + math('\\times0', 510, 213, 140, 54, 28, teal)
      + arrow(130, 239, 320, 294) + arrow(510, 239, 320, 294)
      + panel(64, 306, 512, 73, '#f6ece7')
      + math(zReady ? `z=${texNumber(negative.logit)}` : 'z=(-.74\\times1)+(2.90\\times0)', 320, 343, 494, 65, zReady ? 35 : 27, rust);
    if (phase === 0) {
      [130, 510].forEach((x, j) => {
        const y = lerp(143, 260, progress);
        const value = y < 213 ? negative.output[j] : negative.output[j] * negative.readout[j];
        out += packet(texNumber(value), x, y, 103, j === 0 ? rust : muted);
      });
    } else if (phase === 1) {
      const subphase = (total - 1) * 2;
      const j = Math.min(1, Math.floor(subphase));
      const local = ease(subphase - j);
      out += packet(texNumber(negative.output[j] * negative.readout[j]), lerp([130, 510][j], 320, local), lerp(258, 294, local), 103, j === 0 ? rust : muted);
    }
    if (phase < 2) {
      out += math('p=\\sigma(z)', 320, 445, 430, 70, 34, muted)
        + label('First compute the score; then convert it.', 320, 507, 24, muted);
    } else {
      out += arrow(320, 381, 320, 413);
      if (progress < .9) {
        out += packet(`\\sigma(${texNumber(negative.logit)})`, 320, lerp(403, 442, progress), 190, rust)
          + label('Apply the sigmoid to the logit.', 320, 507, 24, muted);
      } else out += probabilityBar(negative, 445);
    }
    return out;
  }
  function compareTerms() {
    let out = label('Mixing weight', 235, 60, 25, blue)
      + label('Signed logit term', 481, 60, 25, ink)
      + math('\\alpha', 235, 93, 170, 54, 25, blue) + math('\\alpha\\,(v\\beta)', 481, 93, 220, 54, 25, ink);
    negative.tokens.forEach((token, j) => {
      const y = 174 + j * 107;
      const contribution = negative.contributions[j];
      out += label(token, 35, y + 7, 26, j === 2 ? rust : ink, 'start')
        + rect(156, y - 17, 150, 28, '#eeeeeb', 'none', 3)
        + rect(156, y - 17, 150 * negative.weights[j], 28, blue, 'none', 3)
        + label(fixed(negative.weights[j]), 329, y + 7, 23, blue);
      const width = Math.abs(contribution) * 150;
      out += line(493, y - 23, 493, y + 18, muted, 1.2);
      if (width) out += rect(contribution < 0 ? 493 - width : 493, y - 17, width, 28, contribution < 0 ? rust : teal, 'none', 2);
      out += label(contribution === 0 ? '0.00' : signed(contribution), 606, y + 7, 24, contribution < 0 ? rust : teal, 'end');
    });
    out += label('More attention need not mean a larger signed term.', 320, 463, 24, ink)
      + math('\\sum_jc_j=z=-0.74', 320, 506, 530, 62, 27, muted);
    return out;
  }
  function sameWeights(t) {
    const changed = t >= .5;
    const model = changed ? positive : negative;
    let out = label(changed ? '2 · Change only the last source value' : '1 · Start with the negative review', 320, 41, 25, muted)
      + tokens(model, 117);
    model.weights.forEach((weight, j) => {
      out += label(percent(weight), [130, 320, 510][j], 201, 35, blue);
    });
    out += label('Exactly the same keys and attention weights.', 320, 250, 24, blue)
      + math(`v_{\\mathrm{${changed ? 'great' : 'awful'}}}=${texVector(model.values[2])}`, 320, 299, 550, 64, 29, changed ? teal : rust)
      + math(`h=${texVector(model.output)},\\quad z=${texNumber(model.logit)}`, 320, 355, 610, 64, 27, ink)
      + probabilityBar(model, 435);
    return out;
  }
  function removeSource(t) {
    const changed = t >= .5;
    let out = label('Deleting film makes the weights rebalance.', 320, 46, 25, muted)
      + label('Original', 246, 105, 26, blue)
      + label('After deletion', 493, 105, 26, changed ? rust : muted);
    negative.tokens.forEach((token, j) => {
      const y = 175 + j * 79;
      out += label(token, 65, y, 27, j === 2 ? rust : ink, 'start')
        + label(fixed(negative.weights[j]), 246, y, 30, blue);
      if (j === 0) {
        out += label(changed ? 'removed' : '—', 493, y, 25, rust);
      } else {
        out += label(changed ? fixed(deletion.deleted.weights[j - 1]) : '—', 493, y, 30, rust);
      }
      out += line(49, y + 22, 594, y + 22, rule);
    });
    out += label('logit', 65, 412, 27, ink, 'start')
      + label(fixed(negative.logit), 246, 412, 35, blue)
      + label(changed ? fixed(deletion.deleted.logit) : 'recompute →', 493, 412, changed ? 35 : 24, rust);
    out += math(changed ? '\\Delta z=-1.26' : 'c_{\\mathrm{film}}=+0.06', 320, 473, 520, 63, 30, changed ? rust : teal)
      + label(changed ? 'That is not just subtracting its +0.06 term.' : 'Now remove the source and recompute the mixture.', 320, 522, 23, muted);
    return out;
  }
  function useAttention() {
    const cards = [
      {y: 25, color: blue, title: 'Inspect the computation', first: 'Which head, layer, and destination?', second: 'Attention shows a particular mixing operation.'},
      {y: 189, color: teal, title: 'Test the prediction', first: 'Edit a word or phrase, then rerun.', second: 'Compare score changes or output attribution.'},
      {y: 353, color: rust, title: 'Optional case · LiDDA', first: 'Normalized attention assigns credit.', second: 'Credit allocation is not incremental impact.'},
    ];
    return cards.map(({y, color, title, first, second}) => panel(28, y, 584, 143, '#fff', rule)
      + rect(28, y, 5, 143, color, 'none', 2)
      + label(title, 61, y + 40, 28, color, 'start')
      + label(first, 61, y + 82, 24, ink, 'start')
      + label(second, 61, y + 117, 22, muted, 'start')).join('');
  }
  return [
    {part: '01 · From attention to a prediction', label: 'Start with the output', title: 'Attention gives a vector.',
      caption: 'For “film was awful,” attention combines information into a summary vector. A separate readout turns that vector into a sentiment prediction.',
      equation: 'h=\\sum_j\\alpha_j v_j',
      detail: 'All numbers are constructed for teaching. This toy uses one fixed summary query, attention pooling, and a linear readout. It has no later layers.',
      visual: 'The two coordinates are model features, not probabilities or named human concepts.', draw: outputVector},
    {part: '01 · From attention to a prediction', label: 'Three roles', title: 'Match sources. Carry information.',
      caption: 'The query and keys determine the mixing weights. Each source also supplies a value vector: the information that receives that weight.',
      equation: '\\alpha_j=\\operatorname{softmax}_j(qk_j)',
      detail: 'Our query/key dimension is one. Choose q = 1 and keys log(.6), log(.3), log(.1). In a trained model, learned maps construct these quantities. <a href="https://arxiv.org/html/1706.03762v7#S3.SS2">Mechanism: Vaswani et al.</a>',
      visual: 'One query gives a distribution over three source positions. Values are separate vectors.', draw: roles},
    {part: '01 · From attention to a prediction', label: 'Combine the values', title: 'Multiply, then add.',
      caption: 'Multiply each value vector by its attention weight, then add the vectors. Both coordinates participate in the mixture.',
      equation: 'h=.60v_{\\mathrm{film}}+.30v_{\\mathrm{was}}+.10v_{\\mathrm{awful}}',
      detail: 'The weights sum to one. A weight of .60 scales a vector; it does not mean “60% of the reason for the classification.”',
      visual: 'The output is [−0.74, 2.90]. The next step specifies how the classifier reads those features.', draw: mixture},
    {part: '01 · From attention to a prediction', label: 'Read the representation', title: 'The readout supplies the meaning.',
      caption: 'This classifier reads the first coordinate and ignores the second. Its logit is −0.74; applying the sigmoid gives a 32.3% positive probability.',
      equation: 'z=h\\beta,\\quad\\beta=[1,0]^{\\mathsf T},\\quad p=\\sigma(z)',
      detail: 'The intercept is zero. This exact linear readout is a teaching simplification; later layers, residual paths, and nonlinearities complicate a full transformer.',
      visual: 'A vector, a logit, and a probability are three different objects.', draw: readout},
    {part: '02 · What can we interpret?', label: 'A weight is only part of the calculation', title: 'Trace the signed terms.',
      caption: '“Film” receives most attention. “Awful” supplies the largest negative term in the logit. Each term depends on the weight, the value, and the readout.',
      equation: 'c_j=\\alpha_j(v_j\\beta),\\qquad z=\\sum_jc_j',
      detail: 'These are exact forward-pass logit terms for this toy, not removal effects or additive probability contributions. <a href="https://aclanthology.org/2020.emnlp-main.574/">Why values matter: Kobayashi et al.</a>',
      visual: 'Mixing weights are nonnegative. Logit terms can be positive, zero, or negative.', draw: compareTerms},
    {part: '02 · What can we interpret?', label: 'Keep the attention map fixed', title: 'The same weights. A different prediction.',
      caption: 'In this toy, “awful” and “great” share a key but carry different values. Replacing one with the other preserves attention and changes the prediction.',
      equation: '\\alpha=[.60,.30,.10]\\quad\\text{in both reviews}',
      detail: 'The probability changes from 32.3% to 70.3%. This is a constructed counterexample, not an empirical statement about a trained sentiment model.',
      visual: 'Scroll between the two complete states; scrolling back restores the original review.', draw: sameWeights, discretePhases: true},
    {part: '02 · What can we interpret?', label: 'Intervene and recompute', title: 'Deletion asks a different question.',
      caption: 'Deleting “film” makes the surviving weights .75 and .25. The logit becomes −2.00, even though its original direct term was only +.06.',
      equation: 'z_{\\mathrm{delete\\ film}}=-2.00',
      detail: 'Here deletion changes the logit by −1.26. The query and remaining values stay fixed in this toy; a full model can change those too. A direct term is not the whole effect of an edit.',
      visual: 'First inspect the original calculation; then recompute after removing the source.', draw: removeSource, discretePhases: true},
    {part: '02 · What can we interpret?', label: 'Use the right evidence', title: 'Inspect, then test.',
      caption: 'Use attention to inspect information flow. To explain a prediction, also test input edits or use an output-attribution method. State exactly which quantity you are explaining.',
      equation: '\\text{mixing weight}\\;\\ne\\;\\text{prediction change}',
      detail: 'Integrated Gradients explains a score difference relative to a baseline. Attention may be informative when validated; it is not automatically an explanation. <a href="https://proceedings.mlr.press/v70/sundararajan17a.html">Integrated Gradients</a> · <a href="https://aclanthology.org/D19-1002/">Attention and explanation</a> · <a href="https://arxiv.org/html/2505.09861v3">LiDDA v3, §§2.1/2.4</a>. The case below separates credit allocation from incremental impact.',
      visual: 'A useful attention pattern is a starting point for a precise, testable claim.', draw: useAttention},
  ];
}
