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
  function matrixGrid({row = -1, column = -1} = {}) {
    const x = 204, y = 136, width = 115, height = 82;
    let out = label('Source token j', 376, 48, 27, blue)
      + label('Destination i', 92, 101, 23, blue);
    negative.tokens.forEach((token, j) => {
      out += label(token, x + (j + .5) * width, 101, 26, column === j ? rust : ink);
    });
    negative.attentionMatrix.forEach((weights, i) => {
      out += label(negative.tokens[i], 163, y + (i + .5) * height + 8, 28, row === i ? blue : ink, 'end');
      weights.forEach((weight, j) => {
        const selected = i === row && (column < 0 || j === column);
        out += rect(x + j * width, y + i * height, width, height, '#fff', rule, 0)
          + `<rect x="${x+j*width}" y="${y+i*height}" width="${width}" height="${height}" fill="${blue}" opacity="${.05+.27*weight}"/>`
          + label(fixed(weight), x + (j + .5) * width, y + (i + .5) * height + 9, 31, ink);
        if (selected) out += rect(x + j * width + 2, y + i * height + 2, width - 4, height - 4, 'none', column < 0 ? blue : rust, 0, 'stroke-width="3"');
      });
    });
    return out;
  }
  function attentionMatrix(t) {
    const row = Math.min(2, Math.floor(clamp(t) * 3));
    return matrixGrid({row})
      + math('A\\in\\mathbb R^{3\\times3}', 320, 441, 450, 64, 36, blue)
      + label('One row of weights for each destination.', 320, 510, 25, muted);
  }
  function oneEntry(t) {
    const progress = ease(t);
    return matrixGrid({row: 0, column: 1})
      + label('was', 170, 450, 30, rust)
      + label('film', 483, 450, 30, blue)
      + arrow(218, 441, 433, 441, blue)
      + `<circle cx="${lerp(224,425,progress)}" cy="441" r="7" fill="${rust}"/>`
      + label('source j = 2', 170, 491, 23, muted)
      + label('destination i = 1', 483, 491, 23, muted);
  }
  function oneRow(t) {
    let out = matrixGrid({row: 0});
    let x = 106;
    negative.attentionMatrix[0].forEach((weight, j) => {
      const width = weight * 470;
      const colors = [blue, teal, rust];
      out += rect(x, 445, width * ease(t), 29, colors[j], '#fff', 0)
        + label(percent(weight), x + width / 2, 426, 23, colors[j]);
      x += width;
    });
    return out + math('0.80+0.15+0.05=1', 320, 513, 540, 54, 30, blue);
  }
  function sourceValues(t) {
    let out = label('Source token', 105, 49, 26, muted)
      + label('Value vector', 423, 49, 26, teal);
    negative.values.forEach((value, j) => {
      const y = 142 + j * 133;
      out += label(negative.tokens[j], 105, y + 8, 30, j === 2 ? rust : ink)
        + arrow(178, y, 278, y, rule)
        + packet(`v_${j+1}=${texVector(value)}`, lerp(340, 435, ease(t)), y, 275, teal);
    });
    return out + label('Two features per token in this example.', 320, 514, 25, muted);
  }
  function allOutputs(t) {
    const count = Math.min(3, 1 + Math.floor(clamp(t) * 3));
    const texMatrix = rows => `\\begin{bmatrix}${rows.map(row => row.map(texNumber).join('&')).join('\\\\')}\\end{bmatrix}`;
    let out = math('A', 168, 46, 230, 54, 32, blue)
      + math('V', 478, 46, 200, 54, 32, teal)
      + math(texMatrix(negative.attentionMatrix), 168, 146, 296, 162, 27, blue)
      + math('\\times', 336, 147, 60, 60, 32, ink)
      + math(texMatrix(negative.values), 478, 146, 250, 162, 27, teal)
      + arrow(320, 246, 320, 285)
      + math('U', 320, 312, 110, 50, 34, blue);
    negative.outputs.forEach((output, i) => {
      const y = 363 + i * 70;
      out += label(negative.tokens[i], 142, y + 8, 28, ink, 'end')
        + panel(174, y - 27, 405, 56, i === count - 1 ? '#edf3f6' : '#fff', rule)
        + math(i < count ? `u_${i+1}=${texVector(output)}` : `u_${i+1}`, 375, y, 390, 53, 29, blue);
    });
    return out;
  }
  function poolOutputs(t) {
    const total = clamp(t) * 3;
    const active = Math.min(2, Math.floor(total));
    let out = label('Attention output U', 320, 40, 27, blue);
    negative.outputs.forEach((output, i) => {
      const y = 100 + i * 77;
      out += label(negative.tokens[i], 135, y + 8, 27, ink, 'end')
        + math(`u_${i+1}=${texVector(output)}`, 372, y, 430, 58, 29, blue);
    });
    out += arrow(320, 288, 320, 347)
      + label('mean of the three rows', 449, 321, 23, muted)
      + panel(62, 361, 516, 125, '#edf3f6')
      + label('Sentence summary', 320, 398, 27, blue)
      + math(total >= 3 ? `h=${texVector(negative.output)}` : 'h=\\frac{u_1+u_2+u_3}{3}', 320, 446, 488, 70, 33, blue)
      + label('This is the vector the classifier will read.', 320, 529, 24, muted);
    if (total < 3) {
      out += packet(`\\tfrac13 u_${active+1}`, 212, lerp(100+active*77, 339, ease(total-active)), 120, teal);
    }
    return out;
  }
  function roles(t) {
    let out = panel(42, 26, 556, 91, '#fff', rule)
      + label('Input representations X', 320, 80, 30, ink)
      + arrow(237, 121, 165, 180) + arrow(403, 121, 475, 180)
      + panel(40, 191, 272, 89, '#edf3f6') + panel(365, 191, 235, 89, '#f5f6f5')
      + math('Q=XW_Q,\\ K=XW_K', 176, 235, 259, 65, 24, blue)
      + math('V=XW_V', 481, 235, 224, 65, 29, teal)
      + arrow(176, 285, 176, 352)
      + label('match and normalize', 173, 326, 23, muted)
      + math('A', 176, 400, 120, 65, 42, blue)
      + arrow(476, 285, 476, 408)
      + arrow(211, 405, 416, 462)
      + math('U=AV', 467, 469, 245, 65, 36, teal);
    return out + `<circle cx="176" cy="${lerp(288,358,ease(t))}" r="7" fill="${blue}"/>`;
  }
  function mixture(t) {
    const total = clamp(t) * 3;
    const completed = Math.min(3, Math.floor(total));
    const active = Math.min(completed, 2);
    const progress = ease(total - completed);
    const weightedValues = negative.values.map((value, j) => value.map(coordinate => coordinate * negative.attentionMatrix[0][j]));
    const partial = [0, 1].map(coordinate => weightedValues.slice(0, completed).reduce((sum, value) => sum + value[coordinate], 0));
    let out = '';
    negative.tokens.forEach((token, j) => {
      const x = [130, 320, 510][j];
      out += label(token, x, 46, 27, j === 2 ? rust : ink)
        + math(`v_${j+1}=${texVector(negative.values[j])}`, x, 109, 202, 62, 24, blue)
        + math(`${texNumber(negative.attentionMatrix[0][j])}\\times v_${j+1}`, x, 169, 190, 58, 26, blue)
        + math(texVector(weightedValues[j]), x, 229, 190, 60, 25, teal)
        + arrow(x, 264, 320, 377, j === active && completed < 3 ? teal : rule);
    });
    out += panel(81, 400, 478, 101, '#edf3f6')
      + label(completed === 3 ? 'The complete sum' : `${completed} of 3 weighted vectors added`, 320, 429, 23, blue)
      + math(`${completed === 3 ? 'u_1' : 'u_{1,\\mathrm{partial}}'}=${texVector(partial)}`, 320, 472, 460, 62, 32, blue)
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
    let out = label('Mean source weight', 235, 60, 23, blue)
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
    out += label('Same matrix A; same mean source weights.', 320, 250, 24, blue)
      + math(`v_{\\mathrm{${changed ? 'great' : 'awful'}}}=${texVector(model.values[2])}`, 320, 299, 550, 64, 29, changed ? teal : rust)
      + math(`h=${texVector(model.output)},\\quad z=${texNumber(model.logit)}`, 320, 355, 610, 64, 27, ink)
      + probabilityBar(model, 435);
    return out;
  }
  function removeSource(t) {
    const changed = t >= .5;
    let out = label('Delete film, renormalize, then pool again.', 320, 46, 25, muted)
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
    {id: 'matrix', part: '01 · The attention matrix', label: '', title: 'Start with the attention matrix.',
      caption: 'Rows are destination tokens. Columns are source tokens. Each destination has its own row of weights.',
      equation: 'A=(a_{ij})',
      detail: 'One self-attention head, one layer, one input. The numbers in this example are constructed.',
      visual: 'The same tokens label both axes; the two axes have different roles.', draw: attentionMatrix},
    {id: 'entry', part: '01 · The attention matrix', label: '', title: 'Read one entry.',
      caption: 'This entry is the weight “film” gives “was.” Information travels from the source column to the destination row.',
      equation: 'a_{12}=0.15',
      detail: 'The first index identifies the destination; the second identifies the source.',
      visual: 'Source: was. Destination: film.', draw: oneEntry},
    {id: 'row', part: '01 · The attention matrix', label: '', title: 'Read one row.',
      caption: '“Film” gives 80% of its weight to itself, 15% to “was,” and 5% to “awful.”',
      equation: 'a_{ij}\\geq0,\\qquad\\sum_j a_{ij}=1',
      detail: 'Each row sums to one. Columns need not sum to one, and the matrix need not be symmetric.',
      visual: 'These are mixing weights, not probabilities that a word is important.', draw: oneRow},
    {id: 'values', part: '02 · What the matrix does', label: '', title: 'Each source supplies a vector.',
      caption: 'The weights multiply value vectors: one row of features for each source token.',
      equation: 'V=\\begin{bmatrix}v_1\\\\v_2\\\\v_3\\end{bmatrix}',
      detail: 'Our values have two coordinates. A weight scales both coordinates of its source vector.',
      visual: 'Values carry information; attention weights determine how it is mixed.', draw: sourceValues},
    {id: 'mixture', part: '02 · What the matrix does', label: '', title: 'One row makes one output.',
      caption: 'Multiply each source value by the weight in the “film” row, then add. The result stays at the “film” position.',
      equation: 'u_1=.80v_1+.15v_2+.05v_3',
      detail: 'Each destination uses its own attention row to combine the same source values.',
      visual: 'The output at film is [−0.32, 3.45]. It is a feature vector, not a class probability.', draw: mixture},
    {id: 'outputs', part: '02 · What the matrix does', label: '', title: 'Every token gets an output.',
      caption: 'Repeat with every row of A. Matrix multiplication collects the output vectors into U.',
      equation: '\\underbrace{A}_{3\\times3}\\underbrace{V}_{3\\times2}=\\underbrace{U}_{3\\times2}',
      detail: 'In a transformer, these rows feed into representation updates and subsequent processing.',
      visual: 'A is the matrix of weights. U is the matrix of attention outputs.', draw: allOutputs},
    {id: 'roles', part: '02 · What the matrix does', label: '', title: 'Where do the weights come from?',
      caption: 'Queries and keys determine A by matching destinations to sources. Values supply the information that A combines.',
      equation: 'A=\\operatorname{softmax}_{\\mathrm{row}}(QK^\\top/\\sqrt{d_k})',
      detail: 'Learned projections produce Q, K and V from the input. The full calculation follows later.',
      visual: 'Matching produces the weights; multiplying by V produces the outputs.', draw: roles},
    {id: 'pooling', part: '03 · From outputs to a prediction', label: '', title: 'Make a sentence summary.',
      caption: 'For this classifier, average the three output vectors. Call this sentence summary h.',
      equation: 'h=\\frac13(u_1+u_2+u_3)',
      detail: 'Mean pooling is our chosen readout design; attention itself returns all three rows of U.',
      visual: 'U supplies the token outputs. Their mean h goes to the classifier.', draw: poolOutputs},
    {id: 'readout', part: '03 · From outputs to a prediction', label: '', title: 'The classifier reads h.',
      caption: 'This classifier reads the first feature of h. Its score is −0.74; the sigmoid converts that score to a probability.',
      equation: 'z=h\\beta,\\quad\\beta=[1,0]^{\\mathsf T},\\quad p=\\sigma(z)',
      detail: 'The intercept is zero. This toy uses one attention head, mean pooling and a linear readout, with no later layers.',
      visual: 'Token outputs → sentence summary → class score → class probability.', draw: readout},
    {id: 'terms', part: '04 · Interpreting attention', label: '', title: 'A large weight is not a large contribution.',
      caption: 'Averaging the outputs also averages the attention rows. “Film” has the largest mean weight; “awful” contributes most to the negative score.',
      equation: '\\alpha_j=\\tfrac13\\sum_i a_{ij},\\quad c_j=\\alpha_j(v_j\\beta)',
      detail: 'The signed terms sum to the logit, not the probability. They are not deletion effects. <a href="https://aclanthology.org/2020.emnlp-main.574/">Kobayashi et al.</a>',
      visual: 'α is computed by averaging across destinations, rather than selecting a destination row.', draw: compareTerms},
    {id: 'substitution', part: '04 · Interpreting attention', label: '', title: 'Same attention. Different prediction.',
      caption: 'Replace “awful” with “great.” Our construction keeps queries and keys fixed but changes the last value vector.',
      equation: 'A_{\\mathrm{awful}}=A_{\\mathrm{great}}',
      detail: 'The positive probability changes from 32.3% to 70.3%. A trained model may change queries and keys too.',
      visual: 'The entire attention matrix is unchanged; the information it combines changes.', draw: sameWeights, discretePhases: true},
    {id: 'deletion', part: '04 · Interpreting attention', label: '', title: 'Delete a token and recompute.',
      caption: 'Removing “film” removes its row and column. Renormalize the remaining rows, combine values, and pool again.',
      equation: 'z_{\\mathrm{delete\\ film}}=-2.00',
      detail: 'The logit changes by −1.26, not −.06. Remaining Q/K/V vectors stay fixed in this toy; a deeper model can change them too.',
      visual: 'The table shows mean source weights before and after deletion.', draw: removeSource, discretePhases: true},
    {id: 'uses', part: '04 · Interpreting attention', label: '', title: 'Inspect, then test.',
      caption: 'Use attention to inspect information flow. Test input edits or output attributions to investigate a prediction.',
      equation: '\\text{mixing weight}\\;\\ne\\;\\text{prediction change}',
      detail: '<a href="https://proceedings.mlr.press/v70/sundararajan17a.html">Integrated Gradients</a> · <a href="https://aclanthology.org/D19-1002/">Attention and explanation</a> · <a href="https://arxiv.org/html/2505.09861v3">LiDDA</a>',
      visual: 'Identify the head, layer, destination and output you are explaining.', draw: useAttention},
  ];
}
