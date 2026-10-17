import { illustrativeClasses, probeSentences, trainingSentence, observations, initialParameters, completedParameters, forward, crossEntropy, interpolateParameters } from './model.mjs';

const C = { ink: '#242424', blue: '#234E70', rust: '#A54F32', teal: '#007F73', muted: '#606060', rule: '#767676', light: '#f4f5f6', white: '#fff' };
const colors = [C.blue, C.rust, C.teal];
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const clamp = value => Math.max(0, Math.min(1, value));
const mix = (a, b, t) => a + (b - a) * t;
const smooth = t => t * t * (3 - 2 * t);
const n = value => Number(value.toFixed(3));
const txt = (x, y, value, size = 24, attrs = '') => `<text x="${n(x)}" y="${n(y)}" font-size="${size}" ${attrs}>${escape(value)}</text>`;
const lines = (x, y, values, size = 24, attrs = '') => `<text x="${n(x)}" y="${n(y)}" font-size="${size}" ${attrs}>${values.map((value, i) => `<tspan x="${n(x)}" dy="${i ? size * 1.3 : 0}">${escape(value)}</tspan>`).join('')}</text>`;
const box = (x, y, w, h, stroke = C.blue, fill = C.white, extra = '') => `<rect x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}" rx="8" stroke="${stroke}" stroke-width="2" fill="${fill}" ${extra}/>`;
const arrow = (x1, y1, x2, y2, color = C.rule, extra = '') => `<path d="M${n(x1)},${n(y1)} L${n(x2)},${n(y2)}" stroke="${color}" stroke-width="2.5" fill="none" marker-end="url(#peloton-arrow)" ${extra}/>`;
const rule = (x1, y1, x2, y2, color = C.rule, extra = '') => `<path d="M${n(x1)},${n(y1)} L${n(x2)},${n(y2)}" stroke="${color}" stroke-width="2" fill="none" ${extra}/>`;
const center = 'text-anchor="middle"';

const scenes = [
  {
    id: 'speech-and-self-improvement', title: 'What does an instructor ask of you?',
    body: '<p>Tara Isabella Burton’s reading of modern fitness culture treats self-love and relentless self-improvement as moral demands. To paraphrase the argument that motivates this example: equal parts Ayn Rand and John Calvin, your only duty is to yourself, and you are never doing well enough.</p><p>We can begin investigating that interpretation with a narrower measurement task: classify when an instructor gives technical coaching, talks about music, or encourages the rider, then consider what those categories say about the larger argument.</p>',
    note: 'Burton motivates the research question. The three speech types here are illustrative teaching labels, not the class list recovered from the original dataset.',
  },
  {
    id: 'sentences', title: 'Begin with a sentence, not a philosophy',
    body: '<p>The supplied teaching script ends by testing the three sentences shown here: given a sentence, return a speech category. Its training CSV and saved model were not supplied, so these sentences do not come with verified model predictions.</p><p>For the explanation, we use the simplified categories Coaching, Music, and Encouragement; the real script constructs its label list from the available labeled data.</p>',
    note: 'These are the script’s new test sentences. Their role in that script is prediction after training, not training examples.',
  },
  {
    id: 'llm-labels', title: 'A larger model supplies provisional labels',
    body: '<p>The workflow first asks Gemini Flash to assign topic labels to transcript sentences, creating paired inputs and targets for a smaller classifier to learn. In the diagram, an invented training sentence receives an illustrative Coaching label.</p><p>Learning to reproduce these labels establishes agreement with the labeling rule, so checking a sample for the distinctions a researcher cares about remains part of defining the measure.</p>',
    note: 'The 30 observations and the three categories are schematic. The sentence “Keep your shoulders relaxed.” is an invented training example.',
  },
  {
    id: 'split', title: 'Reserve sentences before fitting the classifier',
    body: '<p>The script makes an 80/20 training–validation split stratified by topic: each category contributes examples to both parts, and the reserved sentences supply no gradient updates. Our schematic corpus has ten observations per category, so eight move to training and two remain for validation.</p><p>This asks whether the classifier can reproduce labels on held-out sentences from this corpus; to assess generalization to new instructors or classes, we would also need a split that holds those groups apart.</p>',
    equation: '<span>30 illustrative observations = 24 training + 6 validation</span>',
  },
  {
    id: 'tokens', title: 'Turn a training sentence into model inputs',
    body: '<p>The tokenizer turns a sentence into token IDs, adds the special tokens expected by the pretrained model, and supplies an attention mask so padding is ignored. The script uses the uncased tokenizer and truncates long inputs to a maximum of 128 tokens.</p><p>Token IDs identify vocabulary entries rather than meaningful coordinates; the pretrained encoder supplies the learned representations that make them useful.</p>',
    note: 'The displayed token sequence is schematic; token IDs are deliberately omitted. It is not a captured tokenizer output.',
  },
  {
    id: 'encoder', title: 'Start from an encoder that already represents language',
    body: '<p>DistilBERT begins with pretrained parameters rather than learning language from these fitness transcripts alone. Its contextual representations let the same word contribute differently depending on the surrounding sentence.</p><p>For sentence classification, the first token’s final hidden state passes into a task-specific classification head, which turns this representation into scores for the chosen speech categories.</p>',
    equation: '<span>sentence → tokens → pretrained encoder → sentence representation <i>h</i></span>',
  },
  {
    id: 'head', title: 'The classification head turns a representation into scores',
    body: '<p>The head produces one <em>logit</em> for each category: an unconstrained score that need not lie between zero and one or sum with the other scores to one. To make the update visible, we now use a tiny differentiable surrogate with a two-dimensional hidden representation and a three-category linear head.</p><p>Its weights, scores, and probabilities are teaching calculations, not outputs from the unavailable DistilBERT model.</p>',
    equation: '<span>Tiny surrogate: <i>h</i> = tanh(<i>E x</i>), &nbsp; <i>z</i> = <i>W h</i> + <i>b</i></span>',
  },
  {
    id: 'softmax-loss', title: 'Softmax makes a distribution; the label defines a loss',
    body: '<p>Softmax converts the three logits into positive probabilities that sum to one. For an example labeled Coaching, cross-entropy is the negative logarithm of the probability assigned to Coaching: a confident correct assignment has a smaller loss.</p><p>The supplied target label defines this penalty, so reducing the loss teaches agreement with that label rather than independently establishing the validity of the measure.</p>',
    equation: '<span><i>p</i><sub>k</sub> = exp(<i>z</i><sub>k</sub>) / Σ<sub>j</sub> exp(<i>z</i><sub>j</sub>), &nbsp; <i>L</i> = −log <i>p</i><sub>Coaching</sub></span>',
    note: 'All displayed probabilities and loss values are calculated from the explicit tiny surrogate and are illustrative.',
  },
  {
    id: 'gradient', title: 'One loss sends a gradient through both parts',
    body: '<p>Backpropagation follows how the loss depends on the head and then on the encoder, supplying gradients for a parameter update. Scroll through the step to see the surrogate’s encoder and head weights move together, increasing its Coaching probability and reducing this example’s loss.</p><p>Both parts are trainable in the supplied DistilBERT workflow: pretrained weights provide a starting point and can change during fine-tuning.</p>',
    equation: '<span><i>θ</i> ← <i>θ</i> − <i>η</i> ∇<sub>θ</sub><i>L</i>, &nbsp; <i>θ</i> = {encoder parameters, head parameters}</span>',
    note: 'This is a full gradient update in the tiny surrogate. It illustrates the dependency structure, not the numerical size of an actual DistilBERT update.',
  },
  {
    id: 'full-versus-frozen', title: 'Fine-tuning can update more than the new head',
    body: '<p>A frozen encoder supplies fixed features while only the head learns; full fine-tuning lets both the encoder and the head adapt to the task. The supplied script uses full fine-tuning and does not freeze the encoder.</p><p>The comparison clarifies which parameters are being trained: adding a new classification head does not, by itself, imply that only the head changes.</p>',
    note: 'The frozen setup is a counterfactual teaching comparison. Full encoder-and-head training matches the supplied script.',
  },
  {
    id: 'epochs', title: 'Repeat the updates over the training portion',
    body: '<p>An <em>epoch</em> is a pass through the training examples. The supplied run schedules four epochs using minibatches of sixteen examples, evaluating the reserved portion after each epoch as the pretrained representation adapts to the labeling task.</p><p>Four passes describe the procedure; they do not tell us the resulting validation accuracy or guarantee that another pass would help.</p>',
    equation: '<span>training minibatches → parameter updates → next epoch; validation supplies no gradients</span>',
    note: 'The four passes come from the script. The moving observations are a schematic corpus, not its actual minibatches.',
  },
  {
    id: 'validation', title: 'Now ask how the model labels reserved sentences',
    body: '<p>At validation, the fitted model predicts without updating its parameters; comparison with the reserved labels yields validation loss, overall accuracy, and accuracy within each topic. The script uses validation results to select a saved checkpoint, making this a model-selection set rather than a separate untouched test set for a final performance estimate.</p><p>No empirical score is shown here because the data and trained weights were not supplied.</p>',
    note: 'The script’s three short demonstration sentences are useful probes, but they are not a substitute for a labeled test set.',
  },
  {
    id: 'embeddings', title: 'Inspect the representation without mistaking a picture for a score',
    body: '<p>The script also extracts the first token’s final hidden state for each training sentence, before the classification head, and projects those vectors into two dimensions to inspect the distinctions their representation makes. The positions shown here are illustrative, not a computed projection.</p><p>Even a real picture of separated training clusters would not establish held-out accuracy: a two-dimensional map discards information, and these are observations the model has already used.</p>',
    note: 'Only the schematic training observations enter this picture. The original script uses a t-SNE projection of its training embeddings.',
  },
  {
    id: 'return-to-question', title: 'Return from predicted categories to the research question',
    body: '<p>A usable speech classifier could help describe how instructors divide their time among types of utterance, provided we examine mislabeled and ambiguous sentences before treating those categories as measurements. Burton’s broader interpretation concerns the moral language of self-love and self-improvement.</p><p>Coaching, Music, and Encouragement do not measure that argument by themselves; the analysis must connect its categories to the substantive claim rather than let a convenient classifier define the question.</p>',
    note: 'This demonstration explains the training workflow. It reports no estimated topic frequencies, trained-model predictions, or empirical validation results.',
  },
];

const sceneSummaries = [
  'Burton reads modern fitness culture as a moral demand for self-love and relentless improvement: to paraphrase, equal parts Ayn Rand and John Calvin, your only duty is to yourself, and you are never doing well enough. We begin with a narrower question: what kinds of speech appear in an instructor’s transcript?',
  'The script’s three final test sentences make the prediction task concrete: given a sentence, return a speech category. Coaching, Music, and Encouragement are simplified teaching labels; the real class list comes from the labeled corpus.',
  'A larger model assigns provisional topic labels to transcript sentences, creating paired inputs and targets for a smaller classifier. Learning those labels teaches agreement with the labeling rule, so examining a sample remains part of defining the measure.',
  'A stratified 80/20 split reserves examples from every category before fitting. In our schematic corpus, eight observations per category move into training and two remain for validation; those reserved sentences supply no gradient updates.',
  'Tokenization converts a sentence into vocabulary IDs, special tokens, and an attention mask. IDs identify vocabulary entries rather than meaningful coordinates; the pretrained encoder supplies the learned representations that make them useful.',
  'A pretrained encoder supplies contextual language representations, so the same word can contribute differently in different sentences. For classification, the first token’s final hidden state passes to a task-specific head that scores the speech categories.',
  'The classification head produces one unconstrained logit per category. To show the mechanics, we use an explicit tiny surrogate with a two-dimensional representation and three-category head; its scores are illustrative calculations rather than unavailable DistilBERT outputs.',
  'Softmax turns logits into probabilities summing to one. For a Coaching target, cross-entropy is −log p(Coaching), so assigning more probability to that supplied label lowers the loss; this rewards agreement with the label, without independently validating the measure.',
  'Backpropagation supplies gradients for both the head and encoder. Scroll through one calculated update: both sets of weights change, Coaching probability rises, and this example’s loss falls. Pretrained weights provide a starting point and can continue learning during fine-tuning.',
  'Freezing the encoder trains a head on fixed features. Full fine-tuning lets both parts adapt; that is the setup used by the supplied script. Adding a classification head does not imply that only the head changes.',
  'An epoch is a pass through the training examples. The script schedules four epochs in minibatches of sixteen, evaluating validation data after each pass. Four passes describe a procedure, not the resulting accuracy or a guarantee that another pass would help.',
  'Validation compares predictions with reserved labels without updating parameters. Loss and accuracy measure different aspects of agreement; using these results to select a checkpoint makes validation part of model selection. A separate untouched test set would assess performance after those choices.',
  'The script projects training-sentence representations into two dimensions to inspect their structure. These positions are illustrative. Even a real picture of separated training clusters would not establish held-out accuracy: the projection discards information and shows observations already used for fitting.',
  'Speech categories can help describe patterns across classes, but they do not directly measure Burton’s argument about the morality of self-improvement. A substantive analysis must connect its labels to that claim and examine ambiguous and mislabeled sentences.',
];
scenes.forEach((scene, i) => { scene.reading = scene.body; scene.body = `<p>${sceneSummaries[i]}</p>`; });

function observationsAt(stage, compact) {
  const w = compact ? 500 : 1000;
  return observations.map(observation => {
    const { id, category, validation } = observation;
    const j = id % 10;
    const base = { x: compact ? 80 + j * 37 : 585 + j * 35, y: compact ? 240 + category * 65 : 200 + category * 90, r: compact ? 8 : 10, a: 1 };
    if (stage < 2) return { ...base, a: 0 };
    if (stage === 2) return base;
    if (stage === 3) return { x: validation ? (compact ? 378 + (j - 8) * 57 : 780 + (j - 8) * 90) : (compact ? 47 + j * 36 : 110 + j * 65), y: compact ? 170 + category * 86 : 190 + category * 100, r: compact ? 8 : 11, a: 1 };
    if (stage === 5) return { x: compact ? 180 + j * 14 : 735 + j * 15, y: compact ? 310 + category * 20 : 278 + category * 18, r: compact ? 5 : 7, a: validation ? 0.15 : 0.8 };
    if (stage === 6) return { x: compact ? 400 + j * 7 : 770 + j * 15, y: compact ? 240 + category * 65 : 190 + category * 95, r: compact ? 4 : 6, a: validation ? 0.1 : 0.7 };
    if (stage === 10) return { x: compact ? 65 + j * 38 : 120 + j * 79, y: compact ? 330 + category * 31 : 330 + category * 43, r: compact ? 7 : 9, a: validation ? 0.15 : 1 };
    if (stage === 11) return { x: compact ? (validation ? 320 + (j - 8) * 80 : 45 + j * 25) : (validation ? 715 + (j - 8) * 135 : 70 + j * 55), y: compact ? 215 + category * 60 : 210 + category * 80, r: compact ? 8 : 11, a: validation ? 1 : 0.25 };
    if (stage === 12) {
      const phase = id * 2.4;
      return { x: compact ? 115 + category * 123 + Math.sin(phase) * 35 : 250 + category * 245 + Math.sin(phase) * 70, y: compact ? 200 + (category % 2) * 75 + Math.cos(phase) * 45 : 220 + (category % 2) * 105 + Math.cos(phase) * 70, r: compact ? 8 : 11, a: validation ? 0 : 1 };
    }
    if (stage === 13) return { x: compact ? 95 + category * 150 + Math.sin(id * 2.4) * 26 : 210 + category * 290 + Math.sin(id * 2.4) * 55, y: compact ? 178 + Math.cos(id * 2.4) * 28 : 185 + Math.cos(id * 2.4) * 45, r: compact ? 6 : 9, a: validation ? 0 : 0.7 };
    return { x: compact ? 42 + id * 14.3 : 70 + id * 29, y: compact ? 430 : 490, r: compact ? 4 : 5, a: 0.18 };
  });
}

function observationLayer(position, compact) {
  const i = Math.floor(position), t = smooth(position - i);
  const a = observationsAt(i, compact), b = observationsAt(Math.min(i + 1, scenes.length - 1), compact);
  return `<g aria-label="Thirty schematic sentences move through the workflow">${observations.map((observation, j) => {
    const x = mix(a[j].x, b[j].x, t), y = mix(a[j].y, b[j].y, t);
    return `<circle data-observation="${j}" cx="${n(x)}" cy="${n(y)}" r="${n(mix(a[j].r, b[j].r, t))}" opacity="${n(mix(a[j].a, b[j].a, t))}" fill="${observation.validation ? C.white : colors[observation.category]}" stroke="${colors[observation.category]}" stroke-width="2"/>`;
  }).join('')}</g>`;
}

function drawProbes(compact) {
  const x = compact ? 28 : 70, w = compact ? 444 : 860;
  return probeSentences.map((sentence, i) => {
    const y = (compact ? 82 : 100) + i * (compact ? 103 : 118);
    const textLines = compact && i === 0 ? ["Let's increase that resistance", 'to zone five.'] : [sentence];
    return box(x, y, w, compact ? 85 : 90, colors[i]) + lines(x + 20, y + (compact ? 30 : 52), textLines, compact ? 22 : 28);
  }).join('') + txt(compact ? 250 : 500, compact ? 438 : 500, 'Exact test sentences from the supplied script', compact ? 20 : 24, center);
}

function matrix(x, y, values, color, cell = 54, font = 23, prefix = '') {
  return values.map((row, i) => row.map((value, j) => box(x + j * cell, y + i * 50, cell - 4, 44, color) + txt(x + j * cell + (cell - 4) / 2, y + i * 50 + 29, `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`, font, `${center} data-weight="${prefix}-${i}-${j}"`)).join('')).join('');
}

function probabilityBars(compact, state, y0 = undefined) {
  const x = compact ? 190 : 310, width = compact ? 235 : 500;
  const y = y0 ?? (compact ? 128 : 145), gap = compact ? 76 : 98;
  return illustrativeClasses.map((label, k) => {
    const yy = y + k * gap;
    return txt(compact ? 30 : 65, yy + 27, label, compact ? 22 : 28) + box(x, yy, width, 38, C.rule, C.white) + `<rect x="${x}" y="${yy}" width="${n(width * state.probabilities[k])}" height="38" rx="5" fill="${colors[k]}"/>` + txt(compact ? 445 : 855, yy + 27, `${(100 * state.probabilities[k]).toFixed(1)}%`, compact ? 20 : 26);
  }).join('');
}

function stateDrawing(stage, compact, position) {
  const mid = compact ? 250 : 500;
  const s = compact ? 22 : 27;
  if (stage === 0) return lines(mid, compact ? 105 : 120, ['Self-love', 'Self-improvement', 'Technical instruction'], compact ? 29 : 38, center) + arrow(mid, compact ? 215 : 265, mid, compact ? 268 : 330) + lines(mid, compact ? 312 : 382, ['What kinds of speech', 'appear in the transcript?'], compact ? 27 : 36, center) + txt(mid, compact ? 435 : 505, 'A research question becomes a labeling task', compact ? 20 : 24, center);
  if (stage === 1) return drawProbes(compact);
  if (stage === 2) {
    if (compact) return box(35, 60, 430, 66) + txt(250, 99, trainingSentence, 23, center) + arrow(250, 128, 250, 166) + box(135, 175, 230, 45, C.rust) + txt(250, 205, 'LLM topic label', 24, center) + illustrativeClasses.map((label, k) => txt(25, 245 + k * 65, label.slice(0, 1), 22, `fill="${colors[k]}"`)).join('') + txt(mid, 440, 'Invented corpus · illustrative categories', 20, center);
    return box(55, 95, 465, 90) + txt(287, 148, trainingSentence, 26, center) + arrow(287, 190, 287, 245) + box(140, 260, 295, 70, C.rust) + txt(287, 303, 'Gemini Flash label', 28, center) + illustrativeClasses.map((label, k) => txt(580, 165 + k * 90, label, 26, `fill="${colors[k]}"`)).join('') + txt(500, 505, 'Labels become the targets for supervised learning', 26, center);
  }
  if (stage === 3) return txt(compact ? 175 : 340, compact ? 90 : 90, 'Training: 80%', compact ? 25 : 32, center) + txt(compact ? 410 : 825, compact ? 90 : 90, 'Validation: 20%', compact ? 22 : 32, center) + rule(compact ? 340 : 690, compact ? 115 : 120, compact ? 340 : 690, compact ? 390 : 440, C.rule, 'stroke-dasharray="6 6"') + txt(mid, compact ? 440 : 505, 'Each category keeps the same 8 : 2 split', compact ? 21 : 27, center);
  if (stage === 4) {
    const tokens = ['[CLS]', 'keep', 'your', 'shoulders', 'relaxed', '.', '[SEP]'];
    const cols = compact ? 3 : 4, w = compact ? 136 : 190, gap = compact ? 12 : 30;
    const startX = compact ? 34 : 70, startY = compact ? 145 : 175;
    return txt(mid, compact ? 62 : 75, trainingSentence, compact ? 24 : 32, center) + arrow(mid, compact ? 78 : 95, mid, compact ? 118 : 140) + tokens.map((token, j) => {
      const x = startX + (j % cols) * (w + gap), y = startY + Math.floor(j / cols) * 68;
      return box(x, y, w, 50, j === 0 || j === 6 ? C.rust : C.blue) + txt(x + w / 2, y + 32, token, compact ? 22 : 27, center);
    }).join('') + txt(mid, compact ? 395 : 415, 'Special tokens + vocabulary entries', compact ? 22 : 28, center) + txt(mid, compact ? 443 : 505, 'Schematic tokens; no actual token IDs shown', compact ? 20 : 24, center);
  }
  if (stage === 5) {
    if (compact) return box(60, 55, 380, 55) + txt(250, 91, 'Tokens + attention mask', 25, center) + arrow(250, 112, 250, 150) + box(105, 165, 290, 95) + lines(250, 202, ['Pretrained encoder', 'contextual representations'], 23, center) + arrow(250, 265, 250, 292) + box(145, 300, 210, 95, C.teal) + txt(250, 382, 'First-token state h', 22, center) + txt(250, 440, 'Language knowledge is the starting point', 20, center);
    return box(55, 190, 200, 120) + lines(155, 240, ['Token IDs', '+ mask'], 30, center) + arrow(265, 250, 330, 250) + box(350, 150, 265, 200) + lines(482, 215, ['Pretrained', 'encoder'], 35, center) + arrow(625, 250, 690, 250) + box(710, 180, 225, 160, C.teal) + lines(822, 228, ['Sentence', 'representation h'], 27, center) + txt(500, 475, 'Context changes the representation of a token', 28, center);
  }
  if (stage === 6) {
    const state = forward();
    if (compact) return box(55, 70, 155, 85, C.teal) + lines(132, 105, ['Representation', 'h'], 22, center) + arrow(215, 112, 270, 112) + box(285, 70, 160, 85) + lines(365, 104, ['Task-specific', 'head'], 22, center) + state.logits.map((value, k) => txt(55, 230 + k * 65, `${illustrativeClasses[k]}: logit ${value.toFixed(3)}`, 25)).join('') + txt(250, 440, 'Tiny surrogate scores · illustrative', 21, center);
    return box(60, 180, 230, 145, C.teal) + lines(175, 235, ['Representation', 'h'], 30, center) + arrow(300, 250, 370, 250) + box(390, 180, 215, 145) + lines(497, 235, ['Classification', 'head'], 30, center) + arrow(615, 250, 685, 250) + state.logits.map((value, k) => txt(705, 170 + k * 95, `${illustrativeClasses[k]}: ${value.toFixed(3)}`, 27)).join('') + txt(500, 485, 'Logits are scores, not probabilities', 30, center);
  }
  if (stage === 7) {
    const state = forward();
    return txt(mid, compact ? 65 : 70, 'Target label: Coaching', compact ? 26 : 32, center) + probabilityBars(compact, state) + txt(mid, compact ? 399 : 455, `L = −log(${state.probabilities[0].toFixed(3)}) = ${crossEntropy(state.probabilities).toFixed(3)}`, compact ? 25 : 32, center) + txt(mid, compact ? 444 : 520, 'Explicit tiny-model calculation · illustrative', compact ? 20 : 24, center);
  }
  if (stage === 8) {
    const t = smooth(clamp(position - 7));
    const parameters = interpolateParameters(initialParameters, completedParameters, t), state = forward(parameters);
    if (compact) return txt(250, 53, `One gradient step: ${(100 * t).toFixed(0)}%`, 25, center) + txt(128, 105, 'Encoder E', 25, center) + matrix(65, 125, parameters.encoder, C.blue, 65, 22, 'encoder') + txt(365, 105, 'Head W', 25, center) + matrix(300, 125, parameters.head, C.rust, 65, 22, 'head') + arrow(275, 285, 150, 285, C.rust) + txt(250, 326, 'Gradient reaches both parts', 23, center) + txt(250, 377, `p(Coaching) = ${state.probabilities[0].toFixed(3)}`, 26, center) + txt(250, 440, 'Tiny-model update · illustrative calculation', 20, center);
    return txt(500, 65, `One gradient step: ${(100 * t).toFixed(0)}% complete`, 32, center) + txt(270, 145, 'Encoder weights E', 30, center) + matrix(195, 175, parameters.encoder, C.blue, 78, 27, 'encoder') + txt(700, 145, 'Head weights W', 30, center) + matrix(625, 175, parameters.head, C.rust, 78, 27, 'head') + arrow(610, 285, 390, 285, C.rust) + txt(500, 365, 'The gradient passes back through the head into the encoder', 27, center) + txt(500, 425, `p(Coaching) = ${state.probabilities[0].toFixed(3)}; loss = ${crossEntropy(state.probabilities).toFixed(3)}`, 31, center) + txt(500, 510, 'Calculated surrogate weights · illustrative, not DistilBERT parameters', 23, center);
  }
  if (stage === 9) {
    const x = compact ? [25, 265] : [70, 555], w = compact ? 210 : 370;
    return x.map((xx, k) => txt(xx + w / 2, compact ? 65 : 75, k ? 'Full fine-tuning' : 'Frozen encoder', compact ? 23 : 32, center) + box(xx, compact ? 105 : 130, w, compact ? 95 : 120, k ? C.rust : C.rule) + lines(xx + w / 2, compact ? 143 : 178, ['Encoder', k ? 'weights change' : 'weights fixed'], compact ? 23 : 30, center) + arrow(xx + w / 2, compact ? 205 : 260, xx + w / 2, compact ? 238 : 310) + box(xx, compact ? 250 : 325, w, compact ? 85 : 100, C.rust) + lines(xx + w / 2, compact ? 284 : 365, ['Head', 'weights change'], compact ? 23 : 30, center)).join('') + txt(mid, compact ? 405 : 485, 'The supplied script uses full fine-tuning', compact ? 23 : 29, center);
  }
  if (stage === 10) {
    const progress = clamp(position - 9) * 4;
    const xs = compact ? [72, 190, 310, 428] : [170, 390, 610, 830];
    return txt(mid, compact ? 65 : 70, 'Four epochs through the training data', compact ? 25 : 34, center) + xs.map((x, i) => {
      const radius = compact ? 31 : 46, cy = compact ? 160 : 185;
      const extent = clamp(progress - i), circumference = 2 * Math.PI * radius;
      return (i < 3 ? arrow(x + radius + 5, cy, xs[i + 1] - radius - 10, cy) : '') + `<circle cx="${x}" cy="${cy}" r="${radius}" fill="white" stroke="${C.rule}" stroke-width="3"/><circle data-epoch="${i + 1}" data-complete="${n(extent)}" cx="${x}" cy="${cy}" r="${radius}" fill="none" stroke="${C.blue}" stroke-width="7" stroke-dasharray="${n(circumference * extent)} ${n(circumference)}" transform="rotate(-90 ${x} ${cy})"/>` + txt(x, cy + (compact ? 8 : 11), i + 1, compact ? 28 : 38, center);
    }).join('') + txt(mid, compact ? 260 : 290, 'Minibatches → updates → another pass', compact ? 24 : 30, center) + txt(mid, compact ? 440 : 510, 'A procedure, not a reported performance curve', compact ? 21 : 26, center);
  }
  if (stage === 11) return txt(compact ? 140 : 285, compact ? 73 : 80, 'Used to fit', compact ? 25 : 34, center) + txt(compact ? 390 : 785, compact ? 73 : 80, 'Reserved', compact ? 25 : 34, center) + rule(compact ? 280 : 580, compact ? 105 : 120, compact ? 280 : 580, compact ? 370 : 430, C.rule, 'stroke-dasharray="6 6"') + lines(compact ? 390 : 785, compact ? 130 : 135, ['Predict; compare', 'with reference labels'], compact ? 20 : 26, center) + txt(mid, compact ? 405 : 470, 'No updates from these reserved sentences', compact ? 22 : 28, center) + txt(mid, compact ? 443 : 520, 'No empirical validation result is available here', compact ? 20 : 24, center);
  if (stage === 12) return txt(mid, compact ? 60 : 60, compact ? 'Training embeddings in two dimensions' : 'Training representations in two dimensions', compact ? 24 : 32, center) + rule(compact ? 55 : 100, compact ? 365 : 420, compact ? 455 : 935, compact ? 365 : 420) + rule(compact ? 55 : 100, compact ? 365 : 420, compact ? 55 : 100, compact ? 105 : 100) + illustrativeClasses.map((label, k) => txt(compact ? 95 + k * 153 : 210 + k * 290, compact ? 402 : 470, label, compact ? 20 : 27, `${center} fill="${colors[k]}"`)).join('') + txt(mid, compact ? 445 : 525, compact ? 'Illustrative positions; no projection supplied' : 'Illustrative positions, not a computed projection', compact ? 20 : 24, center);
  return illustrativeClasses.map((label, k) => txt(compact ? 95 + k * 153 : 210 + k * 290, compact ? 102 : 100, label, compact ? 20 : 28, `${center} fill="${colors[k]}"`)).join('') + arrow(mid, compact ? 225 : 260, mid, compact ? 265 : 325) + lines(mid, compact ? 308 : 375, ['A measure of speech patterns', 'requires a substantive interpretation'], compact ? 24 : 34, center) + txt(mid, compact ? 440 : 510, compact ? 'Speech categories need interpretation' : 'Categories help investigate an argument; they do not prove it', compact ? 20 : 26, center);
}

export function render(position, { compact = false } = {}) {
  const p = Number.isFinite(position) ? Math.max(0, Math.min(scenes.length - 1, position)) : 0;
  // Show one legible annotation state, synchronized with the shell's caption.
  // Observation coordinates and numerical updates remain continuous functions
  // of scroll position; reverse scroll restores exactly the same geometry.
  const annotation = stateDrawing(Math.round(p), compact, p);
  return `<defs><marker id="peloton-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="${C.rule}"/></marker></defs><style>text{font-family:Georgia,'Times New Roman',serif}</style><g data-lesson="peloton-fine-tuning" data-position="${n(p)}" data-layout="${compact ? 'compact' : 'desktop'}">${annotation}${observationLayer(p, compact)}</g>`;
}

export const lesson = {
  id: 'peloton-fine-tuning',
  title: 'Fine-tuning an encoder',
  subtitle: 'Classifying Peloton instructor speech',
  intro: 'A pretrained language encoder can become a sentence classifier without learning language from scratch. Follow a Peloton transcript workflow and see precisely what the labels, parameters, and validation results can tell us.',
  scenes,
  render,
  sources: [
    { label: 'Tara Isabella Burton, “The sin and self-care of Peloton”', url: 'https://religionnews.com/2019/12/09/the-sin-and-self-care-of-peloton/', note: 'The motivation is paraphrased, not quoted.' },
    { label: 'Tara Isabella Burton, Strange Rites', url: 'https://www.hachettebookgroup.com/titles/tara-isabella-burton/strange-rites/9781541762527/?lens=publicaffairs', note: 'Background for the interpretation of contemporary self-improvement.' },
    { label: 'DistilBERT model documentation', url: 'https://huggingface.co/docs/transformers/model_doc/distilbert', note: 'Encoder and sequence-classification architecture.' },
    { label: 'DistilBERT: a distilled version of BERT', url: 'https://arxiv.org/abs/1910.01108', note: 'Sanh et al. describe the pretrained encoder.' },
  ],
  takeaways: [
    'The labeling scheme defines what the classifier learns to reproduce.',
    'Full fine-tuning updates the pretrained encoder as well as the task-specific head.',
    'Cross-entropy connects a target label to a gradient update through both parts of the model.',
    'A stratified validation set measures agreement on reserved sentences; selecting a checkpoint with it makes it part of model selection.',
    'An embedding picture can suggest structure without establishing held-out performance or validating the substantive interpretation.',
  ],
  dataNote: 'The workflow is based on the user-supplied fine_tuning_distilbert.py: Gemini Flash labeling, a stratified 80/20 split, a pretrained DistilBERT sequence classifier, four training epochs, and validation after each epoch. The CSV, label inventory, trained weights, predictions, and evaluation results were not supplied. Coaching, Music, and Encouragement are illustrative categories; the 30 observations, token display, and embedding positions are schematic. Numerical logits, probabilities, loss, and parameter changes are calculated from an explicit tiny surrogate in model.mjs and are not DistilBERT outputs. The three displayed demonstration sentences preserve the script’s wording exactly. The opening discussion paraphrases the motivation from Burton; it is not a verified book quotation.',
};

// Concise captions retain the argument on short phone screens.
const shortCaptions = [
  [
    "Self-love as an obligation",
    "Paraphrasing Burton: equal parts Ayn Rand and John Calvin—your only duty is to yourself, and you never do well enough."
  ],
  [
    "Classify an utterance",
    "Turn an instructor’s sentence into a topic prediction. Coaching, Music, and Encouragement are illustrative categories."
  ],
  [
    "Learn from provisional labels",
    "Gemini Flash labels supply the targets. Check their quality before treating the classifier as a research measure."
  ],
  [
    "Split before fitting",
    "Keep 20% of each topic for validation. These sentences supply no gradient updates."
  ],
  [
    "Tokenize the sentence",
    "Token IDs and an attention mask become the encoder’s inputs. Vocabulary numbers identify tokens; they are not meaningful coordinates."
  ],
  [
    "Use pretrained representations",
    "The encoder starts with learned language representations. Its first token’s final state supplies the classification head."
  ],
  [
    "Produce category scores",
    "The head maps the sentence representation to one score per topic. These logits are not yet probabilities."
  ],
  [
    "Let the label define loss",
    "Softmax converts scores to probabilities. Cross-entropy penalizes low probability for the supplied correct topic."
  ],
  [
    "Update encoder and head",
    "The gradient reaches both parts. One illustrative update raises the target probability and lowers this example’s loss."
  ],
  [
    "What can change?",
    "A frozen encoder stays fixed while the head learns. This workflow fine-tunes both."
  ],
  [
    "Train for four epochs",
    "Repeat minibatch updates over four passes. Validate after each pass; validation supplies no gradients."
  ],
  [
    "Evaluate reserved sentences",
    "Compare held-out predictions with reference labels. Validation loss selects the checkpoint; final testing requires new data."
  ],
  [
    "Inspect the embeddings",
    "A two-dimensional picture can reveal structure, but separated training clusters do not establish held-out performance."
  ],
  [
    "Return to the question",
    "Topic frequencies describe speech patterns. Connecting them to Burton’s argument requires a defensible interpretation of the categories."
  ]
];
shortCaptions.forEach(([title, body], index) => { lesson.scenes[index].shortTitle=title; lesson.scenes[index].shortBody=body; });
