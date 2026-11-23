// Guided observations from the checked-in checkpoint export. Geometry may
// reveal or rearrange measurements, but never interpolates attention weights.
import {normalizeTrainedData, attentionColor, formatWeight} from './trained-scenes.mjs';

const escapeText = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const percent = value => `${(100 * value).toFixed(1)}%`;
const probability = value => `${(100 * value).toFixed(2)}%`;
const nearOne = value => value > .9999 && value < 1 ? '>99.99%' : percent(value);
const sum = values => values.reduce((total, value) => total + value, 0);

export function createGuidedTrainedScenes(drawing, data) {
  const {m, text, line, rect, circle, C, ease, lerp, clamp} = drawing;
  const {examples, layers, heads} = normalizeTrainedData(data);
  const negative = examples.find(example => example.tokens.includes('don') && example.tokens.includes('good'));
  const positive = examples.find(example => example.tokens.includes('spectacular'));
  if (!negative || !positive || layers < 6 || heads < 11) {
    throw new Error('The guided tour requires the two Infinite Jest examples and their six-layer, twelve-head export.');
  }
  const position = (example, token) => {
    const index = example.tokens.indexOf(token);
    if (index < 0) throw new Error(`The guided example is missing token ${token}.`);
    return index;
  };
  const cls = position(negative, '[CLS]');
  const apostrophe = position(negative, "'");
  const don = position(negative, 'don');
  const tToken = position(negative, 't');
  const previousRead = negative.attentions[0][2][apostrophe][don];
  const nextRead = negative.attentions[0][10][apostrophe][tToken];
  const finalRead = negative.attentions[5][9][cls][apostrophe];
  const goodWeights = [negative, positive].map(example => example.attentions[4][7][position(example, '[CLS]')][position(example, 'good')]);
  const label = (value, x, y, size = 24, color = C.ink, anchor = 'middle', extra = '') =>
    text(value, x, y, size, color, anchor, extra).replace('style="', `style="font-size:${size}px;`);
  const phase = (value, start, end) => ease(clamp((value - start) / (end - start)));
  const title = value => label(value, 320, 31, 26, C.blue);
  const arrow = (x1, y1, x2, y2, color = C.blue, extra = '') => line(x1, y1, x2, y2, color, 2, `marker-end="url(#arrow)" ${extra}`);
  const tokenBox = (token, x, y, width = 104, color = C.blue) =>
    rect(x - width / 2, y - 25, width, 50, '#fff', color, 4, 'stroke-width="1.7"') + label(token, x, y + 9, 29, color);
  const measuredCell = (value, x, y, width, height, description, extra = '') =>
    `<g><title>${escapeText(`${description}: ${formatWeight(value)}`)}</title>${rect(x, y, width, height, attentionColor(value), '#dce4df', 0, `stroke-width=".6" ${extra}`)}</g>`;
  const wrap = (value, length = 45) => {
    const lines = []; let current = '';
    for (const word of value.split(/\s+/)) {
      if (current && `${current} ${word}`.length > length) { lines.push(current); current = word; }
      else current += `${current ? ' ' : ''}${word}`;
    }
    if (current) lines.push(current);
    return lines;
  };

  function localRouting(progress) {
    const tokens = negative.tokens, matrix = negative.attentions[1][9];
    const n = tokens.length, x = 83, y = 103, width = 315, cell = width / n;
    // The lens traverses ordinary-token rows. [CLS] and [SEP] remain visible
    // with their original values; the final ordinary-token row is not omitted.
    const first = position(negative, 'good'), last = position(negative, 'same');
    const cursor = lerp(first, last, phase(progress, .13, .88));
    const destination = Math.round(cursor);
    const source = matrix[destination].indexOf(Math.max(...matrix[destination]));
    const weight = matrix[destination][source];
    let out = title('Layer 2 · head 10');
    out += label('Read from →', x + width / 2, 69, 22, C.teal);
    out += label('Position · token', 448, 80, 20, C.muted, 'start');
    out += label('Update', 13, y + width / 2, 22, C.blue, 'middle', `transform="rotate(-90 13 ${y + width / 2})"`);
    for (let row = 0; row < n; row++) {
      out += label(row + 1, x - 12, y + (row + .5) * cell + 6, 17, row === destination ? C.rust : C.muted, 'end');
      out += label(row + 1, x + (row + .5) * cell, y - 9, 17, C.muted);
      out += label(`${row + 1}  ${tokens[row]}`, 448, y + (row + .5) * cell + 6, 20, row === destination || row === source ? C.blue : C.ink, 'start');
      for (let column = 0; column < n; column++) {
        out += measuredCell(matrix[row][column], x + column * cell, y + row * cell, cell, cell,
          `Layer 2 head 10. Destination ${row + 1} ${tokens[row]}, source ${column + 1} ${tokens[column]}`);
      }
    }
    out += rect(x - 2, y + cursor * cell - 1, width + 4, cell + 2, 'none', C.rust, 0, 'stroke-width="2.5"');
    out += rect(x + source * cell, y + destination * cell, cell, cell, 'none', C.ink, 0, 'stroke-width="2.5"');
    out += label(`${tokens[destination]}  reads  ${tokens[source]}`, 320, 456, 28, C.blue);
    out += label(`${nearOne(weight)} of this head’s mixture`, 320, 489, 24, C.ink);
    const scaleX = 174, scaleY = 516, scaleWidth = 228;
    for (let k = 0; k < 80; k++) out += rect(scaleX + k * scaleWidth / 80, scaleY, scaleWidth / 80 + .2, 9, attentionColor(k / 79));
    out += label('0', scaleX - 12, scaleY + 10, 18, C.muted, 'end') + label('1', scaleX + scaleWidth + 12, scaleY + 10, 18, C.muted, 'start');
    out += label('absolute scale', 500, scaleY + 10, 18, C.muted);
    return out;
  }

  function parallelHeads(progress) {
    const incoming = phase(progress, .12, .47), outgoing = phase(progress, .48, .89);
    let out = title('Two heads update the same position');
    out += label('Layer 1 reads the initial token states', 320, 76, 23, C.muted);
    out += tokenBox('don', 125, 128) + tokenBox("'", 320, 128) + tokenBox('t', 515, 128);
    out += arrow(125, 158, 160, 211, C.blue) + arrow(515, 158, 480, 211, C.teal);
    out += line(300, 158, 207, 211, C.muted, 1.4, 'stroke-dasharray="4 4"');
    out += line(340, 158, 433, 211, C.muted, 1.4, 'stroke-dasharray="4 4"');
    out += label('same input row', 320, 195, 20, C.muted);
    const headData = [{x:160, head:3, source:'don', weight:previousRead, color:C.blue}, {x:480, head:11, source:'t', weight:nextRead, color:C.teal}];
    for (const entry of headData) {
      out += rect(entry.x - 117, 218, 234, 108, '#fff', entry.color, 4, 'stroke-width="1.6"');
      out += label(`Head ${entry.head} reads ${entry.source}`, entry.x, 250, 24, entry.color);
      out += label(percent(entry.weight), entry.x, 293, 34, entry.color);
      out += label('Own value projection', entry.x, 353, 20, C.muted);
      out += arrow(entry.x, 365, 320 + (entry.x < 320 ? -30 : 30), 425, entry.color);
      // These markers show a route through the computation, not a measured
      // vector component or a numerical attribution.
      const fromX = entry.source === 'don' ? 125 : 515;
      out += rect(lerp(fromX, entry.x, incoming) - 6, lerp(164, 201, incoming) - 6, 12, 12, entry.color);
      out += circle(lerp(entry.x, 320 + (entry.x < 320 ? -30 : 30), outgoing), lerp(367, 423, outgoing), 6, entry.color);
    }
    out += label('Combine heads + residual + FFN', 320, 394, 22, C.blue);
    out += tokenBox("'", 320, 455, 142);
    out += label('Updated state at the apostrophe', 320, 508, 24, C.blue);
    return out;
  }

  function sameWord(progress) {
    let out = title('Layer 5 · head 8 · [CLS] row');
    const x = 42, width = 556, barHeight = 36;
    [negative, positive].forEach((example, index) => {
      const y = 76 + index * 218, color = index ? C.teal : C.blue;
      const row = example.attentions[4][7][position(example, '[CLS]')];
      const weight = goodWeights[index];
      const rest = sum(row.filter((_, token) => token !== position(example, 'good')));
      const reveal = phase(progress, index ? .38 : .08, index ? .86 : .53);
      out += label(index ? 'Second review' : 'First review', x, y, 22, color, 'start');
      out += label(`P(positive) ${probability(example.positive)}`, x + width, y, 22, color, 'end');
      wrap(example.text).forEach((part, lineIndex) => { out += label(part, x, y + 34 + lineIndex * 27, 23, C.ink, 'start'); });
      const barY = y + 93;
      // The full measured distribution is always laid out at its true width.
      // Scrolling uncovers it with a clip, rather than changing its values.
      const clip = `tour-good-row-${index}`;
      out += `<defs><clipPath id="${clip}">${rect(x, barY - 1, width * reveal, barHeight + 2, '#fff')}</clipPath></defs>`;
      out += rect(x, barY, width, barHeight, '#fff', '#a4aaa8', 0, 'stroke-width="1"');
      out += `<g clip-path="url(#${clip})"><title>${escapeText(`Layer 5 head 8 [CLS] row. good ${formatWeight(weight)}; all other positions ${formatWeight(rest)}. Original row sum ${formatWeight(sum(row))}.`)}</title>`;
      out += rect(x, barY, width * weight, barHeight, color);
      out += rect(x + width * weight, barY, width * rest, barHeight, '#e5e8e6');
      out += line(x + width * weight, barY, x + width * weight, barY + barHeight, C.ink, 1.4) + '</g>';
      out += line(x + width * reveal, barY - 5, x + width * reveal, barY + barHeight + 5, C.ink, 1.8);
      out += label(`good: ${percent(weight)}`, x, barY + 66, 25, color, 'start');
      out += label(`all other positions: ${percent(rest)}`, x + width, barY + 66, 22, C.muted, 'end');
    });
    out += label('A large weight does not specify sentiment direction.', 320, 529, 23, C.ink);
    return out;
  }

  function contextualState(progress) {
    const history = phase(progress, .14, .69), finalRoute = phase(progress, .69, .92);
    const stateX = index => 64 + index * 102;
    let out = title('The token label stays the same');
    out += label(`Earlier: apostrophe reads don (${percent(previousRead)})`, 320, 81, 23, C.blue);
    out += label(`and t (${percent(nextRead)}) in two layer-1 heads.`, 320, 113, 23, C.blue);
    out += label('Five blocks revise its representation', 320, 165, 24, C.muted);
    for (let block = 0; block <= 5; block++) {
      const x = stateX(block);
      if (block < 5) out += arrow(x + 35, 251, x + 65, 251, C.muted);
      out += label(block === 0 ? 'Input' : `After ${block}`, x, 205, 20, C.muted);
      out += rect(x - 35, 220, 70, 103, '#fff', C.blue, 4, 'stroke-width="1.4"');
      out += label("'", x, 254, 37, C.blue);
      out += m(`x_{${apostrophe + 1}}^{(${block})}`, x, 295, 69, 42, 23, C.ink);
    }
    // The moving outline follows successive states of one position. No
    // vector values or purported fractions of causal influence are invented.
    out += rect(lerp(stateX(0), stateX(5), history) - 39, 216, 78, 111, 'none', C.rust, 5, 'stroke-width="2.7"');
    out += label('A state history, not a measured attribution path', 320, 357, 20, C.muted);
    const startX = stateX(5), startY = 333, endX = 360, endY = 426;
    out += arrow(startX, startY, endX, endY, C.blue);
    out += circle(lerp(startX, endX, finalRoute), lerp(startY, endY, finalRoute), 7, C.blue);
    out += label(`Layer 6 · head 10: ${percent(finalRead)}`, 292, 401, 25, C.blue);
    out += tokenBox('[CLS]', 320, 456, 158);
    out += label('Reads the layer-5 state at the apostrophe', 320, 512, 24, C.ink);
    return out;
  }

  function readoutRoute(progress) {
    const selection = phase(progress, .16, .79);
    const tokens = negative.tokens, n = tokens.length, x = 48, y = 113, width = 300, cell = width / n;
    const chosenHead = 9, matrix = negative.attentions[5][chosenHead], rowHeight = 23;
    let out = title('Final layer: keep the [CLS] rows');
    out += label(selection < .5 ? 'Head 10 · all destination rows' : 'All heads · [CLS] destination only', 198, 73, 21, C.blue);
    for (let source = 0; source < n; source++) out += label(source + 1, x + (source + .5) * cell, y - 10, 16, C.muted);
    // Other destinations leave the readout diagram; their measured values
    // remain unchanged. The selected [CLS] row moves into the head summary.
    out += `<g opacity="${1 - selection}">`;
    for (let destination = 0; destination < n; destination++) {
      if (destination === cls) continue;
      const rowY = y + destination * cell + selection * 104;
      out += label(destination + 1, x - 10, rowY + cell / 2 + 6, 17, C.muted, 'end');
      for (let source = 0; source < n; source++) out += measuredCell(matrix[destination][source], x + source * cell, rowY, cell, cell,
        `Layer 6 head 10. Destination ${destination + 1} ${tokens[destination]}, source ${source + 1} ${tokens[source]}`);
    }
    out += '</g>';
    for (let head = 0; head < heads; head++) {
      const row = negative.attentions[5][head][cls];
      const rowY = head === chosenHead ? lerp(y + cls * cell, y + head * rowHeight, selection) : y + head * rowHeight;
      const alpha = head === chosenHead ? 1 : selection;
      out += `<g opacity="${alpha}">`;
      const rowLabel = head === chosenHead && selection < .5 ? cls + 1 : head + 1;
      out += label(rowLabel, x - 10, rowY + rowHeight / 2 + 6, 18, head === chosenHead ? C.blue : C.muted, 'end');
      for (let source = 0; source < n; source++) out += measuredCell(row[source], x + source * cell, rowY, cell, lerp(cell, rowHeight, selection),
        `Layer 6 head ${head + 1}. Destination ${cls + 1} [CLS], source ${source + 1} ${tokens[source]}`);
      if (head === chosenHead) out += rect(x - 1, rowY - 1, width + 2, lerp(cell, rowHeight, selection) + 2, 'none', C.rust, 0, 'stroke-width="2.3"');
      out += '</g>';
    }
    out += `<g opacity="${selection}">`;
    out += label('Head', 24, 101, 16, C.muted);
    out += label('All token positions, including [CLS] and [SEP]', 198, 417, 18, C.muted);
    out += line(364, y, 373, y, C.blue, 1.6) + line(373, y, 373, y + heads * rowHeight, C.blue, 1.6) + line(364, y + heads * rowHeight, 373, y + heads * rowHeight, C.blue, 1.6);
    out += arrow(373, 260, 417, 260, C.blue);
    out += label('Earlier [CLS]', 516, 129, 23, C.blue);
    out += arrow(516, 147, 516, 220, C.blue, 'stroke-dasharray="5 4"');
    out += label('residual', 570, 183, 19, C.muted);
    out += rect(420, 226, 191, 91, '#fff', C.blue, 4, 'stroke-width="1.7"');
    out += label('Final [CLS]', 515, 262, 26, C.blue);
    out += label('after block 6', 515, 292, 21, C.muted);
    out += arrow(515, 324, 515, 365, C.blue);
    out += rect(420, 373, 191, 64, '#fff', C.blue, 4, 'stroke-width="1.7"');
    out += label('Classifier', 515, 414, 26, C.blue);
    out += '</g>';
    out += label('Other final-layer token outputs have no later reader.', 320, 477, 23, C.ink);
    out += label('Earlier layers: any row can supply a later read.', 320, 514, 23, C.muted);
    return out;
  }

  return [
    {
      id:'trained-local-routing', part:'05 · A trained sentiment model', label:'Notice a routing pattern',
      title:'First, notice the stripe.',
      caption:'This head usually reads the next token position. Follow the moving row: the destination reads a source just to its right. All fifteen positions, including the special tokens, remain visible.',
      equation:String.raw`A^{(2,10)}_{i,i+1}>0.9999`,
      detail:'The next-position coefficients between ordinary tokens exceed 99.99%. This is a routing pattern in these examples; it does not identify a sentiment word or a universal role for the head.',
      visual:'The full measured 15 by 15 layer-2, head-10 matrix for the negative review, on an absolute linear zero-to-one color scale. The row lens moves through ordinary-token positions and identifies each largest coefficient.',
      reading:'<p>The nearly off-diagonal stripe connects successive token positions. For example, the position labeled <code>good</code> reads <code>and</code>. This head is useful for seeing a simple information-routing rule before trying to interpret sentiment. The boundary rows behave differently: <code>[CLS]</code> and <code>[SEP]</code> are retained, without masking or renormalizing them. These observations are deliberately selected teaching examples, not a claim that every head has one interpretable job.</p>',
      footer:'Measured attention · negative review · layer 2, head 10 · full matrix', draw:localRouting,
    },
    {
      id:'trained-parallel-heads', part:'05 · A trained sentiment model', label:'Compare parallel heads',
      title:'Two heads gather different context.',
      caption:'At the apostrophe in don’t, head 3 reads don while head 11 reads t. Both work from the same initial states, then contribute different value mixtures to one updated apostrophe representation.',
      equation:String.raw`z_i^{(1,h)}=\sum_j A_{ij}^{(1,h)}v_j^{(1,h)}`,
      detail:`In this review, the coefficients are ${percent(previousRead)} for don and ${percent(nextRead)} for t. These heads run in parallel; neither reads the other head’s newly computed output.`,
      visual:`Layer 1, apostrophe destination. Head 3 reads don with weight ${formatWeight(previousRead)}; head 11 reads t with weight ${formatWeight(nextRead)}. Route markers are schematic and do not depict measured vector components.`,
      reading:`<p>The exact coefficients are ${formatWeight(previousRead)} and ${formatWeight(nextRead)}. Other sources retain ${percent(sum(negative.attentions[0][2][apostrophe].filter((_, index) => index !== don)))} and ${percent(sum(negative.attentions[0][10][apostrophe].filter((_, index) => index !== tToken)))} of the respective rows. Each head has its own learned value projection. Their outputs are combined with the residual and feed-forward operations. A head’s number is local to its layer: head 3 in another layer is a different learned module.</p>`,
      footer:'Measured reads · negative review · layer 1, heads 3 and 11 · apostrophe destination', draw:parallelHeads,
    },
    {
      id:'trained-sentiment-context', part:'05 · A trained sentiment model', label:'Separate weight from sentiment',
      title:'Both reviews give good a large weight.',
      caption:'The same head gives good a large [CLS]-row coefficient in both reviews, although the model predicts opposite sentiments. Attention determines how much of a contextual value vector enters this mixture.',
      equation:String.raw`z_{\mathrm{CLS}}^{(5,8)}=\sum_j A_{\mathrm{CLS},j}^{(5,8)}v_j^{(5,8)}`,
      detail:`The weights on good are ${percent(goodWeights[0])} and ${percent(goodWeights[1])}. Positive-class probabilities are ${probability(negative.positive)} and ${probability(positive.positive)}. A large coefficient does not specify sentiment direction.`,
      visual:`Measured layer-5, head-8 [CLS] rows. Negative review: good ${percent(goodWeights[0])}, model P(positive) ${probability(negative.positive)}. Positive review: good ${percent(goodWeights[1])}, model P(positive) ${probability(positive.positive)}. Each bar includes all remaining measured attention mass on an unchanged zero-to-one scale.`,
      reading:'<p>The label <code>good</code> is shared, but its incoming layer-4 representation depends on the surrounding review. The attention output also depends on the head’s value projection and the subsequent network. The two bars group only the named source and all remaining positions; they do not delete or renormalize any tokens. Comparing these reviews demonstrates that a large attention coefficient is compatible with either predicted sentiment.</p>',
      footer:'Measured attention · both reviews · layer 5, head 8 · [CLS] destination', draw:sameWord,
    },
    {
      id:'trained-contextual-state', part:'05 · A trained sentiment model', label:'Follow the changing representation',
      title:'The apostrophe now has a history.',
      caption:'By layer 6, the apostrophe position has passed through five blocks. Head 10 gives its incoming representation a large [CLS]-row weight. The column still says apostrophe; its vector can contain context.',
      equation:String.raw`z_{\mathrm{CLS}}^{(6,10)}=\sum_j A_{\mathrm{CLS},j}^{(6,10)}v_j^{(6,10)}`,
      detail:`The final read is ${percent(finalRead)}. Earlier, two layer-1 heads read don (${percent(previousRead)}) and t (${percent(nextRead)}) into this position. These observations suggest a question about context; they do not prove a negation circuit.`,
      visual:`For the negative review, layer 6 head 10 gives the apostrophe weight ${formatWeight(finalRead)} in the [CLS] row. Five prior block updates are shown schematically. The diagram does not quantify a causal path from don or t to the prediction.`,
      reading:`<p>The exact final coefficient is ${formatWeight(finalRead)}. Attention reads the <em>incoming</em> layer-5 state at the apostrophe, not the apostrophe’s simultaneously computed layer-6 output. The values are learned affine projections of those layer-5 states, including the checkpoint’s value bias. The token label stays fixed while its representation changes. The earlier local reads make the apostrophe an informative location to inspect, but attention matrices alone do not establish which information survives the intervening blocks or how it affects the sentiment score.</p>`,
      footer:'Measured reads + schematic state history · negative review · final read: layer 6, head 10', draw:contextualState,
    },
    {
      id:'trained-readout-route', part:'05 · A trained sentiment model', label:'Choose where to look',
      title:'Start at the vector the classifier reads.',
      caption:'The classifier reads final [CLS]. Keep its row from each final-layer head, then look backward at the representations those rows read. Other final-layer token outputs are never read again in this classifier.',
      equation:String.raw`\hat p=\operatorname{softmax}\!\bigl(g(x_{\mathrm{CLS}}^{(6)})\bigr)`,
      detail:'Earlier layers are different: any token representation can feed a later [CLS] update. Residual paths also preserve earlier information. Final [CLS] rows are a useful starting point, not a complete attribution.',
      visual:`The measured layer-6 head-10 matrix is filtered to its [CLS] row. The ${heads} final-layer [CLS] rows are then shown together, preserving every source position and the absolute zero-to-one color scale. The classifier reads the final [CLS] representation.`,
      reading:'<p>All heads in a block read the same incoming sequence of states. The non-<code>[CLS]</code> outputs produced by the last block therefore cannot feed that block’s <code>[CLS]</code> output, and there is no later block to read them. Feed-forward layers and layer normalization act within each position, so they introduce no later cross-position route. In earlier blocks, those token outputs remain available to subsequent attention. The summary preserves all twelve heads rather than averaging distinct value projections into an importance score.</p>',
      footer:'Measured final-layer rows + architecture schematic · [CLS] readout · no head averaging', draw:readoutRoute,
    },
  ];
}
