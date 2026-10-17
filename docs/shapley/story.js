import { majority as game, fraction } from './game.mjs';
import { createWeightView } from './weight-view.js';
import { createFormulaView } from './formula-view.js';
import { electoralResults } from './electoral-college.mjs';
import { createElectoralView } from './electoral-view.js';
import { createElectoralMapView } from './electoral-map-view.js';

const $ = selector => document.querySelector(selector);
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
const mobile = window.matchMedia('(max-width: 760px)');
const stage = $('.stage-shell');
const stageViz = $('#stage-viz');
const svgNS = 'http://www.w3.org/2000/svg';
const escapeMath = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const payoffMath = node => `<math class="coalition-value-formula"><mrow><mi>v</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo>${node.members.length ? `<mrow><mo stretchy="false">{</mo>${node.members.map(player => `<mi mathvariant="normal">${escapeMath(player)}</mi>`).join('<mo>,</mo>')}<mo stretchy="false">}</mo></mrow>` : '<mo lspace="0" rspace="0">∅</mo>'}<mo stretchy="false">)</mo></mrow><mo>=</mo><mn>${node.value}</mn></mrow></math>`;
const steps = [...document.querySelectorAll('.step')];
const weightView = createWeightView(game);
steps.forEach((step, i) => { if (!step.id) step.id = `scene-${i + 1}`; });
const positions = { 0: [340, 46], 1: [124, 177], 2: [340, 177], 4: [556, 177], 3: [124, 317], 5: [340, 317], 6: [556, 317], 7: [340, 458] };
const svg = (tag, attrs = {}, text) => {
  const el = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  if (text != null) el.textContent = text;
  return el;
};

stageViz.insertAdjacentHTML('beforeend', `<div class="graph-container" id="graph-container" aria-hidden="true"><div class="layer-labels" aria-hidden="true"><span>0 voters</span><span>1 voter</span><span>2 voters</span><span>3 voters</span></div><div class="graph-canvas" id="graph-canvas"><svg class="graph-svg" viewBox="0 0 680 510" role="img" aria-labelledby="graph-title graph-description"><title id="graph-title">Coalition Hasse diagram</title><desc id="graph-description">A majority-voting game. Groups with fewer than two voters fail; groups with two or three pass. Each arrow adds one voter.</desc></svg><div class="graph-nodes" id="graph-nodes"></div></div></div><div class="graph-caption" id="graph-caption" aria-live="polite"></div>`);

const graphSvg = $('.graph-svg');
const defs = svg('defs');
for (const [id, color] of [['neutral', 'var(--edge-neutral)'], ['active', 'var(--blue)']]) {
  const marker = svg('marker', { id: `arrow-${id}`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' });
  marker.append(svg('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: color })); defs.append(marker);
}
graphSvg.append(defs);
const edgeEls = new Map(), nodeEls = new Map();
for (const edge of game.edges) {
  const [x1, y1] = positions[edge.from], [x2, y2] = positions[edge.to];
  const dy = y2 - y1, dx = x2 - x1, inset = 31;
  const start = [x1 + dx * inset / dy, y1 + inset], end = [x2 - dx * inset / dy, y2 - inset];
  const group = svg('g', { class: 'graph-edge', 'data-edge': edge.id });
  const line = svg('path', { d: `M${start[0]},${start[1]} L${end[0]},${end[1]}`, class: 'edge-line', 'marker-end': 'url(#arrow-neutral)' });
  const label = svg('g', { class: 'edge-label', transform: `translate(${(start[0] + end[0]) / 2},${(start[1] + end[1]) / 2})` });
  label.append(svg('rect', { x: -31, y: -19, width: 62, height: 37, rx: 4 }));
  label.append(svg('text', { class: 'edge-delta', 'text-anchor': 'middle', y: -1 }, edge.delta ? `+${edge.delta}` : '0'));
  label.append(svg('text', { class: 'edge-weight', 'text-anchor': 'middle', y: 13 }, `${edge.count}/${game.totalOrders}`));
  group.append(line, label); graphSvg.append(group); edgeEls.set(edge.id, { group, line });
}
for (const node of game.nodes) {
  const [x, y] = positions[node.mask], button = document.createElement('button');
  button.type = 'button'; button.className = `coalition-node ${node.value ? 'is-winning' : ''}`; button.dataset.mask = node.mask;
  button.style.left = `${100 * x / 680}%`; button.style.top = `${100 * y / 510}%`; button.disabled = true;
  button.setAttribute('aria-label', `${node.members.length ? node.members.join(' and ') : 'Empty coalition'}: ${node.value ? 'Pass' : 'Fail'}, value ${node.value}`);
  button.innerHTML = `<span class="coalition-set">${node.label}</span><span class="coalition-payoff" aria-hidden="true"><span class="coalition-value">${node.value ? 'PASS' : 'FAIL'} <b>${node.value}</b></span>${payoffMath(node)}</span>`;
  button.addEventListener('click', () => inspectNode(node.mask)); $('#graph-nodes').append(button); nodeEls.set(node.mask, button);
}
const calc = document.createElement('div'); calc.className = 'calculation'; calc.hidden = true;
calc.innerHTML = `<div class="calculation-heading"><span>CONTRIBUTION × WEIGHT</span><span id="credit-label">A’s credit</span></div><div class="terms" id="terms"></div><div class="calculation-total"><span id="total-label">Shapley value of A</span><strong id="total-value">1/3</strong></div>`;
stageViz.append(calc);
const controls = document.createElement('div'); controls.className = 'explore-controls'; controls.hidden = true;
controls.innerHTML = `<div class="player-selector" role="group" aria-label="Select a voter"><span>Follow a voter</span>${game.players.map((p, i) => `<button type="button" data-select-player="${i}" aria-pressed="${i === 0}">${p}</button>`).join('')}</div><div class="replay-controls"><label for="order-select">Imagined order</label><div class="replay-row"><select id="order-select">${game.orders.map((o, i) => `<option value="${i}">${o.label}</option>`).join('')}</select><button type="button" id="replay">Replay</button></div></div><div class="edge-inspector"><span class="eyebrow">Inspect an edge</span><div id="edge-choices"></div><p id="edge-detail" aria-live="polite"></p></div>`;
steps[8].querySelector('.step-content').append(controls);
const compare = document.createElement('div'); compare.className = 'order-comparison';
compare.innerHTML = `<span>Compare imagined orders</span><div><button type="button" data-compare-order="2" aria-pressed="true">B → A → C</button><button type="button" data-compare-order="0" aria-pressed="false">A → B → C</button></div><p class="small-note">These are accounting orders, not the actual chronology of voting.</p>`;
steps[5].querySelector('.step-content').append(compare);
const list = document.createElement('div'); list.className = 'orders-list';
list.innerHTML = `<div class="orders-list-heading"><span>IMAGINED ORDER</span><span>A JOINS AFTER</span></div>${game.orders.map((o, i) => { const e = o.path.find(e => e.playerIndex === 0); return `<button type="button" data-show-order="${i}"><span>${o.label}</span><span>${game.nodes[e.from].label}</span></button>`; }).join('')}`;
steps[6].querySelector('.step-content').append(list);
steps[6].querySelector('.step-content').insertAdjacentHTML('beforeend', '<p class="small-note">A is first, second, or third in two orders each. Every arrival position receives total weight 1/3.</p>');
const weights = document.createElement('div'); weights.className = 'weights-summary';
weights.innerHTML = `<span class="eyebrow">BEFORE A JOINS</span><div>${[0, 2, 4, 6].map(mask => { const e = game.edges.find(e => e.from === mask && e.playerIndex === 0); return `<button type="button" data-explain-weight="${e.id}" aria-haspopup="dialog" aria-controls="weight-dialog" aria-label="Explain A joining ${game.nodes[mask].label}, weight ${e.count} out of 6"><b>${game.nodes[mask].label}</b><span>${e.count} of 6</span></button>`; }).join('')}</div>`;
steps[7].querySelector('.step-content').append(weights);
const weightTrigger = document.createElement('button');
weightTrigger.type = 'button'; weightTrigger.className = 'weight-explainer-trigger';
weightTrigger.textContent = 'See the six paths behind these weights';
weightTrigger.setAttribute('aria-haspopup', 'dialog'); weightTrigger.setAttribute('aria-controls', 'weight-dialog');
weightTrigger.addEventListener('click', () => explainWeight('0-1', weightTrigger));
steps[7].querySelector('.step-content').append(weightTrigger);
document.querySelectorAll('[data-explain-weight]').forEach(button => button.addEventListener('click', () => explainWeight(button.dataset.explainWeight, button)));

// Keep one stable picture and caption on screen; sections provide the scroll track.
// Move the actual nodes so their controls and event listeners retain their state.
const sceneContents = steps.map(step => step.querySelector('.step-content'));
const sceneCaption = document.createElement('div');
sceneCaption.id = 'scene-caption'; sceneCaption.className = 'scene-caption';
const navigation = document.createElement('nav');
navigation.id = 'scene-navigation'; navigation.setAttribute('aria-label', 'Story steps');
navigation.innerHTML = '<button type="button" id="scene-previous">Previous</button><span id="scene-position"></span><button type="button" id="scene-next">Next</button>';
stage.append(sceneCaption, navigation);
const sceneEquations = new Map();
for (const index of [9, 19]) {
  const equation = sceneContents[index].querySelector('.value-definition');
  if (!equation) continue;
  const holder = document.createElement('div');
  holder.className = 'scene-equation'; holder.dataset.scene = String(index);
  holder.hidden = true; holder.append(equation); stageViz.append(holder);
  sceneEquations.set(index, holder);
}
const detailDialog = document.createElement('dialog');
detailDialog.id = 'scene-detail-dialog'; detailDialog.className = 'scene-detail-dialog';
detailDialog.setAttribute('aria-labelledby', 'scene-detail-title');
detailDialog.innerHTML = '<div class="scene-detail-top"><h2 id="scene-detail-title">More detail</h2><button type="button" id="scene-detail-close">Close</button></div><div id="scene-detail-body"></div>';
document.body.append(detailDialog);
let detailTrigger = null;
const sceneDetails = new Map();
function closeSceneDetails() {
  if (!detailDialog.open) return;
  detailDialog.close();
}
detailDialog.addEventListener('close', () => {
  const content = detailDialog.querySelector('#scene-detail-body').firstElementChild;
  if (content) { content.hidden = true; sceneContents[Number(content.dataset.scene)].append(content); }
  detailTrigger?.focus({ preventScroll: true });
});
$('#scene-detail-close').addEventListener('click', closeSceneDetails);
for (const [index, content] of sceneContents.entries()) {
  let details = content.querySelector('.scene-detail-content');
  if (index === 8) {
    if (!details) { details = document.createElement('div'); details.className = 'scene-detail-content'; content.append(details); }
    details.append(controls);
    content.append(controls.querySelector('.player-selector'));
    const inspection = controls.querySelector('#edge-detail');
    inspection.className = 'explore-inspection'; inspection.hidden = true; content.append(inspection);
  }
  if (index === 16 && details) {
    const equation = content.querySelector(':scope > .value-definition');
    if (equation) details.querySelector('p').after(equation);
  }
  if (!details) continue;
  details.hidden = true; details.dataset.scene = String(index); sceneDetails.set(index, details);
  const trigger = document.createElement('button');
  trigger.type = 'button'; trigger.className = 'scene-detail-trigger';
  trigger.textContent = index === 8 ? 'More exploration' : 'More detail';
  trigger.setAttribute('aria-haspopup', 'dialog'); trigger.setAttribute('aria-controls', 'scene-detail-dialog');
  trigger.addEventListener('click', () => {
    stopPlayback(); detailTrigger = trigger;
    $('#scene-detail-title').textContent = content.querySelector('h2').textContent;
    details.hidden = false; $('#scene-detail-body').append(details); detailDialog.showModal();
    $('#scene-detail-close').focus();
  });
  content.append(trigger);
}
sceneContents.forEach(content => { content.hidden = true; });
document.body.classList.add('is-presentation');

let current = -1, selectedPlayer = 0, inspectedEdge = null, pathState = null, playbackTimer = null, scrollFrame = 0, nodeNotationVisible = false;
const kickers = ["THE RULE", "PASS OR FAIL", "THE SMALLEST GROUPS", "ADD C", "COMPLETE THE PAIRS", "THE COMPLETE DIAGRAM", "COUNT THE ORDERS", "THE WEIGHTED AVERAGE", "THE HASSE DIAGRAM", "THE VALUE FUNCTION", "THE SUMMATION", "THE CONTRIBUTION", "BEFORE A", "AFTER A", "THE WEIGHT", "THE SHAPLEY VALUE", "THE ELECTORAL COLLEGE", "ELECTORAL GEOGRAPHY", "ELECTORAL VOTES", "SHAPLEY-ADJUSTED VOTES"]
  .map((label, index) => `${String(index + 1).padStart(2, '0')} / ${label}`);
const counts = ["3 voters · 2 votes to pass", "2 votes to pass", "4 groups · 3 edges", "6 coalitions · 5 edges", "7 coalitions · 9 edges", "8 groups · 12 edges", "6 imagined orders", "4 terms · 1 Shapley value", "Majority voting · 3 voters", "0 means failure · 1 means success", "i = A · n = 3", "One joining edge", "Order the voters in S", "Order the remaining voters", "Matching orders / all orders", "Four edges · one weighted sum", "51 blocs · 270 votes to win", "50 states + DC · 2024 / 2028", "538 electoral votes", "538 adjusted votes"];
const formulaView = createFormulaView(game, {
  onCoalitionChange: () => { updateGraph(); updateCaption(); },
  onExplainWeight: explainWeight
});
formulaView.element.hidden = true;
stageViz.append(formulaView.element);
const electoralView = createElectoralView(electoralResults);
electoralView.element.hidden = true;
stageViz.append(electoralView.element);
const electoralMapView = createElectoralMapView(electoralResults, {
  onPhaseChange: phase => navigateScene(17 + phase)
});
electoralMapView.element.hidden = true;
stageViz.append(electoralMapView.element);
function explainWeight(edgeId, source) {
  // Keep the story still while its optional explanation is open.
  clearTimeout(playbackTimer); playbackTimer = null;
  $('#replay').textContent = 'Replay'; $('#replay').disabled = false;
  weightView.open(edgeId, source);
}
function stopPlayback() {
  clearTimeout(playbackTimer); playbackTimer = null; pathState = null;
  const b = $('#replay'); if (b) { b.textContent = 'Replay'; b.disabled = false; }
}
function setNodeNotation(visible) {
  if (visible === nodeNotationVisible) return;
  nodeNotationVisible = visible;
  stage.classList.toggle('has-payoff-notation', visible);
  stage.dataset.payoffNotation = String(visible);
  for (const node of game.nodes) {
    const name = node.members.length ? node.members.join(' and ') : 'Empty coalition';
    nodeEls.get(node.mask).setAttribute('aria-label', visible ? `${name}: v of ${name} equals ${node.value}` : `${name}: ${node.value ? 'Pass' : 'Fail'}, value ${node.value}`);
  }
  updateCaption();
}
function showStep(index) {
  if (index === current) return;
  closeSceneDetails();
  if (current >= 0) {
    sceneContents[current].hidden = true;
    steps[current].append(sceneContents[current]);
  }
  stopPlayback(); current = index; inspectedEdge = null; stage.dataset.scene = current;
  const inFormula = current >= 10 && current < 16;
  const inElectoral = current >= 16;
  const inElectoralMap = current >= 17;
  stage.classList.toggle('is-formula-scene', inFormula);
  stage.classList.toggle('is-electoral-scene', inElectoral);
  document.body.classList.toggle('is-electoral-section', inElectoral);
  stage.classList.toggle('is-electoral-map-scene', inElectoralMap);
  stage.classList.toggle('is-value-function-scene', current === 9);
  stage.classList.toggle('is-pair-level-scene', current === 3 || current === 4);
  if (current !== 9) setNodeNotation(false);
  stage.setAttribute('aria-label', inElectoral ? (current === 16 ? 'A pivotal Electoral College state' : 'Electoral College geography, electoral votes, and Shapley-adjusted votes') : current < 2 ? (current === 1 ? 'Majority voting rule' : 'Voting illustration') : current === 9 ? 'Coalition diagram and voting rule' : inFormula ? 'Coalition diagram and Shapley formula' : 'Coalition diagram and Shapley calculation');
  $('#voters').hidden = current !== 0;
  $('#outcome').hidden = current !== 0;
  $('#voters').setAttribute('aria-hidden', String(current !== 0));
  $('#outcome').setAttribute('aria-hidden', String(current !== 0));
  $('#majority-rule').hidden = current !== 1;
  steps.forEach((s, i) => s.classList.toggle('is-active', i === current));
  $('#stage-kicker').textContent = kickers[current]; $('#stage-count').textContent = counts[current];
  $('#graph-container').setAttribute('aria-hidden', String(current < 2 || inElectoral));
  controls.hidden = current !== 8; calc.hidden = current < 7 || current >= 9;
  $('#edge-detail').hidden = true;
  formulaView.element.hidden = !inFormula;
  if (inFormula) formulaView.update(current - 10);
  electoralView.element.hidden = current !== 16;
  if (current === 16) electoralView.update(0);
  electoralMapView.element.hidden = !inElectoralMap;
  if (inElectoralMap) electoralMapView.update(current - 17);
  else electoralMapView.pause();
  for (const [scene, equation] of sceneEquations) {
    // Reserve the final equation's space across all maps, preserving screen scale.
    const reserve = scene === 19 && inElectoralMap;
    equation.hidden = scene !== current && !reserve;
    equation.style.visibility = reserve && current !== 19 ? 'hidden' : '';
    equation.setAttribute('aria-hidden', String(scene !== current));
  }
  sceneContents[current].hidden = false; sceneCaption.append(sceneContents[current]);
  reserveMapCaptionHeight();
  $('#scene-position').textContent = `${current + 1} / ${steps.length}`;
  $('#scene-previous').textContent = current === 0 ? 'Title' : 'Previous';
  $('#scene-next').textContent = current === steps.length - 1 ? 'Back to title' : 'Next';
  if (current === 9) setNodeNotation(true);
  nodeEls.forEach(b => { b.disabled = current !== 8; b.classList.remove('is-inspected'); });
  if (current === 5) pathState = { orderIndex: 2, count: 3 };
  if (current !== 8) selectedPlayer = 0;
  updateSelectedControls(); updateGraph(); updateCalculation(); updateCaption();
}
function updateGraph() {
  const inFormula = current >= 10 && current < 16;
  const selectedFormulaEdge = inFormula ? formulaView.edge : null;
  const allFormulaEdges = inFormula && (formulaView.phase === 0 || formulaView.phase === 5);
  const partialPairs = current === 3;
  const maxSize = current === 2 ? 1 : current <= 4 ? 2 : 3;
  const order = pathState ? game.orders[pathState.orderIndex] : null, path = order?.path.slice(0, pathState.count) ?? [];
  nodeEls.forEach((b, mask) => {
    const node = game.nodes[mask], visible = current >= 2 && node.size <= maxSize && (!partialPairs || mask !== 3);
    b.classList.toggle('is-visible', visible); b.style.setProperty('--reveal-delay', `${node.size * 75}ms`);
    const involved = !order || mask === 0 || path.some(e => e.to === mask);
    b.classList.toggle('is-subdued', current >= 5 && !!order && !involved);
    b.classList.toggle('is-path-node', !!order && involved && visible);
    b.classList.toggle('is-formula-source', inFormula && (allFormulaEdges ? !(mask & 1) : mask === selectedFormulaEdge.from));
    b.classList.toggle('is-formula-target', inFormula && !allFormulaEdges && mask === selectedFormulaEdge.to);
  });
  for (const e of game.edges) {
    const addsC = e.id === '1-5' || e.id === '2-6';
    const el = edgeEls.get(e.id), visible = current >= 2 && game.nodes[e.to].size <= maxSize && (!partialPairs || e.from === 0 || addsC);
    let active = partialPairs && addsC;
    if (current >= 6 && !order) active = e.playerIndex === selectedPlayer;
    if (current === 9) active = false;
    if (inFormula) active = allFormulaEdges ? e.playerIndex === 0 : e.id === selectedFormulaEdge.id;
    if (order) active = path.some(p => p.id === e.id);
    el.group.classList.toggle('is-visible', visible); el.group.classList.toggle('is-highlighted', active);
    el.group.classList.toggle('is-decisive', active && e.delta !== 0);
    el.group.classList.toggle('show-label', active && current >= 3);
    el.group.classList.toggle('show-weight', current >= 6 && !order && (!inFormula || formulaView.phase >= 4));
    el.group.classList.toggle('is-inspected', e.id === inspectedEdge);
    el.line.setAttribute('marker-end', active ? 'url(#arrow-active)' : 'url(#arrow-neutral)');
    el.line.style.strokeWidth = current >= 6 && !order && active ? 2 + e.weight * 7 : active ? 3 : 1.35;
    if (visible && active) graphSvg.append(el.group);
  }
  document.querySelectorAll('[data-show-order]').forEach(b => b.classList.toggle('is-current', !!pathState && Number(b.dataset.showOrder) === pathState.orderIndex));
}
function updateCaption() {
  const captions = ["Every voter supports the proposal.", "Two or three voters pass (1); zero or one fail (0).", "0 means the group fails. 1 means it passes.", "Add C to coalition", "Every two-voter coalition passes.", "Every path ends at the same full group.", "Edge labels: contribution above, fraction of orders below.", "Click a term to see which of the six paths give it its weight.", "Click a term to explain its weight, or inspect an edge below.", "The value function names the numbers already on the nodes.", "Four coalitions without A. Four starting nodes.", "The bracket measures the change along the selected edge.", "Hold the coalition fixed; count its internal orders.", "The minus one removes A from the remaining voters.", "The factorial ratio is the fraction of paths using this edge.", "The formula adds the same four weighted contributions.", "The state that first reaches 270 receives the contribution of one.", "Geographic shapes show where the electoral votes are.", "One square represents one electoral vote.", "One square represents one Shapley-adjusted vote; boundary squares can be fractional."];
  const text = current === 9 && nodeNotationVisible ? 'The labels now write the same numbers as the value function.' : captions[current];
  $('#stage-bottom').innerHTML = `<span class="legend-mark"></span><span>${text}</span>`;
  const caption = $('#graph-caption');
  if (pathState) {
    const o = game.orders[pathState.orderIndex], decisive = o.path.find(e => e.delta === 1), finished = pathState.count === 3;
    const joining = o.path[Math.max(0, pathState.count - 1)].player;
    caption.innerHTML = `<span class="path-order">${o.label}</span><span>${finished ? `${decisive.player} supplies the deciding vote: +1` : `Adding ${joining}…`}</span>`;
  } else if (current === 3) caption.innerHTML = '<span class="path-order">Add C to coalition</span>';
  else if (current === 6) caption.innerHTML = `<span class="path-order">Follow ${game.players[selectedPlayer]}</span><span>Four joining edges. Six equally weighted orders.</span>`;
  else caption.innerHTML = '';
}
function updateCalculation() {
  const player = game.players[selectedPlayer];
  const edges = game.edges.filter(e => e.playerIndex === selectedPlayer).sort((a, b) => game.nodes[a.from].size - game.nodes[b.from].size || a.from - b.from);
  $('#credit-label').textContent = `${player}’s credit`; $('#total-label').textContent = `Shapley value of ${player}`; $('#total-value').textContent = fraction(game.shares[selectedPlayer]);
  $('#terms').innerHTML = edges.map(e => `<button type="button" class="term ${e.delta ? 'is-nonzero' : ''}" data-term-edge="${e.id}" ${current >= 7 ? '' : 'disabled'} aria-haspopup="dialog" aria-controls="weight-dialog" aria-label="Explain ${player} joining ${game.nodes[e.from].label}: contribution ${e.delta}, weight ${e.count} out of 6"><span class="term-coalition">${game.nodes[e.from].label}</span><span class="term-product">${e.delta} <span>×</span> ${e.count}/${game.totalOrders}</span></button>`).join('<span class="term-plus" aria-hidden="true">+</span>');
  document.querySelectorAll('[data-term-edge]').forEach(b => b.addEventListener('click', () => explainWeight(b.dataset.termEdge, b)));
  $('#edge-choices').innerHTML = edges.map(e => `<button type="button" data-inspect-edge="${e.id}" aria-pressed="${e.id === inspectedEdge}">${game.nodes[e.from].label} → ${game.nodes[e.to].label}</button>`).join('');
  document.querySelectorAll('[data-inspect-edge]').forEach(b => b.addEventListener('click', () => inspectEdge(b.dataset.inspectEdge)));
  if (current === 8 && !inspectedEdge) $('#edge-detail').textContent = `Select one of ${player}’s joining edges to see its contribution and weight.`;
}
function updateSelectedControls() { document.querySelectorAll('[data-select-player]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.selectPlayer) === selectedPlayer))); }
function selectPlayer(i) {
  stopPlayback(); selectedPlayer = i; inspectedEdge = null; nodeEls.forEach(b => b.classList.remove('is-inspected'));
  $('#edge-detail').hidden = true;
  updateSelectedControls(); updateGraph(); updateCalculation(); updateCaption();
}
function inspectEdge(id) {
  stopPlayback(); inspectedEdge = id; const e = game.edges.find(e => e.id === id);
  nodeEls.forEach((b, mask) => b.classList.toggle('is-inspected', mask === e.from || mask === e.to));
  updateGraph(); updateCalculation(); updateCaption();
  $('#edge-detail').textContent = `${game.nodes[e.from].label} → ${game.nodes[e.to].label}: contribution ${e.delta}; weight ${e.count}/${game.totalOrders}. Weighted credit: ${fraction(e.delta * e.weight)}.`;
  $('#edge-detail').hidden = false; closeSceneDetails();
}
function inspectNode(mask) {
  stopPlayback(); inspectedEdge = null; nodeEls.forEach((b, i) => b.classList.toggle('is-inspected', i === mask));
  const node = game.nodes[mask]; updateGraph(); updateCalculation(); updateCaption();
  $('#edge-detail').textContent = `${node.label}: ${node.size} ${node.size === 1 ? 'voter' : 'voters'}, ${node.value ? 'passes' : 'fails'}. Value: ${node.value}.`;
  $('#edge-detail').hidden = false;
}
function playOrder(orderIndex, animate = true) {
  stopPlayback(); inspectedEdge = null; nodeEls.forEach(b => b.classList.remove('is-inspected'));
  $('#edge-detail').hidden = true;
  pathState = { orderIndex, count: animate && !reduced.matches ? 0 : 3 }; const replay = $('#replay');
  if (current === 8) { replay.disabled = pathState.count < 3; replay.textContent = pathState.count < 3 ? 'Playing…' : 'Replay'; }
  const render = () => { updateGraph(); updateCaption(); }; updateCalculation(); render();
  if (pathState.count < 3) {
    const advance = () => {
      if (!pathState) return;
      pathState.count++; render();
      if (pathState.count < 3) playbackTimer = setTimeout(advance, 950);
      else if (current === 8) { replay.disabled = false; replay.textContent = 'Replay'; }
    };
    playbackTimer = setTimeout(advance, 300);
  }
}
document.querySelectorAll('[data-select-player]').forEach(b => b.addEventListener('click', () => selectPlayer(Number(b.dataset.selectPlayer))));
document.querySelectorAll('[data-compare-order]').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('[data-compare-order]').forEach(other => other.setAttribute('aria-pressed', String(other === b)));
  playOrder(Number(b.dataset.compareOrder));
}));
document.querySelectorAll('[data-show-order]').forEach(b => b.addEventListener('click', () => playOrder(Number(b.dataset.showOrder))));
$('#replay').addEventListener('click', () => { playOrder(Number($('#order-select').value)); closeSceneDetails(); });
$('#order-select').addEventListener('change', () => { playOrder(Number($('#order-select').value), false); closeSceneDetails(); });
function updateScroll() {
  scrollFrame = 0; const header = mobile.matches ? 54 : 68;
  const anchor = header + 2;
  let index = 0; for (let i = 0; i < steps.length; i++) if (steps[i].getBoundingClientRect().top <= anchor) index = i;
  showStep(index);
  // Scrolling through this passage visits all six accounting paths, then
  // gathers them into the four weighted joining edges. Buttons can replay any.
  if (current === 6) {
    const rect = steps[6].getBoundingClientRect();
    const phase = Math.min(6, Math.floor(Math.max(0, anchor - rect.top) / rect.height * 8));
    const next = phase < 6 ? phase : null;
    if (next !== (pathState?.orderIndex ?? null)) {
      stopPlayback();
      pathState = next === null ? null : { orderIndex: next, count: 3 };
      updateGraph(); updateCaption();
    }
  }
  const start = scrollY + steps[0].getBoundingClientRect().top - header;
  const end = scrollY + steps.at(-1).getBoundingClientRect().bottom - header;
  const percentage = Math.max(0, Math.min(100, (scrollY - start) / Math.max(1, end - start) * 100));
  $('#progress-fill').style.width = `${percentage}%`;
}
window.addEventListener('scroll', () => { if (!scrollFrame) scrollFrame = requestAnimationFrame(updateScroll); }, { passive: true });
window.addEventListener('resize', updateScroll);
window.addEventListener('resize', reserveMapCaptionHeight);
document.fonts.ready.then(reserveMapCaptionHeight);
reduced.addEventListener('change', () => { stopPlayback(); updateGraph(); updateCaption(); });
showStep(0); updateScroll();

function goToStep(index, focusDestination = false) {
  showStep(index);
  requestAnimationFrame(() => {
    const header = mobile.matches ? 54 : 68;
    const top = scrollY + steps[index].getBoundingClientRect().top;
    window.scrollTo({ top: top - header, behavior: 'instant' });
    updateScroll();
    if (focusDestination) {
      const heading = sceneContents[index].querySelector('h2');
      heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: true });
    }
  });
}
function stepForHash(hash) {
  return steps.findIndex(step => `#${step.id}` === hash);
}
function goToBeginning(focusDestination = false) {
  stopPlayback();
  window.scrollTo({ top: 0, behavior: 'instant' });
  updateScroll();
  if (focusDestination) $('#beginning').focus({ preventScroll: true });
}
function navigateScene(index) {
  if (index < 0 || index >= steps.length) {
    history.replaceState(null, '', '#beginning'); goToBeginning(true); return;
  }
  history.replaceState(null, '', `#${steps[index].id}`); goToStep(index, true);
}
$('#scene-previous').addEventListener('click', () => navigateScene(current - 1));
$('#scene-next').addEventListener('click', () => navigateScene(current + 1));
document.addEventListener('keydown', event => {
  if (event.defaultPrevented || detailDialog.open || document.querySelector('#weight-dialog[open]') || event.target.closest('input, select, textarea, [contenteditable], dialog')) return;
  if (scrollY < $('.story').offsetTop - innerHeight / 2) return;
  if (event.key === 'ArrowRight') { event.preventDefault(); navigateScene(current + 1); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); navigateScene(current - 1); }
});
document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', event => {
  const hash = link.getAttribute('href'), index = stepForHash(hash);
  if (hash === '#beginning') {
    event.preventDefault(); history.replaceState(null, '', hash); goToBeginning(true);
    return;
  }
  if (index < 0) return;
  event.preventDefault(); history.replaceState(null, '', hash); goToStep(index, true);
}));
function followHash() {
  if (location.hash === '#beginning') { goToBeginning(); return; }
  const index = stepForHash(location.hash);
  if (index >= 0) goToStep(index);
  else updateScroll();
}
window.addEventListener('hashchange', followHash);
window.addEventListener('pageshow', followHash);
followHash();

// Keep numerical edge labels at readable screen size as the graph resizes.
new ResizeObserver(entries => {
  const width = entries[0].contentRect.width;
  if (!width) return;
  const scale = 680 / width;
  graphSvg.querySelectorAll('.edge-delta').forEach(el => { el.style.fontSize = `${14 * scale}px`; el.setAttribute('y', -2 * scale); });
  graphSvg.querySelectorAll('.edge-weight').forEach(el => { el.style.fontSize = `${12 * scale}px`; el.setAttribute('y', 13 * scale); });
  graphSvg.querySelectorAll('.edge-label rect').forEach(el => {
    el.setAttribute('x', -22 * scale); el.setAttribute('y', -17 * scale);
    el.setAttribute('width', 44 * scale); el.setAttribute('height', 34 * scale);
  });
}).observe($('#graph-canvas'));

function sizeVisual() {
  const gap = parseFloat(getComputedStyle(stageViz).gap) || 0;
  const companions = [$('#graph-caption'), calc, formulaView.element, ...sceneEquations.values()]
    .filter(element => !element.hidden && getComputedStyle(element).display !== 'none' && element.getBoundingClientRect().height > 0);
  const reserved = companions.reduce((sum, element) => sum + element.getBoundingClientRect().height + gap, 0);
  stageViz.style.setProperty('--graph-height', `${Math.max(70, stageViz.clientHeight - reserved - 8)}px`);
  if (current === 16) electoralView.update(0);
  if (current >= 17) electoralMapView.resize();
}
function reserveMapCaptionHeight() {
  if (current < 17) { sceneCaption.style.minHeight = ''; return; }
  let height = 0;
  for (const content of sceneContents.slice(17)) {
    const copy = content.cloneNode(true);
    copy.hidden = false; copy.setAttribute('aria-hidden', 'true');
    Object.assign(copy.style, { position: 'absolute', visibility: 'hidden', width: `${sceneCaption.clientWidth}px`, pointerEvents: 'none' });
    sceneCaption.append(copy);
    height = Math.max(height, copy.getBoundingClientRect().height);
    copy.remove();
  }
  sceneCaption.style.minHeight = `${Math.ceil(height)}px`;
}
const visualObserver = new ResizeObserver(sizeVisual);
[stageViz, $('#graph-caption'), calc, formulaView.element, sceneCaption].forEach(element => visualObserver.observe(element));
sizeVisual();
