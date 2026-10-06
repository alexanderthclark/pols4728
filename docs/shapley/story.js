import { majority as game, fraction } from './game.mjs';
import { createWeightView } from './weight-view.js';
import { createFormulaView } from './formula-view.js';

const $ = selector => document.querySelector(selector);
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
const mobile = window.matchMedia('(max-width: 760px)');
const stage = $('.stage-shell');
const stageViz = $('#stage-viz');
const svgNS = 'http://www.w3.org/2000/svg';
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
for (const [id, color] of [['neutral', '#c8ced9'], ['active', '#2747dc']]) {
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
  button.innerHTML = `<span class="coalition-set">${node.label}</span><span class="coalition-value">${node.value ? 'PASS' : 'FAIL'} <b>${node.value}</b></span>`;
  button.addEventListener('click', () => inspectNode(node.mask)); $('#graph-nodes').append(button); nodeEls.set(node.mask, button);
}
const calc = document.createElement('div'); calc.className = 'calculation'; calc.hidden = true;
calc.innerHTML = `<div class="calculation-heading"><span>CONTRIBUTION × WEIGHT</span><span id="credit-label">A’s credit</span></div><div class="terms" id="terms"></div><div class="calculation-total"><span id="total-label">Shapley value of A</span><strong id="total-value">1/3</strong></div>`;
stageViz.append(calc);
const controls = document.createElement('div'); controls.className = 'explore-controls'; controls.hidden = true;
controls.innerHTML = `<div class="player-selector" role="group" aria-label="Select a voter"><span>Follow a voter</span>${game.players.map((p, i) => `<button type="button" data-select-player="${i}" aria-pressed="${i === 0}">${p}</button>`).join('')}</div><div class="replay-controls"><label for="order-select">Imagined order</label><div class="replay-row"><select id="order-select">${game.orders.map((o, i) => `<option value="${i}">${o.label}</option>`).join('')}</select><button type="button" id="replay">Replay</button></div></div><div class="edge-inspector"><span class="eyebrow">Inspect an edge</span><div id="edge-choices"></div><p id="edge-detail" aria-live="polite"></p></div>`;
steps[7].querySelector('.step-content').append(controls);
const voteControls = document.createElement('div'); voteControls.className = 'removal-controls';
voteControls.innerHTML = `<span>Try removing</span>${game.players.map((p, i) => `<button type="button" data-remove-player="${i}" aria-pressed="${i === 0}">${p}</button>`).join('')}`;
steps[1].querySelector('.step-content').append(voteControls);
const compare = document.createElement('div'); compare.className = 'order-comparison';
compare.innerHTML = `<span>Compare imagined orders</span><div><button type="button" data-compare-order="2" aria-pressed="true">B → A → C</button><button type="button" data-compare-order="0" aria-pressed="false">A → B → C</button></div><p class="small-note">These are accounting orders, not the actual chronology of voting.</p>`;
steps[4].querySelector('.step-content').append(compare);
const list = document.createElement('div'); list.className = 'orders-list';
list.innerHTML = `<div class="orders-list-heading"><span>IMAGINED ORDER</span><span>A JOINS AFTER</span></div>${game.orders.map((o, i) => { const e = o.path.find(e => e.playerIndex === 0); return `<button type="button" data-show-order="${i}"><span>${o.label}</span><span>${game.nodes[e.from].label}</span></button>`; }).join('')}`;
steps[5].querySelector('.step-content').append(list);
steps[5].querySelector('.step-content').insertAdjacentHTML('beforeend', '<p class="small-note">A is first, second, or third in two orders each. Every arrival position receives total weight 1/3.</p>');
const weights = document.createElement('div'); weights.className = 'weights-summary';
weights.innerHTML = `<span class="eyebrow">BEFORE A JOINS</span><div>${[0, 2, 4, 6].map(mask => { const e = game.edges.find(e => e.from === mask && e.playerIndex === 0); return `<button type="button" data-explain-weight="${e.id}" aria-haspopup="dialog" aria-controls="weight-dialog" aria-label="Explain A joining ${game.nodes[mask].label}, weight ${e.count} out of 6"><b>${game.nodes[mask].label}</b><span>${e.count} of 6</span></button>`; }).join('')}</div>`;
steps[6].querySelector('.step-content').append(weights);
const weightTrigger = document.createElement('button');
weightTrigger.type = 'button'; weightTrigger.className = 'weight-explainer-trigger';
weightTrigger.textContent = 'See the six paths behind these weights';
weightTrigger.setAttribute('aria-haspopup', 'dialog'); weightTrigger.setAttribute('aria-controls', 'weight-dialog');
weightTrigger.addEventListener('click', () => explainWeight('0-1', weightTrigger));
steps[6].querySelector('.step-content').append(weightTrigger);
document.querySelectorAll('[data-explain-weight]').forEach(button => button.addEventListener('click', () => explainWeight(button.dataset.explainWeight, button)));

let current = -1, selectedPlayer = 0, removedPlayer = 0, inspectedEdge = null, pathState = null, playbackTimer = null, scrollFrame = 0;
const kickers = ['01 / THE RULE', '02 / THE REMOVAL PUZZLE', '03 / EMPTY SET + SINGLETONS', '04 / ADD THE PAIRS', '05 / THE COMPLETE LATTICE', '06 / COUNT THE ORDERS', '07 / THE WEIGHTED AVERAGE', '08 / THE HASSE DIAGRAM', '09 / THE SUMMATION', '10 / THE CONTRIBUTION', '11 / BEFORE A', '12 / AFTER A', '13 / THE WEIGHT', '14 / THE SHAPLEY VALUE'];
const counts = ['3 voters · 2 votes to pass', '2 votes still pass', '4 coalitions · 3 edges', '7 coalitions · 9 edges', '8 coalitions · 12 edges', '6 imagined orders', '4 terms · 1 Shapley value', 'Majority voting · 3 voters', 'i = A · n = 3', 'One joining edge', 'Order the voters in S', 'Order the remaining voters', 'Matching orders / all orders', 'Four edges · one weighted sum'];
const formulaView = createFormulaView(game, {
  onCoalitionChange: () => { updateGraph(); updateCaption(); },
  onExplainWeight: explainWeight
});
formulaView.element.hidden = true;
stageViz.append(formulaView.element);
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
function updateRemoval() {
  document.querySelectorAll('.voter').forEach((el, i) => {
    el.classList.toggle('is-removed', current === 1 && i === removedPlayer);
    el.querySelector('.voter-vote').textContent = current === 1 && i === removedPlayer ? 'Removed' : 'Yes';
  });
  $('.outcome-number').textContent = current === 1 ? '2' : '3';
  $('.outcome-result').textContent = current === 1 ? 'Still passes. Change: 0.' : 'The proposal passes';
  document.querySelectorAll('[data-remove-player]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.removePlayer) === removedPlayer)));
}
function showStep(index) {
  if (index === current) return;
  stopPlayback(); current = index; inspectedEdge = null; stage.dataset.scene = current;
  const inFormula = current >= 8;
  stage.classList.toggle('is-formula-scene', inFormula);
  stage.setAttribute('aria-label', current < 2 ? 'Voting illustration' : inFormula ? 'Coalition diagram and Shapley formula' : 'Coalition diagram and Shapley calculation');
  $('#voters').setAttribute('aria-hidden', String(current >= 2));
  $('#outcome').setAttribute('aria-hidden', String(current >= 2));
  steps.forEach((s, i) => s.classList.toggle('is-active', i === current));
  $('#stage-kicker').textContent = kickers[current]; $('#stage-count').textContent = counts[current];
  $('#graph-container').setAttribute('aria-hidden', String(current < 2));
  controls.hidden = current !== 7; calc.hidden = current < 6 || inFormula;
  formulaView.element.hidden = !inFormula;
  if (inFormula) formulaView.update(current - 8);
  updateRemoval();
  nodeEls.forEach(b => { b.disabled = current !== 7; b.classList.remove('is-inspected'); });
  if (current === 4) pathState = { orderIndex: 2, count: 3 };
  if (current !== 7) selectedPlayer = 0;
  updateSelectedControls(); updateGraph(); updateCalculation(); updateCaption();
}
function updateGraph() {
  const inFormula = current >= 8;
  const selectedFormulaEdge = inFormula ? formulaView.edge : null;
  const allFormulaEdges = inFormula && (formulaView.phase === 0 || formulaView.phase === 5);
  const maxSize = current === 2 ? 1 : current === 3 ? 2 : 3;
  const order = pathState ? game.orders[pathState.orderIndex] : null, path = order?.path.slice(0, pathState.count) ?? [];
  nodeEls.forEach((b, mask) => {
    const node = game.nodes[mask], visible = current >= 2 && node.size <= maxSize;
    b.classList.toggle('is-visible', visible); b.style.setProperty('--reveal-delay', `${node.size * 75}ms`);
    const involved = !order || mask === 0 || path.some(e => e.to === mask);
    b.classList.toggle('is-subdued', current >= 4 && !!order && !involved);
    b.classList.toggle('is-path-node', !!order && involved && visible);
    b.classList.toggle('is-formula-source', inFormula && (allFormulaEdges ? !(mask & 1) : mask === selectedFormulaEdge.from));
    b.classList.toggle('is-formula-target', inFormula && !allFormulaEdges && mask === selectedFormulaEdge.to);
  });
  for (const e of game.edges) {
    const el = edgeEls.get(e.id), visible = current >= 2 && game.nodes[e.to].size <= maxSize;
    let active = current === 3 && e.from === 2 && e.to === 3;
    if (current >= 5 && !order) active = e.playerIndex === selectedPlayer;
    if (inFormula) active = allFormulaEdges ? e.playerIndex === 0 : e.id === selectedFormulaEdge.id;
    if (order) active = path.some(p => p.id === e.id);
    el.group.classList.toggle('is-visible', visible); el.group.classList.toggle('is-highlighted', active);
    el.group.classList.toggle('is-decisive', active && e.delta !== 0);
    el.group.classList.toggle('show-label', active && current >= 3);
    el.group.classList.toggle('show-weight', current >= 5 && !order && (!inFormula || formulaView.phase >= 4));
    el.group.classList.toggle('is-inspected', e.id === inspectedEdge);
    el.line.setAttribute('marker-end', active ? 'url(#arrow-active)' : 'url(#arrow-neutral)');
    el.line.style.strokeWidth = current >= 5 && !order && active ? 2 + e.weight * 7 : active ? 3 : 1.35;
    if (visible && active) graphSvg.append(el.group);
  }
  document.querySelectorAll('[data-show-order]').forEach(b => b.classList.toggle('is-current', !!pathState && Number(b.dataset.showOrder) === pathState.orderIndex));
}
function updateCaption() {
  const captions = ['Every voter supports the proposal.', `Removing ${game.players[removedPlayer]} leaves two votes. The outcome stays at 1.`, 'Each node’s number is v(S), the value of its coalition.', 'The highlighted edge changes v(S) from 0 to 1.', 'Every path ends at the same full group.', 'Edge labels: contribution above, fraction of orders below.', 'Click a term to see which of the six paths give it its weight.', 'Click a term to explain its weight, or inspect an edge below.', 'Four coalitions without A. Four starting nodes.', 'The bracket measures the change along the selected edge.', 'Hold the coalition fixed; count its internal orders.', 'The minus one removes A from the remaining voters.', 'The factorial ratio is the fraction of paths using this edge.', 'The formula adds the same four weighted contributions.'];
  $('#stage-bottom').innerHTML = `<span class="legend-mark"></span><span>${captions[current]}</span>`;
  const caption = $('#graph-caption');
  if (pathState) {
    const o = game.orders[pathState.orderIndex], decisive = o.path.find(e => e.delta === 1), finished = pathState.count === 3;
    const joining = o.path[Math.max(0, pathState.count - 1)].player;
    caption.innerHTML = `<span class="path-order">${o.label}</span><span>${finished ? `${decisive.player} supplies the deciding vote: +1` : `Adding ${joining}…`}</span>`;
  } else if (current === 3) caption.innerHTML = '<span class="path-order">{B} → {A, B}</span><span>A changes failure to success: +1</span>';
  else if (current === 5) caption.innerHTML = `<span class="path-order">Follow ${game.players[selectedPlayer]}</span><span>Four joining edges. Six equally weighted orders.</span>`;
  else caption.innerHTML = '';
}
function updateCalculation() {
  const player = game.players[selectedPlayer];
  const edges = game.edges.filter(e => e.playerIndex === selectedPlayer).sort((a, b) => game.nodes[a.from].size - game.nodes[b.from].size || a.from - b.from);
  $('#credit-label').textContent = `${player}’s credit`; $('#total-label').textContent = `Shapley value of ${player}`; $('#total-value').textContent = fraction(game.shares[selectedPlayer]);
  $('#terms').innerHTML = edges.map(e => `<button type="button" class="term ${e.delta ? 'is-nonzero' : ''}" data-term-edge="${e.id}" ${current >= 6 ? '' : 'disabled'} aria-haspopup="dialog" aria-controls="weight-dialog" aria-label="Explain ${player} joining ${game.nodes[e.from].label}: contribution ${e.delta}, weight ${e.count} out of 6"><span class="term-coalition">${game.nodes[e.from].label}</span><span class="term-product">${e.delta} <span>×</span> ${e.count}/${game.totalOrders}</span></button>`).join('<span class="term-plus" aria-hidden="true">+</span>');
  document.querySelectorAll('[data-term-edge]').forEach(b => b.addEventListener('click', () => explainWeight(b.dataset.termEdge, b)));
  $('#edge-choices').innerHTML = edges.map(e => `<button type="button" data-inspect-edge="${e.id}" aria-pressed="${e.id === inspectedEdge}">${game.nodes[e.from].label} → ${game.nodes[e.to].label}</button>`).join('');
  document.querySelectorAll('[data-inspect-edge]').forEach(b => b.addEventListener('click', () => inspectEdge(b.dataset.inspectEdge)));
  if (current === 7 && !inspectedEdge) $('#edge-detail').textContent = `Select one of ${player}’s joining edges to see its contribution and weight.`;
}
function updateSelectedControls() { document.querySelectorAll('[data-select-player]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.selectPlayer) === selectedPlayer))); }
function selectPlayer(i) {
  stopPlayback(); selectedPlayer = i; inspectedEdge = null; nodeEls.forEach(b => b.classList.remove('is-inspected'));
  updateSelectedControls(); updateGraph(); updateCalculation(); updateCaption();
}
function inspectEdge(id) {
  stopPlayback(); inspectedEdge = id; const e = game.edges.find(e => e.id === id);
  nodeEls.forEach((b, mask) => b.classList.toggle('is-inspected', mask === e.from || mask === e.to));
  updateGraph(); updateCalculation(); updateCaption();
  $('#edge-detail').innerHTML = `Adding <strong>${e.player}</strong>: value <strong>${game.nodes[e.from].value} → ${game.nodes[e.to].value}</strong>. Contribution <strong>${e.delta}</strong>, weight <strong>${e.count}/${game.totalOrders}</strong>. Weighted contribution: <strong>${fraction(e.delta * e.weight)}</strong>.`;
}
function inspectNode(mask) {
  stopPlayback(); inspectedEdge = null; nodeEls.forEach((b, i) => b.classList.toggle('is-inspected', i === mask));
  const node = game.nodes[mask]; updateGraph(); updateCalculation(); updateCaption();
  $('#edge-detail').innerHTML = `<strong>${node.label}</strong> contains ${node.size} ${node.size === 1 ? 'voter' : 'voters'}. ${node.value ? 'At least two supporters: the proposal passes.' : 'Fewer than two supporters: the proposal fails.'} Value: <strong>${node.value}</strong>.`;
}
function playOrder(orderIndex, animate = true) {
  stopPlayback(); inspectedEdge = null; nodeEls.forEach(b => b.classList.remove('is-inspected'));
  pathState = { orderIndex, count: animate && !reduced.matches ? 0 : 3 }; const replay = $('#replay');
  if (current === 7) { replay.disabled = pathState.count < 3; replay.textContent = pathState.count < 3 ? 'Playing…' : 'Replay'; }
  const render = () => { updateGraph(); updateCaption(); }; updateCalculation(); render();
  if (pathState.count < 3) {
    const advance = () => {
      if (!pathState) return;
      pathState.count++; render();
      if (pathState.count < 3) playbackTimer = setTimeout(advance, 950);
      else if (current === 7) { replay.disabled = false; replay.textContent = 'Replay'; }
    };
    playbackTimer = setTimeout(advance, 300);
  }
}
document.querySelectorAll('[data-remove-player]').forEach(b => b.addEventListener('click', () => { removedPlayer = Number(b.dataset.removePlayer); updateRemoval(); updateCaption(); }));
document.querySelectorAll('[data-select-player]').forEach(b => b.addEventListener('click', () => selectPlayer(Number(b.dataset.selectPlayer))));
document.querySelectorAll('[data-compare-order]').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('[data-compare-order]').forEach(other => other.setAttribute('aria-pressed', String(other === b)));
  playOrder(Number(b.dataset.compareOrder));
}));
document.querySelectorAll('[data-show-order]').forEach(b => b.addEventListener('click', () => playOrder(Number(b.dataset.showOrder))));
$('#replay').addEventListener('click', () => playOrder(Number($('#order-select').value)));
$('#order-select').addEventListener('change', () => playOrder(Number($('#order-select').value), false));
function updateScroll() {
  scrollFrame = 0; const header = mobile.matches ? 54 : 68;
  const anchor = mobile.matches ? Math.min(innerHeight - 80, header + stage.getBoundingClientRect().height + 80) : innerHeight * .53;
  let index = 0; for (let i = 0; i < steps.length; i++) if (steps[i].getBoundingClientRect().top <= anchor) index = i;
  showStep(index);
  // Scrolling through this passage visits all six accounting paths, then
  // gathers them into the four weighted joining edges. Buttons can replay any.
  if (current === 5) {
    const rect = steps[5].getBoundingClientRect();
    const phase = Math.min(6, Math.floor(Math.max(0, anchor - rect.top) / rect.height * 8));
    const next = phase < 6 ? phase : null;
    if (next !== (pathState?.orderIndex ?? null)) {
      stopPlayback();
      pathState = next === null ? null : { orderIndex: next, count: 3 };
      updateGraph(); updateCaption();
    }
  }
  const start = $('.story').offsetTop, end = steps.at(-1).offsetTop;
  const percentage = Math.max(0, Math.min(100, (scrollY - start) / Math.max(1, end - start) * 100));
  $('#progress-fill').style.width = `${percentage}%`;
}
window.addEventListener('scroll', () => { if (!scrollFrame) scrollFrame = requestAnimationFrame(updateScroll); }, { passive: true });
window.addEventListener('resize', updateScroll);
reduced.addEventListener('change', () => { stopPlayback(); updateGraph(); updateCaption(); });
showStep(0); updateScroll();

function goToStep(index) {
  showStep(index);
  requestAnimationFrame(() => {
    const header = mobile.matches ? 54 : 68;
    const offset = mobile.matches ? header + stage.getBoundingClientRect().height + 15 : header + 30;
    const top = mobile.matches ? scrollY + steps[index].querySelector('.step-content').getBoundingClientRect().top : steps[index].offsetTop;
    window.scrollTo({ top: top - offset, behavior: 'instant' });
    updateScroll();
  });
}
function stepForHash(hash) {
  return hash === '#beginning' ? 0 : steps.findIndex(step => `#${step.id}` === hash);
}
document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', event => {
  const hash = link.getAttribute('href'), index = stepForHash(hash);
  if (index < 0) return;
  event.preventDefault(); history.replaceState(null, '', hash); goToStep(index);
}));
function followHash() {
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
