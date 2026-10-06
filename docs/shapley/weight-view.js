import { matchingOrders } from './game.mjs';

// All miniatures share the same geometry so an edge stays in the same place.
const points = { 0: [160, 20], 1: [58, 83], 2: [160, 83], 4: [262, 83], 3: [58, 147], 5: [160, 147], 6: [262, 147], 7: [160, 211] };
const svgNS = 'http://www.w3.org/2000/svg';
function mark(tag, attributes = {}, text) {
  const element = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (text != null) element.textContent = text;
  return element;
}
function miniature(game, order, edge, index, matches) {
  const figure = document.createElement('figure');
  figure.className = `order-miniature${matches ? ' is-match' : ''}`;
  figure.dataset.matches = String(matches);
  figure.dataset.order = order.label;
  const heading = document.createElement('figcaption');
  heading.className = 'miniature-heading';
  const label = document.createElement('span'); label.className = 'miniature-order'; label.textContent = order.label;
  const status = document.createElement('span'); status.className = 'miniature-status'; status.textContent = matches ? '● Uses this edge' : 'Different joining edge';
  heading.append(label, status); figure.append(heading);
  const drawing = mark('svg', { viewBox: '0 0 320 238', class: 'miniature-svg', role: 'img', 'aria-labelledby': `miniature-title-${index} miniature-description-${index}` });
  drawing.append(mark('title', { id: `miniature-title-${index}` }, `${order.label}: ${matches ? 'uses' : 'does not use'} the selected edge`));
  const joining = order.path.find(e => e.playerIndex === edge.playerIndex);
  drawing.append(mark('desc', { id: `miniature-description-${index}` }, `${edge.player} joins ${game.nodes[joining.from].label}, changing value ${game.nodes[joining.from].value} to ${game.nodes[joining.to].value}. Each arrow adds one voter; each node shows its coalition and value.`));
  const definitions = mark('defs');
  for (const kind of ['background', 'path', 'target']) {
    const marker = mark('marker', { id: `mini-arrow-${index}-${kind}`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 4, markerHeight: 4, orient: 'auto' });
    marker.append(mark('path', { d: 'M0 0 L10 5 L0 10z', class: `mini-arrow-${kind}` })); definitions.append(marker);
  }
  drawing.append(definitions);
  const paths = new Set(order.path.map(e => e.id));
  const inPath = new Set([0, ...order.path.map(e => e.to)]);
  const layers = ['background', 'path', 'target'].map(() => mark('g'));
  for (const candidate of game.edges) {
    const [x1, y1] = points[candidate.from], [x2, y2] = points[candidate.to];
    const ratio = 17 / (y2 - y1);
    const startX = x1 + (x2 - x1) * ratio, endX = x2 - (x2 - x1) * ratio;
    const kind = matches && candidate.id === edge.id ? 'target' : paths.has(candidate.id) ? 'path' : 'background';
    const layer = kind === 'target' ? 2 : kind === 'path' ? 1 : 0;
    layers[layer].append(mark('path', { d: `M${startX},${y1 + 17} L${endX},${y2 - 17}`, class: `mini-edge mini-edge-${kind}`, 'marker-end': `url(#mini-arrow-${index}-${kind})`, 'data-mini-edge': candidate.id }));
  }
  drawing.append(...layers);
  for (const node of game.nodes) {
    const [x, y] = points[node.mask];
    const width = node.size === 3 ? 120 : node.size === 2 ? 96 : 80;
    const group = mark('g', { class: `mini-node${inPath.has(node.mask) ? ' is-on-path' : ''}`, transform: `translate(${x},${y})` });
    group.append(mark('rect', { x: -width / 2, y: -14, width, height: 28, rx: 3 }));
    group.append(mark('text', { 'text-anchor': 'middle', 'dominant-baseline': 'central' }, `${node.label.replaceAll(', ', ',')} · ${node.value}`));
    drawing.append(group);
  }
  figure.append(drawing);
  const detail = document.createElement('p'); detail.className = 'miniature-detail';
  detail.textContent = `${edge.player} joins ${game.nodes[joining.from].label} · contribution ${joining.delta > 0 ? '+' : ''}${joining.delta}`;
  figure.append(detail);
  return figure;
}

export function createWeightView(game) {
  const dialog = document.createElement('dialog');
  dialog.className = 'weight-dialog'; dialog.id = 'weight-dialog';
  dialog.setAttribute('aria-labelledby', 'weight-view-title');
  dialog.setAttribute('aria-describedby', 'weight-view-caption');
  dialog.innerHTML = `<div class="weight-dialog-top"><span class="eyebrow">COUNTING PATHS</span><button type="button" class="weight-close" aria-label="Close weight explanation">Close</button></div><h2 id="weight-view-title"></h2><p class="weight-view-caption" id="weight-view-caption" aria-live="polite"></p><div class="weight-edge-selector" role="group" aria-label="Choose the joining edge"><span id="weight-player-label"></span><div id="weight-edge-buttons"></div></div><div class="weight-view-key"><span><i class="matching-swatch" aria-hidden="true"></i>Paths that use this edge</span><span><i class="other-swatch" aria-hidden="true"></i>Other paths</span><span>A thicker arrow marks the selected edge.</span></div><div class="weight-grid" id="weight-grid"></div><p class="weight-view-note">The weight counts how often this particular edge occurs. The contribution measures how much the outcome changes along it.</p>`;
  document.body.append(dialog);
  let trigger = null;
  function render(edgeId) {
    const edge = game.edges.find(e => e.id === edgeId);
    if (!edge) throw new Error('Unknown joining edge.');
    const matches = matchingOrders(game, edgeId);
    dialog.dataset.edge = edgeId;
    dialog.dataset.matchCount = matches.length;
    document.querySelector('#weight-view-title').textContent = `Where ${edge.count}/${game.totalOrders} comes from`;
    const context = edge.from === 0 ? 'the empty coalition' : game.nodes[edge.from].label;
    document.querySelector('#weight-view-caption').textContent = `${edge.player} joins ${context} in ${matches.length} of the ${game.totalOrders} orders. Contribution: ${game.nodes[edge.to].value} − ${game.nodes[edge.from].value} = ${edge.delta}. Weight: ${edge.count}/${game.totalOrders}.`;
    document.querySelector('#weight-player-label').textContent = `Before ${edge.player} joins`;
    const buttons = document.querySelector('#weight-edge-buttons'); buttons.replaceChildren();
    for (const candidate of game.edges.filter(e => e.playerIndex === edge.playerIndex).sort((a, b) => game.nodes[a.from].size - game.nodes[b.from].size || a.from - b.from)) {
      const button = document.createElement('button'); button.type = 'button'; button.dataset.weightEdge = candidate.id;
      button.setAttribute('aria-pressed', String(candidate.id === edgeId));
      button.textContent = `${game.nodes[candidate.from].label} · ${candidate.count}/${game.totalOrders}`;
      button.addEventListener('click', () => {
        render(candidate.id);
        document.querySelector(`[data-weight-edge="${candidate.id}"]`).focus({ preventScroll: true });
      });
      buttons.append(button);
    }
    document.querySelector('#weight-grid').replaceChildren(...game.orders.map((order, index) => miniature(game, order, edge, index, order.path.some(e => e.id === edgeId))));
  }
  function close() { dialog.close(); }
  dialog.querySelector('.weight-close').addEventListener('click', close);
  dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) close(); } });
  dialog.addEventListener('close', () => { document.body.classList.remove('weight-view-open'); trigger?.focus({ preventScroll: true }); });
  return {
    open(edgeId, source = document.activeElement) {
      trigger = source;
      render(edgeId);
      document.body.classList.add('weight-view-open');
      dialog.showModal();
      dialog.querySelector('.weight-close').focus({ preventScroll: true });
    }
  };
}
