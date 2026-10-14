import { prepareElectoralMap, rectangleBoundary } from './electoral-map-geometry.mjs';

const svgNS = 'http://www.w3.org/2000/svg';
const phases = ['geographic', 'electoral', 'adjusted'];
const callouts = ['MA', 'RI', 'CT', 'NJ', 'DE', 'DC'];
let viewCount = 0;

function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text != null) node.textContent = text;
  return node;
}

const copyRings = rings => rings.map(ring => ring.map(point => [...point]));
const interpolate = (a, b, t) => a + (b - a) * t;
const ease = t => t < .5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
const adjustedNumber = value => value.toFixed(2);
const adjustment = state => (state.adjustedVotes / state.votes - 1) * 100;
const adjustmentLabel = state => `${adjustment(state) >= 0 ? '+' : '−'}${Math.abs(adjustment(state)).toFixed(2)}%`;

function pathData(rings) {
  return rings.map(ring => `M${ring.map(point => `${point[0].toFixed(2)},${point[1].toFixed(2)}`).join('L')}Z`).join('');
}
const cellPath = rectangles => rectangles.filter(r => r[2] > 1e-7 && r[3] > 1e-7).map(([x,y,w,h]) => `M${x.toFixed(3)},${y.toFixed(3)}h${w.toFixed(3)}v${h.toFixed(3)}h${(-w).toFixed(3)}Z`).join('');
const boundaryPath = rectangles => rectangleBoundary(rectangles).map(([a,b]) => `M${a[0].toFixed(3)},${a[1].toFixed(3)}L${b[0].toFixed(3)},${b[1].toFixed(3)}`).join('');

/** A political map becomes unit electoral squares, then fractional Shapley squares. */
export function createElectoralMapView(results, { onPhaseChange = () => {} } = {}) {
  const ELECTORAL_MAP = prepareElectoralMap(results);
  const id = `electoral-map-${++viewCount}`;
  const total = results.reduce((sum, state) => sum + state.votes, 0);
  const resultsByCode = new Map(results.map(state => [state.abbreviation, state]));
  const states = ELECTORAL_MAP.states.map(state => {
    const result = resultsByCode.get(state.abbreviation);
    return { ...state, adjustedVotes: result ? total * result.share : state.adjustedVotes };
  });
  const byCode = new Map(states.map(state => [state.abbreviation, state]));
  const alphabetic = [...states].sort((a, b) => a.name.localeCompare(b.name));
  const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
  let selected = byCode.get('CA') || states[0];
  let currentPhase = -1;
  let animation = 0;
  let compact = false;
  let drawnPhase = 0;
  let currentRings = states.map(state => copyRings(state.geographic));
  let currentLabels = states.map(state => geographicLabel(state));
  let currentLeaders = states.map(state => [...state.labels.geographic]);
  let currentCells = states.map(state => state.cells.map(cell => [...cell.electoral]));
  let drawCells = false;

  const element = document.createElement('section');
  element.className = 'cartogram-view';
  element.setAttribute('aria-label', 'Electoral College maps');
  element.innerHTML = `
    <nav class="cartogram-phases" aria-label="Choose a map stage">
      <button type="button" data-phase="0">1 Geography</button>
      <button type="button" data-phase="1">2 Electoral votes</button>
      <button type="button" data-phase="2">3 Shapley votes</button>
    </nav>
    <div class="cartogram-key"><span class="cartogram-measure"></span><span class="cartogram-total">${total} electoral votes</span></div>
    <div class="cartogram-canvas">
      <svg class="cartogram-map" viewBox="0 0 ${ELECTORAL_MAP.width} ${ELECTORAL_MAP.height}" role="group" aria-roledescription="interactive map" aria-labelledby="${id}-title ${id}-description">
        <title id="${id}-title"></title><desc id="${id}-description"></desc>
      </svg>
    </div>
    <div class="cartogram-controls">
      <div class="cartogram-state-control"><label for="${id}-state">Inspect a state or DC</label><select id="${id}-state"></select></div>
      <div class="cartogram-actions"><button class="cartogram-replay" type="button">Replay morph</button><label class="cartogram-outline-control" hidden><input type="checkbox"> <span>Outline electoral-vote areas</span></label></div>
    </div>
    <dl class="cartogram-readout" aria-live="polite" aria-atomic="true"></dl>`;
  const map = element.querySelector('.cartogram-map');
  const title = map.querySelector('title');
  const description = map.querySelector('desc');
  const measure = element.querySelector('.cartogram-measure');
  const totalLabel = element.querySelector('.cartogram-total');
  const select = element.querySelector('select');
  const replay = element.querySelector('.cartogram-replay');
  const outlineControl = element.querySelector('.cartogram-outline-control');
  const outlineCheckbox = outlineControl.querySelector('input');
  replay.setAttribute('aria-label', 'Replay map morph');
  outlineCheckbox.setAttribute('aria-label', 'Outline electoral-vote areas');
  const readout = element.querySelector('.cartogram-readout');
  const phaseButtons = [...element.querySelectorAll('.cartogram-phases button')];
  phaseButtons.forEach(button => button.addEventListener('click', () => onPhaseChange(Number(button.dataset.phase))));
  const fills = svg('g', { class: 'cartogram-states' });
  const grids = svg('g', { class: 'cartogram-cell-grid', 'aria-hidden': 'true' });
  const borders = svg('g', { class: 'cartogram-borders', 'aria-hidden': 'true' });
  const outlines = svg('g', { class: 'cartogram-baseline', 'aria-hidden': 'true' });
  const leaders = svg('g', { class: 'cartogram-leaders', 'aria-hidden': 'true' });
  const labels = svg('g', { class: 'cartogram-labels', 'aria-hidden': 'true' });
  map.append(fills, grids, borders, outlines, leaders, labels);
  const marks = states.map(state => {
    const path = svg('path', { class: 'cartogram-state', 'data-state': state.abbreviation, 'fill-rule': 'evenodd', role: 'button', tabindex: '-1' });
    const accessible = `${state.name}, ${state.votes} electoral votes.`;
    path.setAttribute('aria-label', accessible);
    path.append(svg('title', {}, accessible));
    path.addEventListener('click', () => choose(state));
    path.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        choose(state);
      } else if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const index = alphabetic.indexOf(state);
        const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? alphabetic.length - 1 : (index + (['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1) + alphabetic.length) % alphabetic.length;
        choose(alphabetic[nextIndex]);
        marks.find(mark => mark.state === selected).path.focus();
      }
    });
    const baseline = svg('path', { d: pathData(state.electoral), 'fill-rule': 'evenodd' });
    const grid = svg('path');
    const border = svg('path', { class: 'cartogram-border' });
    const leader = svg('line', { class: 'cartogram-label-leader' });
    const label = svg('text', { class: 'cartogram-label', 'text-anchor': 'middle' });
    label.classList.toggle('is-major', state.votes >= 10);
    leader.classList.toggle('is-major', state.votes >= 10);
    const abbreviation = svg('tspan', { class: 'cartogram-abbreviation', x: 0 }, state.abbreviation);
    const value = svg('tspan', { class: 'cartogram-label-value', x: 0, dy: '1.15em' }, String(state.votes));
    label.append(abbreviation, value);
    label.addEventListener('click', () => choose(state));
    fills.append(path);
    grids.append(grid);
    borders.append(border);
    outlines.append(baseline);
    leaders.append(leader);
    labels.append(label);
    return { state, path, grid, border, baseline, leader, label, abbreviation, value };
  });

  for (const state of alphabetic) {
    const option = document.createElement('option');
    option.value = state.abbreviation;
    option.textContent = state.name;
    select.append(option);
  }
  select.addEventListener('change', () => choose(byCode.get(select.value)));
  replay.addEventListener('click', () => {
    if (currentPhase < 1) return;
    stopAnimation();
    settle(currentPhase - 1);
    morph(currentPhase, 550);
  });
  outlineCheckbox.addEventListener('change', () => element.classList.toggle('has-baseline', outlineCheckbox.checked));

  function geographicLabel(state) {
    const callout = callouts.indexOf(state.abbreviation);
    return callout < 0 ? [...state.labels.geographic] : [ELECTORAL_MAP.width - 29, ELECTORAL_MAP.height * (.22 + callout * .07)];
  }

  function labelsFor(phase) {
    return states.map(state => phase === 0 ? geographicLabel(state) : [...state.labels[phases[phase]]]);
  }

  function stopAnimation() {
    if (animation) cancelAnimationFrame(animation);
    animation = 0;
    element.classList.remove('is-morphing');
  }

  function paint() {
    marks.forEach((mark, index) => {
      const at = currentLabels[index];
      const from = currentLeaders[index];
      const outline = drawCells ? '' : pathData(currentRings[index]);
      mark.path.setAttribute('d', drawCells ? cellPath(currentCells[index]) : outline);
      mark.border.setAttribute('d', drawCells ? boundaryPath(currentCells[index]) : outline);
      mark.grid.setAttribute('d', drawCells ? cellPath(currentCells[index]) : '');
      mark.label.setAttribute('transform', `translate(${at[0].toFixed(2)} ${at[1].toFixed(2)})`);
      mark.leader.setAttribute('x1', from[0]);
      mark.leader.setAttribute('y1', from[1]);
      mark.leader.setAttribute('x2', at[0]);
      mark.leader.setAttribute('y2', at[1]);
      const leaderLength = Math.hypot(at[0] - from[0], at[1] - from[1]);
      mark.leader.style.opacity = leaderLength > 5 ? '1' : '0';
      const value = interpolate(mark.state.votes, mark.state.adjustedVotes, Math.max(0, drawnPhase - 1));
      mark.value.textContent = drawnPhase > 1.001 ? value.toFixed(1) : String(mark.state.votes);
    });
  }

  function settle(phase) {
    currentRings = states.map(state => copyRings(state[phases[phase]]));
    currentLabels = labelsFor(phase);
    currentLeaders = states.map(state => [...state.labels[phases[phase]]]);
    drawnPhase = phase;
    drawCells = phase >= 1;
    currentCells = states.map(state => state.cells.map(cell => [...cell[phase === 2 ? 'adjusted' : 'electoral']]));
    paint();
  }

  function morph(phase, dwell = phase === 2 ? 400 : 0) {
    stopAnimation();
    if (motionPreference.matches) {
      settle(phase);
      return;
    }
    const fromRings = currentRings.map(copyRings);
    const fromLabels = currentLabels.map(point => [...point]);
    const fromLeaders = currentLeaders.map(point => [...point]);
    const fromPhase = drawnPhase;
    const fromCells = currentCells.map(rectangles => rectangles.map(rectangle => [...rectangle]));
    const targetCells = states.map(state => state.cells.map(cell => cell[phase === 2 ? 'adjusted' : 'electoral']));
    drawCells = fromPhase >= 1 && phase >= 1;
    const targetLabels = labelsFor(phase);
    const targetLeaders = states.map(state => state.labels[phases[phase]]);
    let start;
    element.classList.add('is-morphing');
    const frame = timestamp => {
      if (start == null) start = timestamp;
      const progress = Math.max(0, Math.min(1, (timestamp - start - dwell) / (phase === 2 ? 2000 : 1500)));
      const t = ease(progress);
      currentRings = states.map((state, index) => state[phases[phase]].map((ring, ringIndex) => ring.map((point, pointIndex) => point.map((coordinate, dimension) => interpolate(fromRings[index][ringIndex][pointIndex][dimension], coordinate, t)))));
      currentLabels = targetLabels.map((point, index) => point.map((coordinate, dimension) => interpolate(fromLabels[index][dimension], coordinate, t)));
      currentLeaders = targetLeaders.map((point, index) => point.map((coordinate, dimension) => interpolate(fromLeaders[index][dimension], coordinate, t)));
      currentCells = targetCells.map((rectangles, index) => rectangles.map((rectangle, cellIndex) => rectangle.map((coordinate, dimension) => interpolate(fromCells[index][cellIndex][dimension], coordinate, t))));
      drawnPhase = interpolate(fromPhase, phase, t);
      paint();
      if (progress < 1) animation = requestAnimationFrame(frame);
      else { animation = 0; element.classList.remove('is-morphing'); settle(phase); }
    };
    animation = requestAnimationFrame(frame);
  }

  function choose(state) {
    selected = state;
    select.value = state.abbreviation;
    for (const mark of marks) {
      const active = mark.state === state;
      mark.path.classList.toggle('is-selected', active);
      mark.border.classList.toggle('is-selected', active);
      mark.baseline.classList.toggle('is-selected', active);
      mark.label.classList.toggle('is-selected', active);
      mark.leader.classList.toggle('is-selected', active);
      mark.path.setAttribute('aria-pressed', String(active));
      mark.path.setAttribute('tabindex', active ? '0' : '-1');
    }
    // Raising a selected path keeps its outline visible along shared borders.
    fills.append(marks.find(mark => mark.state === state).path);
    borders.append(marks.find(mark => mark.state === state).border);
    readout.replaceChildren();
    const values = currentPhase === 2 ? [
      [`${state.name} · electoral → Shapley`, `${state.votes} → ${adjustedNumber(state.adjustedVotes)}`],
      ['Area adjustment', adjustmentLabel(state)]
    ] : [[state.name, `${state.votes} electoral votes`]];
    for (const [term, definition] of values) {
      const pair = document.createElement('div');
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = term;
      dd.textContent = definition;
      pair.append(dt, dd);
      readout.append(pair);
    }
    readout.setAttribute('aria-label', currentPhase === 2 ? `${state.name}; adjusted votes are ${total} times the Shapley value` : `${state.name}, ${state.votes} electoral votes`);
  }

  function resize() {
    const bounds = map.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const scale = Math.min(bounds.width / ELECTORAL_MAP.width, bounds.height / ELECTORAL_MAP.height);
    element.classList.toggle('is-tiny', scale * ELECTORAL_MAP.width < 260);
    compact = bounds.width < 600;
    element.classList.toggle('is-compact', compact);
    // Labels keep their map positions; only their type follows display size.
    map.style.setProperty('--cartogram-label-size', `${(compact ? 7.5 : 11.5) / scale}px`);
  }

  function update(phase, { animate = true } = {}) {
    if (!Number.isInteger(phase) || phase < 0 || phase > 2) throw new Error('Use Electoral College map phases 0–2.');
    if (phase === currentPhase) { resize(); return; }
    const initial = currentPhase < 0;
    currentPhase = phase;
    element.dataset.phase = String(phase);
    phaseButtons.forEach((button, i) => {
      button.classList.toggle('is-active', i === phase);
      if (i === phase) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    replay.disabled = phase === 0;
    replay.textContent = phase === 2 ? 'Replay adjustment' : 'Replay morph';
    replay.setAttribute('aria-label', phase === 2 ? 'Replay electoral votes to Shapley-adjusted votes' : 'Replay geography to electoral votes');
    outlineControl.hidden = phase !== 2;
    const titleText = ['States and their electoral votes', 'State areas proportional to electoral votes', 'State areas proportional to Shapley-adjusted electoral votes'][phase];
    title.textContent = titleText;
    measure.textContent = ['Geographic map', '1 square = 1 electoral vote', 'Area ∝ Shapley-adjusted votes'][phase];
    totalLabel.textContent = phase === 2 ? `${total} adjusted votes in total` : `${total} electoral votes in total`;
    for (const mark of marks) {
      const state = mark.state;
      const accessible = phase === 2 ? `${state.name}, ${state.votes} electoral votes; ${adjustedNumber(state.adjustedVotes)} adjusted votes; ${adjustmentLabel(state)} adjustment.` : `${state.name}, ${state.votes} electoral votes.`;
      mark.path.setAttribute('aria-label', accessible);
      mark.path.querySelector('title').textContent = accessible;
    }
    choose(selected);
    description.textContent = `${titleText}. The 50 states and DC are treated as 51 winner-take-all blocs, using the 2024 and 2028 electoral allocation. ${phase === 0 ? 'This geographic map labels electoral votes; Alaska and Hawaii are shown as insets.' : phase === 1 ? 'Each square represents one electoral vote. There are 538 complete squares.' : 'Each state’s filled area is proportional to 538 times its Shapley pivotal probability. Whole squares retain the same area per vote; partial squares show fractional adjusted votes. Both cartograms have the same total filled area and coordinate scale. The adjustment is shown at its actual size. A dashed outline shows the selected state’s electoral-vote area. The checkbox shows all electoral-vote outlines.'} Select a state or DC below for electoral votes, adjusted votes, and percentage adjustment. The map also supports arrow keys after focusing a state.`;
    // A direct visit to the final stage still demonstrates the small adjustment
    // from electoral squares, rather than blending geography straight into it.
    if (animate && ((initial && phase > 0) || (phase === 2 && drawnPhase < 1))) settle(phase - 1);
    if (animate && (!initial || phase > 0)) morph(phase);
    else { stopAnimation(); settle(phase); }
    resize();
  }

  function pause() {
    stopAnimation();
    if (currentPhase >= 0) settle(currentPhase);
  }

  motionPreference.addEventListener('change', () => {
    if (motionPreference.matches && currentPhase >= 0) { stopAnimation(); settle(currentPhase); }
  });
  const observer = new ResizeObserver(resize);
  observer.observe(element);
  choose(selected);
  paint();
  return { element, update, resize, pause };
}
