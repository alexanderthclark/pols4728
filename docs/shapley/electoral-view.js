const svgNS = 'http://www.w3.org/2000/svg';
const beforeCalifornia = ['TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ'];
let viewCount = 0;

function svg(tag, attributes = {}, text) {
  const element = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (text != null) element.textContent = text;
  return element;
}

const percent = value => `${(value * 100).toFixed(2)}%`;

/** Two views of the same 51-player weighted-voting game, without a coalition diagram. */
export function createElectoralView(results) {
  const id = `electoral-${++viewCount}`;
  const byCode = new Map(results.map(state => [state.abbreviation, state]));
  const california = byCode.get('CA');
  const before = beforeCalifornia.map(code => byCode.get(code));
  const after = results.filter(state => !beforeCalifornia.includes(state.abbreviation) && state !== california);
  const totalVotes = results.reduce((sum, state) => sum + state.votes, 0);
  const beforeVotes = before.reduce((sum, state) => sum + state.votes, 0);
  const afterCalifornia = beforeVotes + california.votes;
  const quota = Math.floor(totalVotes / 2) + 1;
  let selected = california;
  let currentPhase = -1;

  const element = document.createElement('section');
  element.className = 'electoral-view';
  element.setAttribute('aria-label', 'Electoral College pivotal power');
  element.innerHTML = `
    <div class="electoral-scene electoral-order-scene">
      <h3 class="electoral-heading">One imagined accounting order</h3>
      <p class="electoral-model">${results.length} players · ${totalVotes} electoral votes · ${quota} to win</p>
      <svg class="electoral-tally" role="img" aria-labelledby="${id}-order-title ${id}-order-description"></svg>
      <p class="electoral-change" aria-label="Coalition value changes from zero to one when California joins">
        <span>Before CA: <span class="electoral-payoff">0</span></span>
        <span aria-hidden="true">→</span>
        <span>With CA: <span class="electoral-payoff electoral-winning">1</span></span>
      </p>
      <p class="electoral-order-note">Before CA: ${beforeCalifornia.join(', ')}. The remaining ${after.length} players follow CA.</p>
    </div>
    <div class="electoral-scene electoral-power-scene" hidden>
      <svg class="electoral-scatter" role="img" aria-labelledby="${id}-power-title ${id}-power-description"></svg>
      <div class="electoral-state-control">
        <label for="${id}-state">Inspect a state or DC</label>
        <select id="${id}-state"></select>
      </div>
      <dl class="electoral-readout" aria-live="polite" aria-atomic="true"></dl>
      <p class="electoral-selection-note"></p>
    </div>`;

  const orderScene = element.querySelector('.electoral-order-scene');
  const powerScene = element.querySelector('.electoral-power-scene');
  const tally = element.querySelector('.electoral-tally');
  const scatter = element.querySelector('.electoral-scatter');
  const select = element.querySelector('select');
  const readout = element.querySelector('.electoral-readout');
  const selectionNote = element.querySelector('.electoral-selection-note');

  for (const state of [...results].sort((a, b) => a.name.localeCompare(b.name))) {
    const option = document.createElement('option');
    option.value = state.abbreviation;
    option.textContent = `${state.name} (${state.votes} votes)`;
    select.append(option);
  }
  select.value = selected.abbreviation;

  function drawTally(width) {
    const height = width < 400 ? 140 : 164;
    const left = 12, right = width - 12;
    const x = votes => left + votes / totalVotes * (right - left);
    const barY = width < 400 ? 55 : 63;
    const barHeight = width < 400 ? 28 : 34;
    const labelY = width < 400 ? 22 : 26;
    const tickY = barY + barHeight + 22;
    const quotaY = height - 6;
    tally.setAttribute('viewBox', `0 0 ${width} ${height}`);
    tally.replaceChildren(
      svg('title', { id: `${id}-order-title` }, 'California crosses the 270-vote threshold in one illustrative order'),
      svg('desc', { id: `${id}-order-description` }, `This is an imagined accounting order, not a forecast or a chronology of voting. Before California: ${before.map(state => `${state.name}, ${state.votes} votes`).join('; ')}. Those ${before.length} players total ${beforeVotes} votes, below the ${quota}-vote threshold, so the coalition value is zero. California adds ${california.votes} votes, bringing the total to ${afterCalifornia}, so the value becomes one. California is pivotal and its marginal contribution is one minus zero, or one. After California: ${after.map(state => state.name).join(', ')}, with ${totalVotes - afterCalifornia} votes in total. The bar measures cumulative electoral votes, from zero to ${totalVotes}.`)
    );
    const labels = [
      ['Before CA', `${beforeVotes} votes`, (x(0) + x(beforeVotes)) / 2],
      ['CA', `+${california.votes}`, (x(beforeVotes) + x(afterCalifornia)) / 2],
      ['After CA', `${totalVotes - afterCalifornia} votes`, (x(afterCalifornia) + x(totalVotes)) / 2]
    ];
    for (const [label, votes, at] of labels) {
      tally.append(svg('text', { class: 'electoral-tally-label', x: at, y: labelY, 'text-anchor': 'middle' }, label));
      tally.append(svg('text', { class: label === 'CA' ? 'electoral-tally-votes electoral-ca-text' : 'electoral-tally-votes', x: at, y: labelY + 18, 'text-anchor': 'middle' }, votes));
    }
    tally.append(
      svg('rect', { class: 'electoral-before-bar', x: left, y: barY, width: x(beforeVotes) - left, height: barHeight }),
      svg('rect', { class: 'electoral-ca-bar', x: x(beforeVotes), y: barY, width: x(afterCalifornia) - x(beforeVotes), height: barHeight }),
      svg('rect', { class: 'electoral-after-bar', x: x(afterCalifornia), y: barY, width: right - x(afterCalifornia), height: barHeight }),
      svg('line', { class: 'electoral-quota-line', x1: x(quota), x2: x(quota), y1: barY - 6, y2: quotaY - 15 })
    );
    for (const votes of [0, beforeVotes, afterCalifornia, totalVotes]) {
      tally.append(svg('line', { class: 'electoral-tick', x1: x(votes), x2: x(votes), y1: barY + barHeight, y2: barY + barHeight + 5 }));
      tally.append(svg('text', { class: 'electoral-tick-label', x: x(votes), y: tickY, 'text-anchor': votes === 0 ? 'start' : votes === totalVotes ? 'end' : 'middle' }, String(votes)));
    }
    tally.append(svg('text', { class: 'electoral-quota-label', x: x(quota), y: quotaY, 'text-anchor': 'middle' }, `${quota} to win`));
  }

  function drawScatter(width) {
    const compact = width < 400;
    const desiredHeight = compact ? 160 : Math.min(308, Math.round(width * .53));
    const otherHeight = [...powerScene.children].filter(child => child !== scatter).reduce((sum, child) => {
      const style = getComputedStyle(child);
      return sum + child.getBoundingClientRect().height + parseFloat(style.marginTop) + parseFloat(style.marginBottom);
    }, 0);
    const availableHeight = element.parentElement?.clientHeight;
    const height = availableHeight ? Math.max(150, Math.min(desiredHeight, Math.floor(availableHeight - otherHeight - 2))) : desiredHeight;
    const bounds = { left: 44, right: width - 18, top: 30, bottom: height - 38 };
    const x = votes => bounds.left + votes / 60 * (bounds.right - bounds.left);
    const y = power => bounds.bottom - (power - .95) / .14 * (bounds.bottom - bounds.top);
    scatter.setAttribute('viewBox', `0 0 ${width} ${height}`);
    scatter.replaceChildren(
      svg('title', { id: `${id}-power-title` }, 'Shapley pivotal power relative to electoral vote share'),
      svg('desc', { id: `${id}-power-description` }, `The horizontal axis runs from zero to 60 electoral votes. The vertical axis runs from 0.95 to 1.09 and shows each player's Shapley share divided by its electoral vote share. The dashed line at one means power proportional to electoral votes. All ${results.length} players are shown; players with equal vote counts coincide. Larger blocs have greater relative pivotal power in this 270-of-538 voting game. California has relative power ${california.relativePower.toFixed(4)}. The seven three-vote players have relative power ${byCode.get('DC').relativePower.toFixed(4)}. A rust-colored hollow ring highlights ${selected.name}: ${selected.votes} electoral votes, ${percent(selected.voteShare)} of electoral votes, ${percent(selected.share)} of Shapley credit, and relative power ${selected.relativePower.toFixed(4)}. Use the state selector below for every player's numerical values.`),
      svg('text', { class: 'electoral-axis-caption', x: bounds.left, y: 14 }, 'Relative power'),
      svg('text', { class: 'electoral-reference-caption', x: bounds.right, y: 14, 'text-anchor': 'end' }, '1 = proportional'),
      svg('line', { class: 'electoral-axis', x1: bounds.left, x2: bounds.left, y1: bounds.top, y2: bounds.bottom }),
      svg('line', { class: 'electoral-axis', x1: bounds.left, x2: bounds.right, y1: bounds.bottom, y2: bounds.bottom }),
      svg('line', { class: 'electoral-proportional-line', x1: bounds.left, x2: bounds.right, y1: y(1), y2: y(1) })
    );
    for (const votes of [0, 10, 20, 30, 40, 50, 60]) {
      scatter.append(svg('line', { class: 'electoral-tick', x1: x(votes), x2: x(votes), y1: bounds.bottom, y2: bounds.bottom + 4 }));
      scatter.append(svg('text', { class: 'electoral-tick-label', x: x(votes), y: bounds.bottom + 18, 'text-anchor': 'middle' }, String(votes)));
    }
    for (const power of [.96, 1, 1.04, 1.08]) {
      scatter.append(svg('line', { class: 'electoral-tick', x1: bounds.left - 4, x2: bounds.left, y1: y(power), y2: y(power) }));
      scatter.append(svg('text', { class: 'electoral-tick-label', x: bounds.left - 8, y: y(power) + 4, 'text-anchor': 'end' }, power.toFixed(2)));
    }
    scatter.append(svg('text', { class: 'electoral-axis-caption', x: (bounds.left + bounds.right) / 2, y: height - 2, 'text-anchor': 'middle' }, 'Electoral votes'));
    const marks = svg('g', { 'aria-hidden': 'true' });
    for (const state of results) {
      const point = svg('circle', { class: 'electoral-state-point', 'data-state': state.abbreviation, cx: x(state.votes), cy: y(state.relativePower), r: compact ? 3.4 : 4.2 });
      point.append(svg('title', {}, `${state.name}: ${state.votes} votes; relative power ${state.relativePower.toFixed(4)}`));
      marks.append(point);
    }
    marks.append(svg('circle', { class: 'electoral-selected-point', cx: x(selected.votes), cy: y(selected.relativePower), r: compact ? 6.5 : 7.5 }));
    scatter.append(marks);
    const labels = [
      ['CA', 'CA', -8, -8, 'end'],
      ['TX', 'TX', 8, 4, 'start'],
      ['FL', 'FL', 7, -7, 'start'],
      ['NY', 'NY', -12, -9, 'end'],
      ['DC', '3 votes', 8, -9, 'start']
    ];
    for (const [code, label, dx, dy, anchor] of labels) {
      const state = byCode.get(code);
      if (code === 'NY') scatter.append(svg('line', { class: 'electoral-label-leader', x1: x(state.votes) - 4, y1: y(state.relativePower) - 3, x2: x(state.votes) - 9, y2: y(state.relativePower) - 7, 'aria-hidden': 'true' }));
      scatter.append(svg('text', { class: 'electoral-point-label', x: x(state.votes) + dx, y: y(state.relativePower) + dy, 'text-anchor': anchor, 'aria-hidden': 'true' }, label));
    }
  }

  function renderSelection() {
    readout.setAttribute('aria-label', `${selected.name}, ${selected.votes} electoral votes`);
    readout.replaceChildren();
    for (const [label, value] of [['Vote share', percent(selected.voteShare)], ['Shapley share', percent(selected.share)], ['Relative power', `${selected.relativePower.toFixed(4)}×`]]) {
      const pair = document.createElement('div');
      const term = document.createElement('dt');
      const definition = document.createElement('dd');
      term.textContent = label;
      definition.textContent = value;
      pair.append(term, definition);
      readout.append(pair);
    }
    const adjustment = (selected.relativePower - 1) * 100;
    const difference = document.createElement('span');
    difference.textContent = `${selected.abbreviation}: ${Math.abs(adjustment).toFixed(2)}% ${adjustment < 0 ? 'less' : 'more'} credit than vote share.`;
    const coincidence = document.createElement('span');
    coincidence.textContent = 'Equal vote counts share a dot.';
    selectionNote.replaceChildren(difference, coincidence);
  }

  function render() {
    const width = Math.max(260, Math.round(element.getBoundingClientRect().width || 600));
    if (currentPhase === 0) drawTally(width);
    if (currentPhase === 1) drawScatter(width);
  }

  select.addEventListener('change', () => {
    selected = byCode.get(select.value);
    renderSelection();
    render();
  });

  const observer = new ResizeObserver(render);
  observer.observe(element);
  window.addEventListener('resize', render);
  renderSelection();

  function update(phase) {
    if (!Number.isInteger(phase) || phase < 0 || phase > 1) throw new Error('Use Electoral College phases 0–1.');
    currentPhase = phase;
    element.dataset.phase = String(phase);
    orderScene.hidden = phase !== 0;
    powerScene.hidden = phase !== 1;
    render();
  }

  return { element, update };
}
