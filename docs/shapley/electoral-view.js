const svgNS = 'http://www.w3.org/2000/svg';
const coalitionCodes = ['TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ'];
let viewCount = 0;

function svg(tag, attributes = {}, text) {
  const element = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (text != null) element.textContent = text;
  return element;
}

/** California's marginal contribution to one illustrative coalition. */
export function createElectoralView(results) {
  const id = `electoral-${++viewCount}`;
  const byCode = new Map(results.map(state => [state.abbreviation, state]));
  const california = byCode.get('CA');
  const coalition = coalitionCodes.map(code => byCode.get(code));
  const totalVotes = results.reduce((sum, state) => sum + state.votes, 0);
  const coalitionVotes = coalition.reduce((sum, state) => sum + state.votes, 0);
  const withCalifornia = coalitionVotes + california.votes;
  const quota = Math.floor(totalVotes / 2) + 1;
  const element = document.createElement('section');
  element.className = 'electoral-view';
  element.setAttribute('aria-label', 'California’s marginal contribution to an Electoral College coalition');
  element.innerHTML = `
    <div class="electoral-scene electoral-coalition-scene">
      <p class="electoral-model">Adding California makes this coalition win.</p>
      <svg class="electoral-tally" role="img" aria-labelledby="${id}-coalition-title ${id}-coalition-description"></svg>
      <p class="electoral-change" aria-label="Coalition value changes from zero to one when California joins">
        <span>Without CA: <span class="electoral-payoff">0</span></span>
        <span aria-hidden="true">→</span>
        <span>With CA: <span class="electoral-payoff electoral-winning">1</span></span>
      </p>
    </div>`;
  const tally = element.querySelector('.electoral-tally');
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
      svg('title', { id: `${id}-coalition-title` }, 'Adding California makes this coalition win'),
      svg('desc', { id: `${id}-coalition-description` }, `The coalition consists of ${coalition.map(state => `${state.name}, ${state.votes} votes`).join('; ')}. Those ${coalition.length} states total ${coalitionVotes} votes, below the ${quota}-vote threshold, so the coalition value is zero. California adds ${california.votes} votes, bringing the total to ${withCalifornia}, so the value becomes one. California’s marginal contribution to this coalition is one minus zero, or one. This comparison does not specify a joining order or claim that California is the only critical state in the resulting coalition. The axis measures electoral votes from zero to the national total of ${totalVotes}.`)
    );
    const labels = [
      ['Coalition', `${coalitionVotes} votes`, (x(0) + x(coalitionVotes)) / 2],
      ['CA', `+${california.votes}`, (x(coalitionVotes) + x(withCalifornia)) / 2]
    ];
    for (const [label, votes, at] of labels) {
      tally.append(svg('text', { class: 'electoral-tally-label', x: at, y: labelY, 'text-anchor': 'middle' }, label));
      tally.append(svg('text', { class: label === 'CA' ? 'electoral-tally-votes electoral-ca-text' : 'electoral-tally-votes', x: at, y: labelY + 18, 'text-anchor': 'middle' }, votes));
    }
    tally.append(
      svg('line', { class: 'electoral-axis', x1: left, x2: right, y1: barY + barHeight, y2: barY + barHeight }),
      svg('rect', { class: 'electoral-coalition-bar', x: left, y: barY, width: x(coalitionVotes) - left, height: barHeight }),
      svg('rect', { class: 'electoral-ca-bar', x: x(coalitionVotes), y: barY, width: x(withCalifornia) - x(coalitionVotes), height: barHeight }),
      svg('line', { class: 'electoral-quota-line', x1: x(quota), x2: x(quota), y1: barY - 6, y2: quotaY - 15 })
    );
    for (const votes of [0, coalitionVotes, withCalifornia, totalVotes]) {
      tally.append(svg('line', { class: 'electoral-tick', x1: x(votes), x2: x(votes), y1: barY + barHeight, y2: barY + barHeight + 5 }));
      tally.append(svg('text', { class: 'electoral-tick-label', x: x(votes), y: tickY, 'text-anchor': votes === 0 ? 'start' : votes === totalVotes ? 'end' : 'middle' }, String(votes)));
    }
    tally.append(svg('text', { class: 'electoral-quota-label', x: x(quota), y: quotaY, 'text-anchor': 'middle' }, `${quota} to win`));
  }


  function render() {
    const width = Math.max(260, Math.round(element.getBoundingClientRect().width || 600));
    drawTally(width);
  }
  function update(phase) {
    if (phase !== 0) throw new Error('The coalition example has one phase.');
    element.dataset.phase = '0';
    render();
  }
  new ResizeObserver(render).observe(element);
  window.addEventListener('resize', render);
  return { element, update };
}
