const svgNS = 'http://www.w3.org/2000/svg';
const beforeCalifornia = ['TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ'];
let viewCount = 0;

function svg(tag, attributes = {}, text) {
  const element = document.createElementNS(svgNS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (text != null) element.textContent = text;
  return element;
}

/** One pivotal state in a valid illustrative Electoral College order. */
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
  const element = document.createElement('section');
  element.className = 'electoral-view';
  element.setAttribute('aria-label', 'A pivotal Electoral College state');
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


  function render() {
    const width = Math.max(260, Math.round(element.getBoundingClientRect().width || 600));
    drawTally(width);
  }
  function update(phase) {
    if (phase !== 0) throw new Error('The pivotal tally has one phase.');
    element.dataset.phase = '0';
    render();
  }
  new ResizeObserver(render).observe(element);
  window.addEventListener('resize', render);
  return { element, update };
}
