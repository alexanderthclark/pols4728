import { fraction, matchingOrders } from './game.mjs';

const factorial = n => n < 2 ? 1 : n * factorial(n - 1);
const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const valueApplication = node => `<mrow><mi>v</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo>${node.members.length ? `<mrow><mo>{</mo>${node.members.map(player => `<mi mathvariant="normal">${escape(player)}</mi>`).join('<mo>,</mo>')}<mo>}</mo></mrow>` : '<mo>∅</mo>'}<mo stretchy="false">)</mo></mrow></mrow>`;

const equation = `
  <div class="formula-equation" role="math" aria-label="The Shapley value of player i in game v equals the sum over all coalitions S contained in N excluding i, of S-size factorial times n minus S-size minus one factorial, divided by n factorial, times the difference between v of S with i added and v of S.">
    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-hidden="true">
      <mrow>
        <mrow><msub><mi>ϕ</mi><mi>i</mi></msub><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo>
        <munder class="formula-sum"><mo>∑</mo><mrow><mi>S</mi><mo>⊆</mo><mi>N</mi><mo>∖</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></munder>
        <mfrac>
          <mrow>
            <mrow class="formula-before"><mrow><mo form="prefix" stretchy="false">|</mo><mi>S</mi><mo form="postfix" stretchy="false">|</mo></mrow><mo>!</mo></mrow>
            <mrow class="formula-after"><mrow><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mrow><mo form="prefix" stretchy="false">|</mo><mi>S</mi><mo form="postfix" stretchy="false">|</mo></mrow><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow><mo>!</mo></mrow>
          </mrow>
          <mrow class="formula-denominator"><mi>n</mi><mo>!</mo></mrow>
        </mfrac>
      </mrow>
    </math>
    <math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-hidden="true">
      <mrow class="formula-difference"><mo stretchy="false">[</mo><mrow><mi>v</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>S</mi><mo>∪</mo><mrow><mo>{</mo><mi>i</mi><mo>}</mo></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mi>v</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>S</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">]</mo></mrow>
    </math>
  </div>`;

/** Connect the general equation to the same coalition edges used by the story. */
export function createFormulaView(game, { onCoalitionChange = () => {}, onExplainWeight = () => {} } = {}) {
  const player = game.players[0];
  const n = game.players.length;
  const joiningEdges = game.edges.filter(edge => edge.playerIndex === 0).sort((a, b) => a.from - b.from);
  let currentEdge = joiningEdges.find(edge => edge.from === 2) || joiningEdges[0];
  let currentPhase = -1;
  const element = document.createElement('section');
  element.className = 'formula-view';
  element.setAttribute('aria-label', 'From coalition paths to the Shapley formula');
  element.innerHTML = `${equation}
    <p class="formula-definitions"></p>
    <div class="formula-choices" role="group" aria-label="Choose the coalition before the selected player joins">
      <span class="formula-choices-label"><math class="math-inline"><mi>S</mi><mo>=</mo></math></span>
      ${joiningEdges.map(edge => `<button type="button" data-formula-edge="${escape(edge.id)}" aria-pressed="false" aria-label="Coalition ${escape(game.nodes[edge.from].label)} before ${escape(player)} joins">${escape(game.nodes[edge.from].label)}</button>`).join('')}
    </div>
    <p class="formula-context" aria-live="polite" aria-atomic="true"></p>
    <div class="formula-orders"></div>
    <div class="formula-numerical" hidden></div>
    <button type="button" class="formula-weight-link" aria-haspopup="dialog" aria-controls="weight-dialog" hidden>See these six paths</button>`;

  const definitions = element.querySelector('.formula-definitions');
  const choices = element.querySelector('.formula-choices');
  const context = element.querySelector('.formula-context');
  const orders = element.querySelector('.formula-orders');
  const numerical = element.querySelector('.formula-numerical');
  const weightButton = element.querySelector('.formula-weight-link');

  function orderStrips(matches) {
    return `<div class="formula-order-head" aria-hidden="true"><span>Before ${escape(player)}</span><span>${escape(player)} joins</span><span>After ${escape(player)}</span></div>
      <div class="formula-order-strips" role="list" aria-label="Orders using this joining edge">${matches.map(order => {
        const at = order.order.indexOf(0);
        const before = order.order.slice(0, at).map(i => game.players[i]);
        const after = order.order.slice(at + 1).map(i => game.players[i]);
        const names = list => list.length ? list.join(' → ') : '∅';
        return `<div class="formula-order-strip" role="listitem" aria-label="${escape(order.label)}; before ${escape(player)}: ${escape(before.join(', ') || 'no players')}; after ${escape(player)}: ${escape(after.join(', ') || 'no players')}"><span class="formula-order-before${currentPhase === 2 ? ' formula-order-emphasis' : ''}">${escape(names(before))}</span><span class="formula-order-player">${escape(player)}</span><span class="formula-order-after${currentPhase === 3 ? ' formula-order-emphasis' : ''}">${escape(names(after))}</span></div>`;
      }).join('')}</div>`;
  }

  function render() {
    const coalition = game.nodes[currentEdge.from];
    const added = game.nodes[currentEdge.to];
    const beforeCount = factorial(coalition.size);
    const afterSize = n - coalition.size - 1;
    const afterCount = factorial(afterSize);
    const matches = matchingOrders(game, currentEdge.id);
    element.dataset.phase = String(currentPhase);
    element.dataset.edge = currentEdge.id;
    element.dataset.matchCount = String(matches.length);
    definitions.innerHTML = `<span><i>N</i> = ${escape(`{${game.players.join(', ')}}`)}</span><span><i>n</i> = ${n}</span><span><i>i</i> = ${escape(player)}</span>${currentPhase > 0 && currentPhase < 5 ? `<span><i>S</i> = ${escape(coalition.label)}</span>` : ''}`;
    for (const button of choices.querySelectorAll('button')) {
      button.setAttribute('aria-pressed', String(button.dataset.formulaEdge === currentEdge.id));
    }
    choices.hidden = currentPhase === 0 || currentPhase === 5;
    numerical.hidden = currentPhase !== 5;
    orders.hidden = ![2, 3, 4].includes(currentPhase);
    weightButton.hidden = ![4, 5].includes(currentPhase);
    weightButton.textContent = currentPhase === 5 ? 'Revisit the six paths behind a weight' : `See these ${game.totalOrders === 6 ? 'six' : game.totalOrders} paths`;
    for (const name of ['sum', 'difference', 'before', 'after', 'denominator']) {
      const activePhase = { sum: 0, difference: 1, before: 2, after: 3, denominator: 4 }[name];
      element.querySelector(`.formula-${name}`).classList.toggle('formula-active', currentPhase === activePhase || currentPhase === 5);
    }
    if (currentPhase === 0) {
      context.textContent = `Sum over the ${joiningEdges.length} coalitions that exclude ${player}: one term for each ${player}-joining edge.`;
    } else if (currentPhase === 1) {
      context.innerHTML = `For ${escape(coalition.label)} → ${escape(added.label)}, the contribution is <math class="math-inline" aria-label="${escape(`v of ${added.label} minus v of ${coalition.label} equals ${added.value} minus ${coalition.value} equals ${currentEdge.delta}`)}"><mrow>${valueApplication(added)}<mo>−</mo>${valueApplication(coalition)}<mo>=</mo><mn>${added.value}</mn><mo>−</mo><mn>${coalition.value}</mn><mo>=</mo><mn>${currentEdge.delta}</mn></mrow></math>.`;
    } else if (currentPhase === 2) {
      context.textContent = `|S|! = ${coalition.size}! = ${beforeCount}: ${beforeCount === 1 ? 'one order' : `${beforeCount} orders`} for the ${coalition.size} ${coalition.size === 1 ? 'player' : 'players'} before ${player}.${coalition.size === 0 ? ' The empty group has one arrangement: 0! = 1.' : ''}`;
      orders.innerHTML = orderStrips(matches);
    } else if (currentPhase === 3) {
      context.textContent = `(n − |S| − 1)! = ${afterSize}! = ${afterCount}: ${afterCount === 1 ? 'one order' : `${afterCount} orders`} for the ${afterSize} ${afterSize === 1 ? 'player' : 'players'} after ${player}.${afterSize === 0 ? ' The empty group has one arrangement: 0! = 1.' : ''}`;
      orders.innerHTML = orderStrips(matches);
    } else if (currentPhase === 4) {
      context.textContent = `n! = ${n}! = ${game.totalOrders} equally likely orders. Weight = ${beforeCount} × ${afterCount} / ${game.totalOrders} = ${currentEdge.count}/${game.totalOrders}; ${matches.length} use this edge.`;
      orders.innerHTML = `<div class="formula-all-orders" role="list" aria-label="All ${game.totalOrders} equally likely orders">${game.orders.map(order => `<span role="listitem" class="formula-order-chip${order.path.some(edge => edge.id === currentEdge.id) ? ' formula-order-match' : ''}" aria-label="${escape(order.label)}${order.path.some(edge => edge.id === currentEdge.id) ? '; uses this edge' : '; uses a different joining edge'}">${escape(order.label)}</span>`).join('')}</div>`;
    } else {
      context.textContent = `Add the ${joiningEdges.length} weighted contributions. ${player}’s Shapley value is ${fraction(game.shares[0])}.`;
      numerical.innerHTML = `<div class="formula-numeric-sum" role="math" aria-label="${escape(joiningEdges.map(edge => `${edge.delta} times ${edge.count} over ${game.totalOrders}`).join(' plus ') + ` equals ${fraction(game.shares[0])}`)}">${joiningEdges.map((edge, index) => `${index ? '<span class="formula-numeric-plus" aria-hidden="true">+</span>' : ''}<span class="formula-numeric-term${edge.delta ? ' formula-numeric-nonzero' : ''}" aria-hidden="true">${edge.delta} × <span>${edge.count}/${game.totalOrders}</span></span>`).join('')}<span class="formula-numeric-total" aria-hidden="true">= ${fraction(game.shares[0])}</span></div>`;
    }
  }

  choices.addEventListener('click', event => {
    const button = event.target.closest('button[data-formula-edge]');
    if (!button) return;
    currentEdge = joiningEdges.find(edge => edge.id === button.dataset.formulaEdge);
    render();
    onCoalitionChange(currentEdge);
  });
  weightButton.addEventListener('click', () => onExplainWeight(currentEdge.id, weightButton));

  function update(phase) {
    if (!Number.isInteger(phase) || phase < 0 || phase > 5) throw new Error('Use formula phases 0–5.');
    if (phase === currentPhase) return;
    currentPhase = phase;
    if (phase === 1) currentEdge = joiningEdges.find(edge => edge.from === 2) || currentEdge;
    if (phase === 2) currentEdge = joiningEdges.find(edge => edge.from === 6) || currentEdge;
    if (phase === 3) currentEdge = joiningEdges.find(edge => edge.from === 0) || currentEdge;
    render();
  }
  update(0);
  return { element, update, get edge() { return currentEdge; }, get phase() { return currentPhase; } };
}
