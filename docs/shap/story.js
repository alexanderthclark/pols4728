import { coalition, hybridValues, marginal, precedingMasks, validateData } from './calculation.mjs';
import { renderWaterfall } from './waterfall.js';
import { featureSymbols, shapleyFormulaMarkup, valueMarkup, joiningMarkup } from './formula.js';

const $ = selector => document.querySelector(selector);
const steps = [...document.querySelectorAll('.step')];
const names = ['The Shapley formula', 'The observation x', 'The value function', 'The existing feature group S', 'The additional feature i', 'The feature’s final credit', 'A fixed model; one observation', 'The term vₓ(S)', 'The term vₓ(S ∪ {i})', 'The prediction difference', 'The factorial weight', 'The weighted sum', 'The complete explanation'];
const captions = [
  'Scroll to give each part its meaning in a prediction problem.',
  'x is one complete observation; its values supply the fixed inputs.',
  'vₓ is the value function. vₓ(S) returns the prediction average assigned to S.',
  'S is the group already fixed to x, before the additional feature joins.',
  'i is the additional feature. It belongs to F and is not already in S.',
  'ϕᵢ(vₓ) is feature i’s contribution relative to the background average prediction.',
  'Fit f once, then hold it fixed. Here we supply a transparent teaching equation.',
  'vₓ(∅): no columns fixed to x. Average the eight model predictions.',
  'After fixes ability to x; before uses its background values. Other columns stay the same.',
  'S = {n}, i = a. Subtract the two prediction averages to get one marginal.',
  'One row per preceding group S. Its weight counts how often it precedes ability.',
  'The sum of weighted prediction differences is the feature’s SHAP value.',
  'Final SHAP contributions connect the same baseline to this person’s prediction.',
];
const formulaStops = {
  'shapley-formula':{focus:'all'},
  'value-function':{focus:'value',symbol:'v<sub>x</sub>(S)',meaning:'The prediction average assigned to feature group S by the value function vₓ.'},
  'observation-symbol':{focus:'observation',symbol:'x',meaning:'The observation whose prediction we explain. Its inputs supply the values to fix.'},
  'preceding-features':{focus:'coalition',symbol:'S',meaning:'The features already fixed to x. Their prediction average is the “before” value.'},
  'joining-feature':{focus:'joining',symbol:'i ∉ S',meaning:'The additional feature joins S. Fixing its value gives the “after” group S ∪ {i}.'},
  'feature-credit':{focus:'result',symbol:'ϕ<sub>i</sub>(v<sub>x</sub>)',meaning:'The final credit for feature i: its weighted average marginal contribution.'},
};
const visual = $('#visual'), shell = $('.stage-shell');
const observationSelect = $('#observation-select');
const dialog = $('#rows-dialog');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const stackedLayout = matchMedia('(max-width:649px), (max-width:860px) and (min-height:551px)');
$('.stage-top').setAttribute('role', 'status');
$('.stage-top').setAttribute('aria-live', 'polite');
$('.stage-top').setAttribute('aria-atomic', 'true');
document.querySelectorAll('[data-formula-focus]').forEach(element => {
  element.innerHTML = shapleyFormulaMarkup(element.dataset.formulaFocus);
});
const state = { data:null, observation:null, scene:-1, feature:0, explorerMask:0, rowsTrigger:null };

function escape(value) {
  return String(value).replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
}
function number(value, decimals = 1) {
  return new Intl.NumberFormat('en-US', {maximumFractionDigits:decimals}).format(value).replace('-', '−');
}
function signed(value) { return value === 0 ? '0' : `${value < 0 ? '−' : '+'}${number(Math.abs(value))}`; }
function inputNumber(value) { return number(value, 2); }
function dollars(value, includeSign = false) {
  return `${value < 0 ? '−' : includeSign && value > 0 ? '+' : ''}$${new Intl.NumberFormat('en-US', {maximumFractionDigits:0}).format(Math.abs(value) * 1000)}`;
}
function groupName(mask, short = false) {
  const labels = state.data.features.filter((_, index) => mask & (1 << index)).map(feature => short ? feature.shortLabel : feature.label.toLowerCase());
  return labels.length ? labels.join(short ? ' + ' : ' and ') : short ? 'None' : 'no features';
}
function setNotation(mask) { return mask ? `{${featureSymbols.filter((_, index) => mask & (1 << index)).join(', ')}}` : '∅'; }

function updateFacts() {
  const observation = state.observation;
  const delta = marginal(observation, 0, 2);
  const facts = {
    ability:inputNumber(observation.values[0]), neighborhood:inputNumber(observation.values[1]),
    baselineDollars:dollars(observation.baseValue), predictionDollars:dollars(observation.prediction),
    staticObservation:`For this observation, ability is ${inputNumber(observation.values[0])}, neighborhood opportunity is ${inputNumber(observation.values[1])}, and experience is ${inputNumber(observation.values[2])}. The model predicts ${dollars(observation.prediction)} per year.`,
    abilityMeanDollars:dollars(coalition(observation, 1).value), abilityFirstDollars:dollars(marginal(observation, 0, 0), true),
    neighborhoodMeanDollars:dollars(coalition(observation, 2).value), bothMeanDollars:dollars(coalition(observation, 3).value),
    abilityAfterDollars:dollars(delta, true), abilityShapDollars:dollars(observation.shapValues[0], true),
    contextInterpretation:delta < 0
      ? 'Here the adverse context makes revealing this person’s ability lower the averaged prediction. This sign reversal comes from the interaction in our equation.'
      : delta > 0 ? 'Here revealing this person’s ability raises the averaged prediction. The size of the change depends on the neighborhood value already fixed.'
      : 'Here revealing ability does not change the averaged prediction. The main ability term and the interaction offset one another for this profile.',
  };
  document.querySelectorAll('[data-fact]').forEach(element => {element.textContent = facts[element.dataset.fact];});
  updateAverageNarrative();
  $('#explanation-table').innerHTML = `<table class="input-table"><caption>${escape(observation.name)} · earnings contributions in $1,000/year</caption><thead><tr><th scope="col">Feature</th><th scope="col">Input value</th><th scope="col">SHAP value</th></tr></thead><tbody>${state.data.features.map((feature, index) => `<tr><th scope="row">${escape(feature.label)}</th><td>${inputNumber(observation.values[index])}</td><td>${signed(observation.shapValues[index])}</td></tr>`).join('')}<tr><th scope="row">Baseline</th><td colspan="2">${number(observation.baseValue)}</td></tr><tr><th scope="row">Prediction</th><td colspan="2">${number(observation.prediction)}</td></tr></tbody></table>`;
  renderCoalitionExplorer();
}

function renderFormula(scene) {
  const stop = formulaStops[scene];
  visual.innerHTML = `${shapleyFormulaMarkup(stop.focus)}${stop.symbol ? `<div class="formula-explanation"><p class="formula-focus-symbol math-text">${stop.symbol}</p><p class="formula-focus-meaning">${stop.meaning}</p></div>` : ''}`;
}

function renderObservation() {
  const observation = state.observation;
  visual.innerHTML = `<h3>From training to explanation</h3><div class="training-flow" aria-label="In practice: training data X and y fit one prediction function f"><span>Training data<br><span class="math-text">(X, y)</span></span><span aria-hidden="true">→</span><span>Fit once<br><span class="math-text">f</span></span></div><p class="training-note">Here <var>f</var> is our supplied teaching equation.</p><div class="observation-card"><span class="date">Observation x · ${escape(observation.name)}</span><dl>${state.data.features.map((feature, index) => `<dt>${escape(feature.label)}</dt><dd>${inputNumber(observation.values[index])}${index === 1 ? observation.values[1] < 0 ? ' · adverse' : ' · favorable' : ''}</dd>`).join('')}</dl><div class="prediction-number">${dollars(observation.prediction)}<span class="prediction-label">f(x) · predicted annual earnings</span></div></div><p class="input-note">We explain this output of the fixed model.</p>`;
}

function inputCell(background, mask, index, newFeature = -1) {
  const included = Boolean(mask & (1 << index));
  const value = hybridValues(background.values, state.observation.values, mask)[index];
  const changed = included && index === newFeature && value !== background.values[index];
  return `<td class="${included ? index === newFeature ? 'newly-fixed' : 'fixed' : 'unknown'}">${changed ? `<span class="cell-original">${inputNumber(background.values[index])}</span><span class="replacement-arrow" aria-label="replaced by"> → </span>` : ''}<span class="cell-value">${inputNumber(value)}</span></td>`;
}

function inputTable(mask, {beforeMask, newFeature = -1, fullLabels = false} = {}) {
  const group = coalition(state.observation, mask);
  const before = beforeMask === undefined ? null : coalition(state.observation, beforeMask);
  return `<table class="input-table"><caption>All 8 reference rows · predictions in $1,000/year</caption><thead><tr><th scope="col">Row</th>${state.data.features.map((feature, index) => `<th scope="col" title="${escape(feature.description)}">${fullLabels ? escape(feature.shortLabel) : featureSymbols[index]}<small>${mask & (1 << index) ? 'Included<br>fixed to person' : 'Excluded<br>from background'}</small></th>`).join('')}${before ? `<th scope="col">Before<small>mean =<br>${valueMarkup(beforeMask)}</small></th><th scope="col">After<small>mean =<br>${valueMarkup(mask)}</small></th>` : `<th scope="col">Prediction<small>mean = ${valueMarkup(mask)}</small></th>`}</tr></thead><tbody>${state.data.background.map((background, row) => `<tr><td>${row + 1}</td>${state.data.features.map((_, index) => inputCell(background, mask, index, newFeature)).join('')}${before ? `<td>${number(before.predictions[row])}</td>` : ''}<td>${number(group.predictions[row])}</td></tr>`).join('')}</tbody><tfoot><tr><th scope="row" colspan="4">Average of all 8</th>${before ? `<td>${number(before.value)}</td>` : ''}<td>${number(group.value)}</td></tr></tfoot></table>`;
}

function renderBackground(mask, beforeMask) {
  const before = beforeMask === undefined ? null : coalition(state.observation, beforeMask);
  const after = coalition(state.observation, mask);
  const newFeature = before ? 0 : -1;
  visual.innerHTML = `<h3 class="coalition-heading">${before ? joiningMarkup(beforeMask, 0) : valueMarkup(mask)}</h3><p class="formula-table-context">${before ? `S = ${setNotation(beforeMask)} · i = a · after − before` : 'S = ∅ · no columns fixed to this person'}</p><p class="focal-values">x: a = ${inputNumber(state.observation.values[0])}, n = ${inputNumber(state.observation.values[1])}, e = ${inputNumber(state.observation.values[2])}</p>${inputTable(mask, {beforeMask, newFeature})}<p class="table-key">a: ability · n: neighborhood · e: experience</p><div class="table-marginal"><span>${before ? 'Ability’s marginal contribution' : 'Value of the empty coalition'}</span><span class="${before && after.value - before.value < 0 ? 'negative' : ''}">${before ? `${number(after.value)} − ${number(before.value)} = ${signed(after.value - before.value)}` : `${number(after.value)}`}</span></div>`;
}

function renderWeights() {
  const observation = state.observation;
  const masks = precedingMasks(0, state.data.features.length);
  const rows = masks.map(mask => {
    const count = observation.orders.filter(order => {
      const position = order.features.indexOf(0);
      return order.features.slice(0, position).reduce((result, index) => result | (1 << index), 0) === mask;
    }).length;
    return `<tr><th scope="row"><span class="math-text">${setNotation(mask)}</span></th><td>${number(coalition(observation, mask).value)}</td><td>${number(coalition(observation, mask | 1).value)}</td><td class="${marginal(observation, 0, mask) < 0 ? 'negative' : ''}">${signed(marginal(observation, 0, mask))}</td><td>${count}/6</td></tr>`;
  });
  visual.innerHTML = `${shapleyFormulaMarkup('weight')}<p class="formula-definitions">i = a · four possible preceding groups S</p><table class="weights-table"><caption>Prediction averages in $1,000/year</caption><thead><tr><th scope="col">S</th><th scope="col"><span class="math-text">v<sub>x</sub>(S)</span></th><th scope="col"><span class="math-text">v<sub>x</sub>(S ∪ {a})</span></th><th scope="col">Difference</th><th scope="col">Weight</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
}

function featureContexts() {
  const observation = state.observation, feature = state.feature;
  const partner = feature === 0 ? 1 : feature === 1 ? 0 : null;
  if (partner === null) return [{mask:0,value:marginal(observation, feature, 0),label:'Every preceding context',count:6}];
  return [
    {mask:0,value:marginal(observation, feature, 0),label:`${state.data.features[partner].shortLabel} still unrevealed`,count:3},
    {mask:1 << partner,value:marginal(observation, feature, 1 << partner),label:`${state.data.features[partner].shortLabel} already revealed`,count:3},
  ];
}

function updateAverageNarrative() {
  const feature = state.data.features[state.feature].label.toLowerCase();
  const symbol = featureSymbols[state.feature];
  const contexts = featureContexts();
  const credit = `<span class="math-text">${dollars(state.observation.shapValues[state.feature], true)}</span>`;
  const contributions = contexts.map(context => `<span class="math-text">${dollars(context.value, true)}</span>`);
  $('#average-definition').innerHTML = `The sum visits all four groups that exclude ${escape(feature)}. Weight each prediction difference by the fraction of orders in which that group precedes ${escape(feature)}, then add. This gives <span class="math-text">ϕ<sub>${symbol}</sub>(v<sub>x</sub>)</span>, ${escape(feature)}’s SHAP value for this observation.`;
  if (contexts.length === 2) {
    const partner = state.feature === 0 ? 'neighborhood' : 'ability';
    $('#average-context').innerHTML = `We can combine equal contributions in this example. Before ${partner} is fixed, ${escape(feature)} contributes ${contributions[0]} with total weight <span class="math-text">2/6 + 1/6 = 3/6</span>. After ${partner} is fixed, it contributes ${contributions[1]} with total weight <span class="math-text">1/6 + 2/6 = 3/6</span>. The result is ${credit}.`;
    $('#average-interpretation').textContent = `Experience’s position does not change ${feature}’s marginal here because experience enters additively.`;
  } else {
    $('#average-context').innerHTML = `Experience contributes ${contributions[0]} in every preceding group and every revealing order. Combining all six equally weighted orders gives total weight <span class="math-text">6/6 = 1</span>, so its SHAP value is ${credit}.`;
    $('#average-interpretation').textContent = 'Because experience enters additively, its marginal contribution stays the same whichever other features are already fixed.';
  }
}

function renderAverage() {
  const feature = state.data.features[state.feature];
  const contexts = featureContexts();
  visual.innerHTML = `${shapleyFormulaMarkup('sum')}<label class="orders-feature-label" for="feature-select">Feature i to explain</label><select id="feature-select">${state.data.features.map((item, index) => `<option value="${index}" ${index === state.feature ? 'selected' : ''}>${escape(item.label)}</option>`).join('')}</select><div class="marginal-contexts">${contexts.map(context => `<div class="context-summary"><span>${escape(context.label)}</span><span class="context-number ${context.value < 0 ? 'negative' : ''}">${signed(context.value)}</span><small>total weight ${context.count}/6</small></div>`).join('')}</div><div class="shap-average"><span>${escape(feature.label)}’s SHAP value<br><span class="math-text">ϕ<sub>${featureSymbols[state.feature]}</sub>(v<sub>x</sub>)</span></span><span class="number ${state.observation.shapValues[state.feature] < 0 ? 'negative' : ''}">${signed(state.observation.shapValues[state.feature])}</span></div><p class="average-arithmetic">${contexts.length === 2 ? `(3/6) × (${signed(contexts[0].value)}) + (3/6) × (${signed(contexts[1].value)})` : 'Total weight 6/6; the contribution is the same in every order.'}</p>`;
  $('#feature-select').addEventListener('change', event => {state.feature = Number(event.target.value); render();});
}

function renderFinal() {
  visual.innerHTML = '<div id="waterfall-container"></div><div class="waterfall-total"></div>';
  renderWaterfall($('#waterfall-container'), {features:state.data.features,observation:state.observation,unit:'$1,000/year'});
  $('#waterfall-container').querySelectorAll('.waterfall-contribution').forEach((bar, index, bars) => {bar.style.setProperty('--bar-delay', `${(bars.length-index-1)*.2}s`);});
  const observation = state.observation;
  $('.waterfall-total').textContent = `${number(observation.baseValue)} ${observation.shapValues.map(value => `${value < 0 ? '−' : '+'} ${number(Math.abs(value))}`).join(' ')} = ${number(observation.prediction)} ($1,000/year)`;
}

function renderCoalitionExplorer() {
  $('#coalition-table').innerHTML = inputTable(state.explorerMask, {fullLabels:true});
  $('#coalition-value').textContent = `vₓ(${setNotation(state.explorerMask)}) = ${number(coalition(state.observation, state.explorerMask).value)} ($1,000/year), for ${state.observation.name.toLowerCase()}.`;
}

function render() {
  if (!state.data) return;
  const active = document.activeElement;
  const focusSelector = visual.contains(active) && active.id ? `#${active.id}` : null;
  updateFacts();
  const scene = steps[state.scene].id;
  shell.dataset.scene = scene;
  $('.observation-control').hidden = Boolean(formulaStops[scene]);
  if ($('#stage-name').textContent !== names[state.scene]) {
    $('#stage-name').textContent = names[state.scene];
    $('#stage-count').textContent = `${state.scene+1} / ${steps.length}`;
  }
  $('#stage-caption').textContent = captions[state.scene];
  $('#previous').disabled = state.scene === 0;
  $('#next').textContent = state.scene === steps.length-1 ? 'Math & Python →' : 'Next →';
  if (formulaStops[scene]) renderFormula(scene);
  else if (scene === 'observation') renderObservation();
  else if (scene === 'background') renderBackground(0);
  else if (scene === 'reveal-ability') renderBackground(1, 0);
  else if (scene === 'neighborhood-first') renderBackground(3, 2);
  else if (scene === 'weights') renderWeights();
  else if (scene === 'shap-value') renderAverage();
  else renderFinal();
  const ordersScene = scene === 'weights' || scene === 'shap-value';
  $('#inspect').hidden = !['background','reveal-ability','neighborhood-first','weights','shap-value'].includes(scene);
  $('#inspect').textContent = ordersScene ? 'Six orders' : 'Inspect rows';
  $('#inspect').setAttribute('aria-label', ordersScene ? 'Inspect the six revealing orders' : 'Inspect hybrid input rows and predictions');
  if (focusSelector) visual.querySelector(focusSelector)?.focus({preventScroll:true});
}

function openRows(trigger) {
  state.rowsTrigger = trigger;
  const scene = steps[state.scene].id;
  const mask = scene === 'background' ? 0 : scene === 'reveal-ability' ? 1 : 3;
  const beforeMask = scene === 'reveal-ability' ? 0 : scene === 'neighborhood-first' ? 2 : undefined;
  $('#rows-heading').textContent = `Knowing ${groupName(mask)}`;
  $('#rows-description').textContent = `Included columns use the selected person’s values in every row. Excluded columns keep that reference row’s values. ${beforeMask === undefined ? `vₓ(${setNotation(mask)}) = ${number(coalition(state.observation, mask).value)}.` : `Before: vₓ(${setNotation(beforeMask)}) = ${number(coalition(state.observation, beforeMask).value)}; after: vₓ(${setNotation(mask)}) = ${number(coalition(state.observation, mask).value)}. Their difference is ${signed(marginal(state.observation, 0, beforeMask))}.`} Outputs are in $1,000/year.`;
  $('#rows-table').innerHTML = inputTable(mask, {beforeMask,newFeature:beforeMask === undefined ? -1 : 0,fullLabels:true});
  dialog.showModal();
  $('#close-rows').focus();
}
function openOrders(trigger) {
  state.rowsTrigger = trigger;
  const feature = steps[state.scene].id === 'weights' ? 0 : state.feature;
  $('#rows-heading').textContent = `Six orders for ${state.data.features[feature].label.toLowerCase()}`;
  $('#rows-description').textContent = 'Each row records the marginal contribution when this feature joins. Repeated preceding groups still receive one entry per order. Values are in $1,000/year.';
  $('#rows-table').innerHTML = `<table class="orders-table"><thead><tr><th scope="col">Revealing order</th><th scope="col">Already revealed</th><th scope="col">Marginal</th></tr></thead><tbody>${state.observation.orders.map(order => {
    const position = order.features.indexOf(feature);
    const mask = order.features.slice(0, position).reduce((result,index) => result | (1 << index), 0);
    return `<tr><td>${order.features.map(index => `<span class="${index === feature ? 'order-target' : ''}">${escape(state.data.features[index].shortLabel)}</span>`).join(' → ')}</td><td>${escape(groupName(mask))}</td><td>${signed(marginal(state.observation, feature, mask))}</td></tr>`;
  }).join('')}</tbody><tfoot><tr><th scope="row" colspan="2">Mean: the SHAP value</th><td>${signed(state.observation.shapValues[feature])}</td></tr></tfoot></table>`;
  dialog.showModal();
  $('#close-rows').focus();
}
function navigate(index, behavior = reducedMotion.matches ? 'instant' : 'smooth') {
  const target = index === steps.length ? $('#method') : steps[index];
  if (index < steps.length && stackedLayout.matches) {
    state.scene = index;
    render();
    const readingOffset = $('.masthead').getBoundingClientRect().height + shell.getBoundingClientRect().height + 15;
    const top = scrollY + target.getBoundingClientRect().top - readingOffset;
    scrollTo({top,behavior});
  } else target.scrollIntoView({behavior,block:'start'});
  history.replaceState(null,'',`#${target.id}`);
}
let scrollPending = false;
function updateScroll() {
  scrollPending = false;
  if (!state.data) return;
  const mobile = stackedLayout.matches;
  const readingLine = mobile ? shell.getBoundingClientRect().bottom + 60 : innerHeight * .5;
  let scene = 0;
  steps.forEach((step,index) => {if (step.getBoundingClientRect().top <= readingLine) scene = index;});
  if (state.scene !== scene) {state.scene = scene; render();}
  const progress = Math.max(0,Math.min(1,(scrollY-steps[0].offsetTop)/($('#method').offsetTop-steps[0].offsetTop)));
  $('#progress').style.width = `${progress*100}%`;
}
function queueScroll() {if (!scrollPending) {scrollPending = true; requestAnimationFrame(updateScroll);}}

$('#inspect').addEventListener('click', () => ['weights','shap-value'].includes(steps[state.scene].id) ? openOrders($('#inspect')) : openRows($('#inspect')));
$('#previous').addEventListener('click', () => navigate(Math.max(0,state.scene-1)));
$('#next').addEventListener('click', () => navigate(state.scene+1));
$('#close-rows').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});
dialog.addEventListener('close', () => {if (state.rowsTrigger?.isConnected) state.rowsTrigger.focus();});
observationSelect.addEventListener('change', () => {state.observation = state.data.observations.find(item => item.id === observationSelect.value); render();});
$('#coalition-controls').addEventListener('change', () => {
  state.explorerMask = [...$('#coalition-controls').querySelectorAll('input:checked')].reduce((mask,input) => mask | (1 << Number(input.value)),0);
  renderCoalitionExplorer();
});
addEventListener('scroll',queueScroll,{passive:true});
addEventListener('resize', () => {queueScroll(); if (steps[state.scene]?.id === 'waterfall') renderFinal();});
addEventListener('hashchange', () => {
  if (!state.data) return;
  const index = steps.findIndex(step => `#${step.id}` === location.hash);
  if (index >= 0) navigate(index, 'instant');
});

try {
  const response = await fetch(new URL('./data.json',import.meta.url));
  if (!response.ok) throw new Error(`Data request failed (${response.status}).`);
  state.data = validateData(await response.json());
  state.observation = state.data.observations.find(item => item.id === state.data.defaultObservationId) || state.data.observations[0];
  observationSelect.innerHTML = state.data.observations.map(item => `<option value="${escape(item.id)}">${escape(item.name)}</option>`).join('');
  observationSelect.value = state.observation.id;
  observationSelect.disabled = false;
  updateScroll();
  const anchor = location.hash && document.getElementById(location.hash.slice(1));
  if (anchor) {
    const index = steps.indexOf(anchor);
    if (index >= 0) navigate(index, 'instant');
    else anchor.scrollIntoView({behavior:'instant',block:'start'});
    queueScroll();
  }
} catch (error) {
  visual.innerHTML = '<p>The interactive data could not be loaded. The explanation and Python operation below remain available.</p>';
  $('#stage-caption').textContent = error.message;
  $('#next').disabled = true;
  console.error(error);
}
