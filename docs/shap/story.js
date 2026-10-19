import { coalition, hybridValues, marginal, precedingMasks, validateData } from './calculation.mjs';
import { renderWaterfall } from './waterfall.js';
import { buildPopulation } from './aggregate.mjs';
import { renderAggregate } from './aggregate-view.js';
import { featureLabels, shapleyFormulaMarkup, valueMarkup, joiningMarkup, predictionMarkup } from './formula.js';
import { validateBreadPeace, breadPeaceFacts, renderBreadPeace } from './bread-peace.js';

const $ = selector => document.querySelector(selector);
const steps = [...document.querySelectorAll('.step')];
const names = ['The Shapley formula', `The observation ${symbol('x')}`, `The model prediction ${yHatMarkup()}`, 'Average background predictions', 'The feature’s final credit', 'Bread and Peace', 'Three correlations', 'Three OLS fits', 'One election; the model to explain', 'Bread first', 'Peace first', 'Two orders; two SHAP values', 'The value-function recap', 'When features interact', 'An interaction model; one person', `The term ${valueSymbolMarkup('<mi>S</mi>')}`, `The term ${valueSymbolMarkup('<mi>S</mi><mo>∪</mo><mrow><mo stretchy="false">{</mo><mi>i</mi><mo stretchy="false">}</mo></mrow>')}`, 'The prediction difference', 'The factorial weight', 'The weighted sum', 'The complete explanation', 'One common zero', 'One person, three dots', 'Explain more people', 'Make overlap visible', 'Color by input value', 'Opposing contributions cancel', 'Measure distance from zero', 'Average the distances', 'Rank the averages'];
const captions = [
  'Scroll to give each part its meaning in a prediction problem.',
  `${symbol('x')} is one complete observation; its values supply the fixed inputs.`,
  `${predictionMarkup()} is the fitted model’s prediction for one complete input row.`,
  'Fix the included inputs, predict with each background row, then average the predictions.',
  `${creditMarkup()} is feature ${symbol('i')}’s contribution relative to the background average prediction.`,
  '',
  'Standardized features and observed vote: three correlations determine the OLS fits.',
  '',
  `Linear model + overall background mean ${inlineMath('<mtext>peace</mtext><mo>=</mo><mn>0</mn>')}:`,
  'For this linear model, averaging predictions equals predicting at the excluded input’s background mean, 0.',
  'For this linear model, averaging predictions equals predicting at the excluded input’s background mean, 0.',
  `Each order has weight ${fractionMarkup(1, 2)}. Predictions are in SD of vote share.`,
  '',
  '',
  `Fit ${symbol('f')} once, then hold it fixed. Here we supply a transparent teaching equation.`,
  `${valueMarkup(0)}: no columns fixed to ${symbol('x')}. Average the eight model predictions.`,
  `After fixes ability to ${symbol('x')}; before uses its background values. Other columns stay the same.`,
  `${coalitionDefinitionMarkup(2)}, ${inlineMath('<mi>i</mi><mo>=</mo><mtext>ability</mtext>')}. Subtract the two prediction averages to get one marginal.`,
  `One row per preceding group ${symbol('S')}. Its weight counts how often it precedes ability.`,
  'The sum of weighted prediction differences is the feature’s SHAP value.',
  'Start at the baseline and add contributions from bottom to top. Bar direction shows whether each contribution raises or lowers the prediction.',
  'The length and sign stay the same; cumulative positions move to a common zero.',
  'Three dots are one person’s three contributions, in $1,000/year.',
  'These 125 profiles are being explained. The same eight background rows define every explanation; the model and baseline stay fixed.',
  'Horizontal position is SHAP value. Vertical packing only reveals overlap.',
  'Color shows low-to-high input values within each feature. Horizontal position shows the signed contribution to the prediction.',
  'For this balanced grid, all three signed means are zero. Positive and negative contributions cancel despite having substantial size.',
  'Reflect negative contributions to the right of zero. Absolute values retain size and remove direction.',
  'Average the 125 absolute contributions within each feature. The bar shows average size, in $1,000/year; direction has been removed.',
  'This ranking summarizes these 125 profiles. It depends on the prediction model, background rows, and profiles being summarized.',
];
const aggregateScenes = new Set(['contribution-axis','one-person-dots','many-people','beeswarm','feature-color','signed-average','absolute-values','global-importance','importance-ranking']);
const chartScenes = new Set(['waterfall', ...aggregateScenes]);
steps.filter(step => chartScenes.has(step.id)).forEach(step => step.classList.add('chart-step'));
const formulaStops = {
  'shapley-formula':{focus:'all',meaning:'The players are input features. The game assigns credit for one observation’s prediction.',note:`${symbol('F')}: all features; ${inlineMath('<mi>p</mi><mo>=</mo><mrow><mo stretchy="false">|</mo><mi>F</mi><mo stretchy="false">|</mo></mrow>')}. Excluded inputs come from background rows.`},
  'observation-symbol':{focus:'observation',symbol:symbol('x'),meaning:'One complete input row. Its values supply every feature fixed in this explanation.'},
  'prediction-symbol':{focus:'all',symbol:predictionMarkup(),meaning:`${predictionMarkup()} is the fitted model’s prediction for the complete row ${symbol('x')}. ${symbol('y')} is observed.`},
  'background-average':{focus:'value',definition:true},
  'feature-credit':{focus:'result',symbol:creditMarkup(),meaning:`Feature ${symbol('i')}’s marginal contribution is the change in the average prediction when we also fix it to its value in ${symbol('x')}.`,note:'Average these marginals across orders to get its SHAP value, in prediction units.'},
};
const transitionScenes = new Set(['bread-peace-intro', 'earnings-intro']);
const readableScenes = new Set(['ols-model','ols-income-first','ols-fatalities-first','ols-shap','observation','background','reveal-ability','neighborhood-first','weights','shap-value']);
steps.filter(step => readableScenes.has(step.id)).forEach(step => step.classList.add('readable-step'));
const isCenteredScene = scene => Boolean(formulaStops[scene]) || transitionScenes.has(scene) || readableScenes.has(scene) || ['ols-fits','ols-refit','ols-recap'].includes(scene);
const canonicalSceneId = id => ({'joining-feature':'feature-credit','preceding-features':'background-average','value-function':'background-average'}[id] || id);
const visual = $('#visual'), shell = $('.stage-shell'), story = $('#story');
const figureScrollHint = $('#figure-scroll-hint');
const observationSelect = $('#observation-select');
const dialog = $('#rows-dialog');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const stackedLayout = matchMedia('(max-width:649px), (max-width:860px) and (min-height:551px)');
$('.stage-top').setAttribute('role', 'status');
$('.stage-top').setAttribute('aria-live', 'polite');
$('.stage-top').setAttribute('aria-atomic', 'true');
const state = { data:null, observation:null, population:null, globalObservation:null, aggregateOrder:null, aggregateProgress:0, breadData:null, breadObservation:null, selectorContext:null, scene:-1, nameScene:-1, feature:0, explorerMask:0, rowsTrigger:null };
const yHat = '<math aria-label="y hat"><mover accent="true"><mi>y</mi><mo>^</mo></mover></math>';

function escape(value) {
  return String(value).replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
}
// Inline notation uses the same native MathML as the displayed formula. Keep
// function fences at the argument's size, even beside fractions or powers.
function inlineMath(content) { return `<math class="math-inline math-text"><mrow>${content}</mrow></math>`; }
function symbol(letter) { return inlineMath(`<mi>${escape(letter)}</mi>`); }
function yHatMarkup() { return inlineMath('<mover accent="true"><mi>y</mi><mo>^</mo></mover>'); }
function parenthesesBody(argument) {
  return `<mrow><mo stretchy="false" lspace="0em" rspace="0em">(</mo>${argument}<mo stretchy="false" lspace="0em" rspace="0em">)</mo></mrow>`;
}
function applicationBody(name, argument) {
  return `<mrow>${name}<mo>⁡</mo>${parenthesesBody(argument)}</mrow>`;
}
function valueSymbolMarkup(argument) { return inlineMath(applicationBody('<msub><mi>v</mi><mi>x</mi></msub>', argument)); }
function creditBody(argument = '<msub><mi>v</mi><mi>x</mi></msub>', feature = '<mi>i</mi>') {
  return applicationBody(`<msub><mi>ϕ</mi>${feature}</msub>`, argument);
}
function creditMarkup(feature) { return inlineMath(creditBody(undefined, feature)); }
function numericBody(text) {
  const value = String(text);
  const sign = /^[−+]/.test(value) ? `<mo>${value[0]}</mo>` : '';
  return `${sign}<mn>${escape(sign ? value.slice(1) : value)}</mn>`;
}
function fractionBody(numerator, denominator) { return `<mfrac><mn>${escape(numerator)}</mn><mn>${escape(denominator)}</mn></mfrac>`; }
function fractionMarkup(numerator, denominator) { return inlineMath(fractionBody(numerator, denominator)); }
function setBody(mask) {
  const members = featureLabels.filter((_, index) => mask & (1 << index));
  return members.length ? `<mrow><mo stretchy="false">{</mo>${members.map(name => `<mtext>${escape(name)}</mtext>`).join('<mo>,</mo>')}<mo stretchy="false">}</mo></mrow>` : '<mo>∅</mo>';
}
function coalitionDefinitionMarkup(mask) { return inlineMath(`<mi>S</mi><mo>=</mo>${setBody(mask)}`); }
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
  };
  document.querySelectorAll('[data-fact]').forEach(element => {element.textContent = facts[element.dataset.fact];});
  const breadFacts = breadPeaceFacts(state.breadData,state.breadObservation);
  document.querySelectorAll('[data-bread-fact]').forEach(element => {element.textContent = breadFacts[element.dataset.breadFact];});
  $('.static-refit').innerHTML = renderBreadPeace('ols-refit',state.breadData,state.breadObservation).replace(/^<h3>.*?<\/h3>/, '');
  $('.static-recap').innerHTML = renderBreadPeace('ols-recap',state.breadData,state.breadObservation).replace(/^<h3>.*?<\/h3>/, '');
  updateAverageNarrative();
  renderCoalitionExplorer();
}

function renderFormula(scene) {
  const stop = formulaStops[scene];
  const definition = stop.definition ? $(`#${scene} .background-average-content`).outerHTML : '';
  visual.innerHTML = `<h1 class="formula-intro-title">From Shapley values to SHAP</h1>${shapleyFormulaMarkup(stop.focus)}<div class="formula-explanation">${definition}${stop.symbol ? `<p class="formula-focus-symbol math-text">${stop.symbol}</p>` : ''}${stop.definition ? '' : `<p class="formula-focus-meaning">${stop.meaning}</p>`}${stop.note ? `<p class="formula-focus-note">${stop.note}</p>` : ''}${scene === 'shapley-formula' ? '<p class="formula-scroll-prompt">Scroll to connect the symbols to predictions, or use Next.</p>' : ''}</div>`;
}

function renderTransition(scene) {
  visual.innerHTML = $(`#${scene} .transition-content`).outerHTML;
}

function renderObservation() {
  const observation = state.observation;
  const equation = $('#observation .prediction-equation').outerHTML;
  visual.innerHTML = `<h3>Ability’s contribution depends on neighborhood</h3>${equation}<p class="training-note">Ability and experience: 0–1. Neighborhood: −1 (adverse) to +1 (favorable). Output: $1,000/year.</p><div class="observation-card"><span class="date">Observation ${symbol('x')} · ${escape(observation.name)}</span><dl>${state.data.features.map((feature, index) => `<dt>${escape(feature.label)}</dt><dd>${inputNumber(observation.values[index])}${index === 1 ? observation.values[1] < 0 ? ' · adverse' : ' · favorable' : ''}</dd>`).join('')}</dl><div class="prediction-number">${dollars(observation.prediction)}<span class="prediction-label">${predictionMarkup()} ${inlineMath(`<mo>=</mo>${applicationBody('<mi>f</mi>', '<mi>x</mi>')}`)} · predicted annual earnings</span></div></div><p class="input-note">Hold this model fixed throughout the explanation.</p>`;
}

function inputCell(background, mask, index, newFeature = -1) {
  const included = Boolean(mask & (1 << index));
  const value = hybridValues(background.values, state.observation.values, mask)[index];
  const changed = included && index === newFeature && value !== background.values[index];
  return `<td class="${included ? index === newFeature ? 'newly-fixed' : 'fixed' : 'unknown'}">${changed ? `<span class="cell-original">${inputNumber(background.values[index])}</span><span class="replacement-arrow" aria-label="replaced by"> → </span>` : ''}<span class="cell-value">${inputNumber(value)}</span></td>`;
}

function inputTable(mask, {beforeMask, newFeature = -1} = {}) {
  const group = coalition(state.observation, mask);
  const before = beforeMask === undefined ? null : coalition(state.observation, beforeMask);
  const predictionPair = (after, prior) => prior === undefined ? number(after) : `<span class="prediction-pair"><span class="prediction-before">${number(prior)}</span><span class="prediction-arrow" aria-label="becomes"> → </span><span class="prediction-after">${number(after)}</span></span>`;
  return `<table class="input-table"><caption>All 8 reference rows · predictions in $1,000/year</caption><thead><tr><th scope="col" class="reference-row">Row</th>${state.data.features.map((feature, index) => `<th scope="col" title="${escape(feature.description)}">${escape(feature.shortLabel)}<small>${mask & (1 << index) ? index === newFeature ? 'Background → person' : 'From person' : 'Background'}</small></th>`).join('')}<th scope="col">${yHat}<small>${before ? 'Before → After' : 'Prediction'}</small></th></tr></thead><tbody>${state.data.background.map((background, row) => `<tr><td class="reference-row">${row + 1}</td>${state.data.features.map((_, index) => inputCell(background, mask, index, newFeature)).join('')}<td>${predictionPair(group.predictions[row],before?.predictions[row])}</td></tr>`).join('')}</tbody><tfoot><tr><td class="reference-row" aria-hidden="true"></td><th scope="row" colspan="3">Average of all 8</th><td>${predictionPair(group.value,before?.value)}</td></tr></tfoot></table>`;
}

function renderBackground(mask, beforeMask) {
  const before = beforeMask === undefined ? null : coalition(state.observation, beforeMask);
  const after = coalition(state.observation, mask);
  const newFeature = before ? 0 : -1;
  const heading = beforeMask === 2 ? 'Reveal ability after neighborhood' : before ? 'Reveal ability first' : 'Start at the background average';
  const explanation = beforeMask === 2 ? `Keep neighborhood at ${inputNumber(state.observation.values[1])} in both predictions; then fix ability to ${inputNumber(state.observation.values[0])}.` : before ? `Fix ability to ${inputNumber(state.observation.values[0])}; keep neighborhood and experience from each reference row.` : 'Use every reference row unchanged. Average the model’s eight predictions.';
  visual.innerHTML = `<h3>${heading}</h3>${before ? joiningMarkup(beforeMask, 0) : `<div class="coalition-heading">${valueMarkup(mask)}</div>`}<p class="formula-table-context">${explanation}</p><p class="focal-values">Person ${symbol('x')}: ${featureLabels.map((label,index) => `<span class="feature-input">${inlineMath(`<mtext>${escape(label)}</mtext><mo>=</mo>${numericBody(inputNumber(state.observation.values[index]))}`)}</span>`).join(' · ')}</p><div class="table-marginal"><span>${before ? 'Ability’s marginal ($1,000/year)' : 'Baseline ($1,000/year)'}</span><span class="${before && after.value - before.value < 0 ? 'negative' : ''}">${before ? inlineMath(`${numericBody(number(after.value))}<mo>−</mo>${numericBody(number(before.value))}<mo>=</mo>${numericBody(signed(after.value - before.value))}`) : `${number(after.value)}`}</span></div>${inputTable(mask, {beforeMask, newFeature})}`;
}

function renderWeights() {
  const observation = state.observation;
  const masks = precedingMasks(0, state.data.features.length);
  const rows = masks.map(mask => {
    const count = observation.orders.filter(order => {
      const position = order.features.indexOf(0);
      return order.features.slice(0, position).reduce((result, index) => result | (1 << index), 0) === mask;
    }).length;
    return `<tr><th scope="row"><span class="feature-set">${escape(groupName(mask, true))}</span></th><td>${number(coalition(observation, mask).value)}</td><td>${number(coalition(observation, mask | 1).value)}</td><td class="${marginal(observation, 0, mask) < 0 ? 'negative' : ''}">${signed(marginal(observation, 0, mask))}</td><td>${fractionMarkup(count, 6)}</td></tr>`;
  });
  visual.innerHTML = `<h3>Four groups before ability joins</h3>${shapleyFormulaMarkup('weight')}<p class="formula-definitions">${inlineMath('<mi>i</mi><mo>=</mo><mtext>ability</mtext>')}. Three features give ${inlineMath('<mn>3</mn><mo>!</mo><mo>=</mo><mn>6</mn>')} orders. Each group’s weight is its fraction of those orders.</p><table class="weights-table"><caption>Prediction averages in $1,000/year</caption><thead><tr><th scope="col">${symbol('S')} · fixed first</th><th scope="col" aria-label="v sub x of S, before adding ability">Before</th><th scope="col" aria-label="v sub x of S union ability, after adding ability">After</th><th scope="col">Marginal</th><th scope="col">Weight</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
}

function featureContexts() {
  const observation = state.observation, feature = state.feature;
  const partner = feature === 0 ? 1 : feature === 1 ? 0 : null;
  if (partner === null) return [{mask:0,value:marginal(observation, feature, 0),label:'Every preceding context',count:6}];
  return [
    {mask:0,value:marginal(observation, feature, 0),label:`${state.data.features[partner].shortLabel} from background`,count:3},
    {mask:1 << partner,value:marginal(observation, feature, 1 << partner),label:`${state.data.features[partner].shortLabel} fixed to person`,count:3},
  ];
}

function updateAverageNarrative() {
  const feature = state.data.features[state.feature].shortLabel;
  const contexts = featureContexts();
  const credit = inlineMath(`<mtext>${escape(dollars(state.observation.shapValues[state.feature], true))}</mtext>`);
  const contributions = contexts.map(context => inlineMath(`<mtext>${escape(dollars(context.value, true))}</mtext>`));
  $('#average-definition').textContent = `${feature}: average six orders.`;
  if (contexts.length === 2) {
    const partner = state.feature === 0 ? 'neighborhood' : 'ability';
    $('#average-context').innerHTML = `${partner[0].toUpperCase()+partner.slice(1)} from background: ${contributions[0]}. Fixed to this person: ${contributions[1]}.`;
    $('#average-interpretation').innerHTML = `Three of six orders in each group. Their weighted average is ${credit}.`;
  } else {
    $('#average-context').innerHTML = `Same marginal in all six orders: ${contributions[0]}.`;
    $('#average-interpretation').innerHTML = `Total weight ${inlineMath('<mn>1</mn>')} → ${credit}.`;
  }
}

function renderAverage() {
  const feature = state.data.features[state.feature];
  const contexts = featureContexts();
  const weightedTerm = context => `${fractionBody(3, 6)}<mo>×</mo>${parenthesesBody(numericBody(signed(context.value)))}`;
  const arithmetic = contexts.length === 2
    ? `${inlineMath(weightedTerm(contexts[0]))} ${inlineMath(`<mo>+</mo>${weightedTerm(contexts[1])}`)} ${inlineMath(`<mo>=</mo>${numericBody(signed(state.observation.shapValues[state.feature]))}`)}`
    : `Total weight ${fractionMarkup(6, 6)}; the contribution is the same in every order.`;
  visual.innerHTML = `<h3>Average the marginal contributions</h3><label class="orders-feature-label" for="feature-select">Feature ${symbol('i')} to explain · $1,000/year</label><select id="feature-select">${state.data.features.map((item, index) => `<option value="${index}" ${index === state.feature ? 'selected' : ''}>${escape(item.label)}</option>`).join('')}</select><div class="marginal-contexts">${contexts.map(context => `<div class="context-summary"><span>${escape(context.label)}</span><span class="context-number ${context.value < 0 ? 'negative' : ''}">${signed(context.value)}</span><small>${context.count} of 6 orders</small></div>`).join('')}</div><div class="shap-average"><span>${escape(feature.label)}’s SHAP value<br>${creditMarkup(`<mtext>${escape(featureLabels[state.feature])}</mtext>`)}</span><span class="number ${state.observation.shapValues[state.feature] < 0 ? 'negative' : ''}">${signed(state.observation.shapValues[state.feature])}</span></div><p class="average-arithmetic">${arithmetic}</p>${shapleyFormulaMarkup('sum')}`;
  $('#feature-select').addEventListener('change', event => {state.feature = Number(event.target.value); render();});
}

function renderFinal() {
  visual.innerHTML = '<div class="waterfall-total"></div><div id="waterfall-container"></div>';
  renderWaterfall($('#waterfall-container'), {features:state.data.features,observation:state.observation,unit:'$1,000/year'});
  $('#waterfall-container').querySelectorAll('.waterfall-contribution').forEach((bar, index, bars) => {bar.style.setProperty('--bar-delay', `${(bars.length-index-1)*.2}s`);});
  const observation = state.observation;
  const additions = [...state.aggregateOrder].reverse().map(index => observation.shapValues[index]);
  $('.waterfall-total').innerHTML = `<span class="waterfall-total-label">Baseline + contributions = prediction</span><span class="waterfall-total-equation">${inlineMath(numericBody(number(observation.baseValue)))}${additions.map(value => inlineMath(`<mo>${value < 0 ? '−' : '+'}</mo>${numericBody(number(Math.abs(value)))}`)).join('')}${inlineMath(`<mo>=</mo>${numericBody(number(observation.prediction))}`)}</span><span class="waterfall-total-unit">$1,000/year</span>`;
}

function renderPopulation() {
  let container = $('#aggregate-container');
  if (!container) {
    visual.innerHTML = '<div id="aggregate-container"></div>';
    container = $('#aggregate-container');
  }
  renderAggregate(container, {
    scene:steps[state.scene].id, progress:state.aggregateProgress, backgroundSize:state.data.background.length,
    features:state.data.features, observation:state.observation, population:state.population,
    featureOrder:state.aggregateOrder, reducedMotion:reducedMotion.matches,
    onSelect:person => openPopulationWaterfall(person, $('#inspect')),
  });
}

function updatePopulationSummary() {
  const person = state.globalObservation;
  $('#population-summary').textContent = `Ability ${inputNumber(person.values[0])}; neighborhood ${inputNumber(person.values[1])}; experience ${inputNumber(person.values[2])}. Prediction: ${number(person.prediction)} ($1,000/year).`;
}

function openPopulationWaterfall(person = state.globalObservation, trigger = $('#population-waterfall')) {
  state.globalObservation = person;
  $('#population-select').value = person.id;
  updatePopulationSummary();
  state.rowsTrigger = trigger;
  $('#rows-heading').textContent = 'One person in the beeswarm';
  $('#rows-description').textContent = `Ability ${inputNumber(person.values[0])}, neighborhood opportunity ${inputNumber(person.values[1])}, experience ${inputNumber(person.values[2])}. The same model and reference background give a baseline of ${number(person.baseValue)} and prediction of ${number(person.prediction)} ($1,000/year).`;
  $('#rows-table').innerHTML = '<div class="dialog-waterfall"></div>';
  dialog.showModal();
  renderWaterfall($('.dialog-waterfall'), {features:state.data.features, observation:person, unit:'$1,000/year'});
  $('#close-rows').focus();
}

function renderCoalitionExplorer() {
  $('#coalition-table').innerHTML = inputTable(state.explorerMask);
  $('#coalition-value').innerHTML = `${valueMarkup(state.explorerMask)} ${inlineMath(`<mo>=</mo>${numericBody(number(coalition(state.observation, state.explorerMask).value))}`)} ($1,000/year), for ${escape(state.observation.name.toLowerCase())}.`;
}

function updateFigureOverflow() {
  figureScrollHint.hidden = true;
  const scrollable = (['worked','chart'].includes(story.dataset.layout) || ['background-average','feature-credit','ols-fits','ols-refit','ols-recap'].includes(shell.dataset.scene)) && visual.scrollHeight > visual.clientHeight + 1;
  figureScrollHint.hidden = !scrollable;
  visual.classList.toggle('has-overflow', scrollable);
  if (scrollable) visual.setAttribute('aria-describedby', figureScrollHint.id);
  else visual.removeAttribute('aria-describedby');
}

function render() {
  if (!state.data || !state.breadData) return;
  const active = document.activeElement;
  const focusSelector = visual.contains(active) && active.id ? `#${active.id}` : null;
  updateFacts();
  const scene = steps[state.scene].id;
  const aggregateScene = aggregateScenes.has(scene);
  shell.dataset.scene = scene;
  shell.classList.toggle('is-aggregate', aggregateScene);
  shell.classList.toggle('is-chart', chartScenes.has(scene));
  shell.classList.toggle('is-worked', readableScenes.has(scene) || ['ols-refit','ols-recap'].includes(scene));
  story.classList.add('is-interactive');
  story.dataset.layout = chartScenes.has(scene) ? 'chart' : isCenteredScene(scene) ? 'formula' : 'worked';
  const breadScene = scene.startsWith('ols-');
  const selectorContext = breadScene ? 'bread' : 'earnings';
  const hideCenteredSelector = isCenteredScene(scene) && !readableScenes.has(scene) && !['ols-refit','ols-recap'].includes(scene);
  const hideCaption = isCenteredScene(scene);
  $('.observation-control').hidden = aggregateScene || hideCenteredSelector || ['ols-model','ols-fits'].includes(scene);
  if (state.selectorContext !== selectorContext) {
    const data = breadScene ? state.breadData : state.data;
    const observation = breadScene ? state.breadObservation : state.observation;
    observationSelect.innerHTML = data.observations.map(item => `<option value="${escape(item.id)}">${escape(item.name)}</option>`).join('');
    observationSelect.value = observation.id;
    $('label[for="observation-select"]').textContent = breadScene ? 'Election to explain' : 'Person to explain';
    state.selectorContext = selectorContext;
  }
  if (state.nameScene !== state.scene) {
    $('#stage-name').innerHTML = names[state.scene];
    $('#stage-count').textContent = `${state.scene+1} / ${steps.length}`;
    state.nameScene = state.scene;
  }
  $('#stage-caption').hidden = hideCaption;
  $('#stage-caption').innerHTML = hideCaption ? '' : captions[state.scene];
  $('#previous').disabled = state.scene === 0;
  $('#next').textContent = state.scene === steps.length-1 ? 'Notes →' : 'Next →';
  if (formulaStops[scene]) renderFormula(scene);
  else if (transitionScenes.has(scene)) renderTransition(scene);
  else if (breadScene) visual.innerHTML = renderBreadPeace(scene,state.breadData,state.breadObservation);
  else if (scene === 'observation') renderObservation();
  else if (scene === 'background') renderBackground(0);
  else if (scene === 'reveal-ability') renderBackground(1, 0);
  else if (scene === 'neighborhood-first') renderBackground(3, 2);
  else if (scene === 'weights') renderWeights();
  else if (scene === 'shap-value') renderAverage();
  else if (aggregateScene) renderPopulation();
  else renderFinal();
  const ordersScene = scene === 'weights' || scene === 'shap-value';
  $('#inspect').hidden = !['background','reveal-ability','neighborhood-first','weights','shap-value','feature-color','importance-ranking'].includes(scene);
  $('#inspect').textContent = aggregateScene ? 'Waterfall' : ordersScene ? 'Six orders' : 'Inspect rows';
  $('#inspect').setAttribute('aria-label', aggregateScene ? 'Inspect one person’s waterfall' : ordersScene ? 'Inspect the six revealing orders' : 'Inspect hybrid input rows and predictions');
  if (focusSelector) visual.querySelector(focusSelector)?.focus({preventScroll:true});
  visual.scrollTop = 0;
  requestAnimationFrame(updateFigureOverflow);
}

function openRows(trigger) {
  state.rowsTrigger = trigger;
  const scene = steps[state.scene].id;
  const mask = scene === 'background' ? 0 : scene === 'reveal-ability' ? 1 : 3;
  const beforeMask = scene === 'reveal-ability' ? 0 : scene === 'neighborhood-first' ? 2 : undefined;
  $('#rows-heading').textContent = `Knowing ${groupName(mask)}`;
  const valueEquation = coalitionMask => `${valueMarkup(coalitionMask)} ${inlineMath(`<mo>=</mo>${numericBody(number(coalition(state.observation, coalitionMask).value))}`)}`;
  $('#rows-description').innerHTML = `Included columns use the selected person’s values in every row. Excluded columns keep that reference row’s values. ${beforeMask === undefined ? `${valueEquation(mask)}.` : `Before: ${valueEquation(beforeMask)}; after: ${valueEquation(mask)}. Their difference is ${signed(marginal(state.observation, 0, beforeMask))}.`} Outputs are in $1,000/year.`;
  $('#rows-table').innerHTML = inputTable(mask, {beforeMask,newFeature:beforeMask === undefined ? -1 : 0});
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
function chartReadingLine() {
  const headerHeight = $('.masthead').getBoundingClientRect().height;
  return stackedLayout.matches ? Math.min(innerHeight - 30, headerHeight + (innerHeight - headerHeight) * .55) : innerHeight * .5;
}
function navigate(index, behavior = 'instant') {
  const target = index === steps.length ? $('#method') : steps[index];
  if (index < steps.length && chartScenes.has(target.id)) {
    state.scene = index;
    state.aggregateProgress = 0;
    render();
    scrollTo({top:Math.ceil(scrollY + target.getBoundingClientRect().top - chartReadingLine()),behavior:'instant'});
    history.replaceState(null,'',`#${target.id}`);
    return;
  }
  const centeredTransition = isCenteredScene(steps[state.scene]?.id) || isCenteredScene(target.id);
  if (centeredTransition) behavior = 'instant';
  if (index < steps.length && (centeredTransition || stackedLayout.matches)) {
    state.scene = index;
    state.aggregateProgress = 0;
    render();
  }
  if (index < steps.length && isCenteredScene(target.id)) {
    const readingOffset = $('.masthead').getBoundingClientRect().height;
    const top = scrollY + target.getBoundingClientRect().top - readingOffset;
    scrollTo({top,behavior});
  } else if (index < steps.length && stackedLayout.matches) {
    const readingOffset = $('.masthead').getBoundingClientRect().height + shell.getBoundingClientRect().height + 15;
    const top = scrollY + target.getBoundingClientRect().top - readingOffset;
    scrollTo({top,behavior});
  } else target.scrollIntoView({behavior,block:'start'});
  history.replaceState(null,'',`#${target.id}`);
}
let scrollPending = false;
function updateScroll() {
  scrollPending = false;
  if (!state.data || !state.breadData) return;
  const mobile = stackedLayout.matches;
  // Keep the phone reading line consistent when the full-width intro becomes the OLS panel.
  const workedIntroHeight = Math.min(650, Math.max(440, innerHeight * .74));
  const introReadingLine = Math.min(innerHeight - 30, $('.masthead').getBoundingClientRect().height + workedIntroHeight + 60);
  let readingLine = mobile
    ? (state.scene < 0 || isCenteredScene(steps[state.scene]?.id) ? introReadingLine : shell.getBoundingClientRect().bottom + 60)
    : innerHeight * .5;
  // Keep the activation line stable as the OLS fits expand to the full-width
  // stage and the following worked example returns to the compact panel.
  if (mobile && steps[state.scene]?.id.startsWith('ols-')) readingLine = introReadingLine;
  // A stable phone reading line prevents the chart's compact height from changing
  // the active scene as it crosses the local-to-global boundary.
  const firstAggregate = $('#contribution-axis');
  if (mobile && (chartScenes.has(steps[state.scene]?.id) || firstAggregate.getBoundingClientRect().top < innerHeight)) {
    readingLine = chartReadingLine();
  }
  let scene = 0;
  steps.forEach((step,index) => {if (step.getBoundingClientRect().top <= readingLine) scene = index;});
  const activeStep = steps[scene];
  const span = steps[scene+1] ? steps[scene+1].offsetTop - activeStep.offsetTop : activeStep.offsetHeight;
  // Leave a reading hold after each transformation, while keeping geometry
  // controlled by native scroll position in both directions.
  state.aggregateProgress = Math.max(0, Math.min(1, (readingLine - activeStep.getBoundingClientRect().top) / (span * .62)));
  if (state.scene !== scene) {state.scene = scene; render();}
  else if (aggregateScenes.has(activeStep.id)) renderPopulation();
  const progress = Math.max(0,Math.min(1,(scrollY-steps[0].offsetTop)/($('#method').offsetTop-steps[0].offsetTop)));
  $('#progress').style.width = `${progress*100}%`;
}
function queueScroll() {if (!scrollPending) {scrollPending = true; requestAnimationFrame(updateScroll);}}

$('#inspect').addEventListener('click', () => aggregateScenes.has(steps[state.scene].id) ? openPopulationWaterfall(state.globalObservation, $('#inspect')) : ['weights','shap-value'].includes(steps[state.scene].id) ? openOrders($('#inspect')) : openRows($('#inspect')));
$('#previous').addEventListener('click', () => navigate(Math.max(0,state.scene-1)));
$('#next').addEventListener('click', () => navigate(state.scene+1));
$('.title-page-start').addEventListener('click', event => {
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || !state.data || !state.breadData) return;
  event.preventDefault();
  navigate(0, 'instant');
  visual.focus({preventScroll:true});
});
$('#close-rows').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});
dialog.addEventListener('close', () => {if (state.rowsTrigger?.isConnected) state.rowsTrigger.focus({preventScroll:true});});
observationSelect.addEventListener('change', () => {
  if (state.selectorContext === 'bread') state.breadObservation = state.breadData.observations.find(item => item.id === observationSelect.value);
  else state.observation = state.data.observations.find(item => item.id === observationSelect.value);
  state.aggregateOrder = state.data.features.map((_,index) => index).sort((a,b) => Math.abs(state.observation.shapValues[b])-Math.abs(state.observation.shapValues[a]) || a-b);
  render();
});
$('#population-select').addEventListener('change', event => {
  state.globalObservation = state.population.find(person => person.id === event.target.value);
  updatePopulationSummary();
});
$('#population-waterfall').addEventListener('click', () => openPopulationWaterfall());
$('#coalition-controls').addEventListener('change', () => {
  state.explorerMask = [...$('#coalition-controls').querySelectorAll('input:checked')].reduce((mask,input) => mask | (1 << Number(input.value)),0);
  renderCoalitionExplorer();
});
addEventListener('scroll',queueScroll,{passive:true});
addEventListener('resize', () => {
  if (aggregateScenes.has(steps[state.scene]?.id)) {
    const step = steps[state.scene];
    const span = steps[state.scene+1] ? steps[state.scene+1].offsetTop-step.offsetTop : step.offsetHeight;
    const line = chartReadingLine();
    scrollTo({top:Math.ceil(scrollY+step.getBoundingClientRect().top-line+state.aggregateProgress*span*.62),behavior:'instant'});
  }
  queueScroll();
  if (steps[state.scene]?.id === 'waterfall') renderFinal();
  if (dialog.open && $('.dialog-waterfall')) renderWaterfall($('.dialog-waterfall'), {features:state.data.features,observation:state.globalObservation,unit:'$1,000/year'});
  requestAnimationFrame(updateFigureOverflow);
});
reducedMotion.addEventListener('change', () => {if (aggregateScenes.has(steps[state.scene]?.id)) renderPopulation();});
addEventListener('hashchange', () => {
  if (!state.data || !state.breadData) return;
  const index = steps.findIndex(step => step.id === canonicalSceneId(location.hash.slice(1)));
  // Let native hash navigation finish its frame before placing the reading line.
  if (index >= 0) requestAnimationFrame(() => navigate(index, 'instant'));
});

try {
  const responses = await Promise.all(['data.json','bread-peace.json','model.json'].map(file => fetch(new URL(file,import.meta.url))));
  for (const response of responses) if (!response.ok) throw new Error(`Data request failed (${response.status}).`);
  const [earningsData,breadData,model] = await Promise.all(responses.map(response => response.json()));
  state.data = validateData(earningsData);
  state.breadData = validateBreadPeace(breadData);
  state.breadObservation = state.breadData.observations.find(item => item.id === state.breadData.defaultObservationId) || state.breadData.observations[0];
  state.observation = state.data.observations.find(item => item.id === state.data.defaultObservationId) || state.data.observations[0];
  state.population = buildPopulation(state.data,model);
  state.globalObservation = state.population.find(person => person.id === state.observation.id) || state.population[0];
  state.aggregateOrder = state.data.features.map((_,index) => index).sort((a,b) => Math.abs(state.observation.shapValues[b])-Math.abs(state.observation.shapValues[a]) || a-b);
  $('#population-select').innerHTML = state.population.map((person,index) => `<option value="${escape(person.id)}">${index+1} · ability ${inputNumber(person.values[0])}, neighborhood ${inputNumber(person.values[1])}, experience ${inputNumber(person.values[2])}</option>`).join('');
  $('#population-select').value = state.globalObservation.id;
  updatePopulationSummary();
  observationSelect.innerHTML = state.data.observations.map(item => `<option value="${escape(item.id)}">${escape(item.name)}</option>`).join('');
  observationSelect.value = state.observation.id;
  observationSelect.disabled = false;
  updateScroll();
  const anchor = location.hash && document.getElementById(canonicalSceneId(location.hash.slice(1)));
  if (anchor) {
    const index = steps.indexOf(anchor);
    if (index >= 0) navigate(index, 'instant');
    else anchor.scrollIntoView({behavior:'instant',block:'start'});
    queueScroll();
  }
} catch (error) {
  visual.innerHTML = '<p>The interactive data could not be loaded. The explanation and notes below remain available.</p>';
  $('#stage-caption').textContent = error.message;
  $('#next').disabled = true;
  console.error(error);
}
