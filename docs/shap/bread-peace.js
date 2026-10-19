import { coalition, mean, marginal } from './calculation.mjs';
import { backgroundAverageMarkup, valueMarkup } from './formula.js';

const labels = ['bread', 'peace'];
const notation = {labels, names:['bread', 'peace']};
const yHat = '<mover accent="true"><mi>y</mi><mo>^</mo></mover>';
const format = value => new Intl.NumberFormat('en-US', {maximumFractionDigits:3}).format(Math.abs(value) < 1e-10 ? 0 : value).replace('-', '−');
const signed = value => Math.abs(value) < 1e-10 ? '0' : `${value > 0 ? '+' : '−'}${format(Math.abs(value))}`;
const math = (content, label) => `<math aria-label="${label}"><mrow>${content}</mrow></math>`;
const numeral = value => `${value < 0 ? '<mo>−</mo>' : ''}<mn>${format(Math.abs(value))}</mn>`;
const signedNumeral = value => `${value > 0 ? '<mo>+</mo>' : ''}${numeral(value)}`;
const parentheses = content => `<mo stretchy="false">(</mo>${content}<mo stretchy="false">)</mo>`;
const prediction = values => `${yHat}${parentheses(values.map(numeral).join('<mo>,</mo>'))}`;
const point = (observation,mask) => observation.values.map((value,index) => mask & (1 << index) ? value : 0);

export function validateBreadPeace(data) {
  if (data.features?.length !== 2 || data.background?.length !== 8 || !data.observations?.length) throw new Error('The OLS warm-up needs two features and eight reference rows.');
  const columns = [0,1].map(index => data.background.map(row => row.values[index]));
  columns.push(data.background.map(row => row.outcome));
  for (const column of columns) {
    if (Math.abs(mean(column)) > 1e-9 || Math.abs(mean(column.map(value => value ** 2)) - 1) > 1e-9) throw new Error('The OLS columns must have mean zero and SD one.');
  }
  const {featureCorrelation:rho, breadVoteCorrelation:bread, peaceVoteCorrelation:peace} = data.refit;
  const slopes = [(bread-rho*peace)/(1-rho**2), (peace-rho*bread)/(1-rho**2)];
  if (slopes.some((slope,index) => !Number.isFinite(slope) || Math.abs(slope-data.model.coefficients[index]) > 1e-9)) throw new Error('The three correlations must recover the fitted OLS slopes.');
  for (const observation of data.observations) {
    if (observation.coalitions.length !== 4 || observation.orders.length !== 2) throw new Error('Missing OLS coalition or order calculations.');
    for (const group of observation.coalitions) {
      if (group.predictions.length !== 8 || Math.abs(mean(group.predictions) - group.value) > 1e-9) throw new Error('An OLS coalition average differs from its predictions.');
    }
    if (Math.abs(observation.baseValue + observation.shapValues.reduce((sum,value) => sum + value,0) - observation.prediction) > 1e-9) throw new Error('The OLS explanation does not reconstruct the prediction.');
  }
  return data;
}

export function breadPeaceFacts(data, observation) {
  return {
    electionName:observation.name.toLowerCase(),
    bread:format(observation.values[0]), peace:format(observation.values[1]),
    prediction:format(observation.prediction), reducedPrediction:format(observation.reducedPrediction),
    breadValue:format(coalition(observation,1).value), peaceValue:format(coalition(observation,2).value),
    breadMarginal:signed(marginal(observation,0,0)), breadLast:signed(marginal(observation,0,2)),
    peaceFirst:signed(marginal(observation,1,0)), peaceLast:signed(marginal(observation,1,1)),
    breadShap:signed(observation.shapValues[0]), peaceShap:signed(observation.shapValues[1]),
  };
}

function modelEquation(data) {
  const [bread,peace] = data.model.coefficients;
  return `<div class="prediction-equation bread-equation" role="math" aria-label="y hat of x equals ${format(bread)} times bread ${signed(peace)} times peace"><math aria-hidden="true"><mrow>${yHat}${parentheses('<mi>x</mi>')}<mo>=</mo>${numeral(bread)}<mo>×</mo><mtext>bread</mtext></mrow></math><math aria-hidden="true"><mrow>${signedNumeral(peace)}<mo>×</mo><mtext>peace</mtext></mrow></math></div>`;
}

function renderCorrelations(data) {
  const correlations = [
    ['Between the two features', '<mi>ρ</mi><mo>=</mo><msub><mi>r</mi><mrow><mtext class="bread-predictor">bread</mtext><mo>,</mo><mtext class="bread-predictor">peace</mtext></mrow></msub>', data.refit.featureCorrelation],
    ['Bread and observed vote', '<msub><mi>r</mi><mrow><mtext class="bread-predictor">bread</mtext><mo>,</mo><mi class="bread-outcome">y</mi></mrow></msub>', data.refit.breadVoteCorrelation],
    ['Peace and observed vote', '<msub><mi>r</mi><mrow><mtext class="bread-predictor">peace</mtext><mo>,</mo><mi class="bread-outcome">y</mi></mrow></msub>', data.refit.peaceVoteCorrelation],
  ];
  return `<h3>Start with three correlations (stylized Hibbs model)</h3><p class="bread-context"><span class="bread-predictor">Bread, peace</span>, and <span class="bread-outcome">observed vote share ${math('<mi class="bread-outcome">y</mi>', 'y')}</span> are centered and standardized: mean 0, SD 1.</p><div class="bread-correlations">${correlations.map(([label,symbol,value]) => `<section><p>${label}</p>${math(`${symbol}<mo>=</mo>${numeral(value)}`,`${label}: correlation ${format(value)}`)}</section>`).join('')}</div><p class="input-note">These are stipulated correlations for a stylized Bread and Peace example.</p>`;
}

function coalitionPredictionEquation(mask, observation) {
  const inputs = point(observation,mask);
  const result = coalition(observation,mask).value;
  return `<div class="bread-coalition-equation" role="math" aria-label="v sub x of ${mask === 3 ? 'bread and peace' : 'bread'} equals y hat of ${inputs.map(format).join(', ')} equals ${format(result)}"><span aria-hidden="true">${valueMarkup(mask,notation)}</span><span aria-hidden="true">${math(`<mo>=</mo>${prediction(inputs)}`, 'model prediction')}</span><span aria-hidden="true">${math(`<mo>=</mo>${numeral(result)}`,format(result))}</span></div>`;
}

function renderFullPrediction(data, observation) {
  return `<h3>Both features are fixed</h3><div class="bread-full-model">${modelEquation(data)}</div>${focalInputs(observation)}${coalitionPredictionEquation(3,observation)}<p class="refit-instruction">Use this election’s bread and peace values in the fitted two-feature model.</p><p class="bread-prediction-unit">Predictions: SD of vote share.</p>`;
}

function focalInputs(observation) {
  return `<p class="bread-context">This election: ${math(`<mtext>bread</mtext><mo>=</mo>${numeral(observation.values[0])}`, `bread equals ${format(observation.values[0])}`)}, ${math(`<mtext>peace</mtext><mo>=</mo>${numeral(observation.values[1])}`, `peace equals ${format(observation.values[1])}`)}. Inputs to ${math(yHat, 'y hat')} are written in that order.</p>`;
}

export function breadPeaceMeanPredictionMarkup(observation) {
  const bread = numeral(observation.values[0]);
  const predictionWithPeace = `${yHat}${parentheses(`${bread}<mo>,</mo><mtext>peace</mtext>`)}`;
  const breadValue = `<msub><mi>v</mi><mi>x</mi></msub>${parentheses('<mo stretchy="false">{</mo><mtext>bread</mtext><mo stretchy="false">}</mo>')}`;
  return `<span class="bread-mean-bridge"><span class="bread-mean-equation" role="math" aria-label="v sub x of bread is the average of y hat of ${format(observation.values[0])}, peace over background peace values, which equals y hat of ${format(observation.values[0])}, zero"><math aria-hidden="true"><mrow>${breadValue}<mo>=</mo></mrow></math><math aria-hidden="true"><mrow><munder><mtext>average</mtext><mtext>background peace</mtext></munder>${predictionWithPeace}</mrow></math><math aria-hidden="true"><mrow><mo>=</mo>${prediction(point(observation,1))}</mrow></math></span></span>`;
}

function renderRefit(data, observation, complete = false) {
  const bread = observation.values[0];
  const reduced = data.refit.breadOnlyCoefficient;
  const reducedY = `<msub>${yHat}<mtext>bread</mtext></msub>`;
  return `<h3>What if only bread is fixed?</h3><p class="bread-fixed-input">${valueMarkup(1,notation)} = ? &nbsp; Bread in this election: ${math(numeral(bread),format(bread))}.</p><div class="refit-transform${complete ? ' refit-complete' : ''}"><section class="refit-candidate${complete ? ' is-hidden' : ''}" aria-hidden="${complete}"><h4>A bread-only refit changes the model</h4><div class="refit-equation">${math(`${reducedY}<mo>=</mo>${numeral(reduced)}<mo>×</mo><mtext>bread</mtext>`, `The bread-only model has slope ${format(reduced)}`)}</div><div class="refit-result">${math(`${reducedY}${parentheses(numeral(bread))}<mo>=</mo>${numeral(observation.reducedPrediction)}`, `The refitted model predicts ${format(observation.reducedPrediction)}`)}</div><p class="refit-warning">Not used for SHAP. We do not refit the model.</p></section><section class="refit-correct${complete ? '' : ' is-hidden'}" aria-hidden="${!complete}"><h4>Keep the model; average over peace</h4>${modelEquation(data)}<p class="refit-instruction">Hold bread fixed; predict with all eight background peace values, then average.</p>${breadPeaceMeanPredictionMarkup(observation)}<p class="refit-note">This linear model lets us use mean peace, 0:</p>${coalitionPredictionEquation(1,observation)}<p class="bread-prediction-unit">Predictions: SD of vote share.</p></section></div>`;
}

const clamp = value => Math.max(0, Math.min(1,value));

export function updateBreadPeaceProgress(container, scene, progress, reducedMotion = false) {
  const transform = container.querySelector('.refit-transform');
  if (!transform) return;
  const p = scene === 'ols-bread-value' ? 1 : reducedMotion ? (progress < .5 ? 0 : 1) : clamp(progress);
  const gray = clamp((p-.12)/.22);
  const shrink = clamp((p-.3)/.3);
  const outgoing = 1-clamp((p-.4)/.2);
  const incoming = clamp((p-.66)/.22);
  const channel = Math.round(36+60*gray);
  transform.style.setProperty('--refit-color',`rgb(${channel}, ${channel}, ${channel})`);
  transform.style.setProperty('--refit-scale',String(1-.28*shrink));
  transform.style.setProperty('--refit-opacity',String(outgoing));
  transform.style.setProperty('--correct-opacity',String(incoming));
  transform.dataset.progress = String(p);
  transform.dataset.phase = p < .3 ? 'refit' : p < .66 ? 'dismiss' : 'average';
  const candidate = transform.querySelector('.refit-candidate');
  const correct = transform.querySelector('.refit-correct');
  candidate.classList.toggle('is-hidden',outgoing === 0);
  correct.classList.toggle('is-hidden',incoming === 0);
  candidate.setAttribute('aria-hidden',String(p >= .66));
  correct.setAttribute('aria-hidden',String(p < .66));
}

function renderDifference(observation, feature, beforeMask) {
  const afterMask = beforeMask | (1 << feature);
  const before = coalition(observation,beforeMask).value;
  const after = coalition(observation,afterMask).value;
  const contribution = after-before;
  const beforePoint = point(observation,beforeMask), afterPoint = point(observation,afterMask);
  const label = feature === 0 ? 'Bread' : 'Peace';
  const group = beforeMask
    ? math(`<mo stretchy="false">{</mo><mtext>${labels[1-feature]}</mtext><mo stretchy="false">}</mo>`, `the set containing ${labels[1-feature]}`)
    : math('<mo>∅</mo>', 'the empty set');
  return `<section class="bread-order-step"><h4>${label} joins ${group}</h4><div class="bread-difference" role="math" aria-label="${label} marginal: y hat of ${afterPoint.join(', ')} minus y hat of ${beforePoint.join(', ')} equals ${format(after)} minus ${format(before)} equals ${signed(contribution)}"><math aria-hidden="true"><mrow>${prediction(afterPoint)}<mo>−</mo>${prediction(beforePoint)}</mrow></math><math aria-hidden="true" class="${contribution < 0 ? 'negative' : ''}"><mrow><mo>=</mo>${numeral(after)}<mo>−</mo>${before < 0 ? parentheses(numeral(before)) : numeral(before)}<mo>=</mo>${signedNumeral(contribution)}</mrow></math></div></section>`;
}

function renderOrder(data, observation, order) {
  const first = order[0], second = order[1];
  return `<h3>${first === 0 ? 'Bread → Peace' : 'Peace → Bread'}</h3>${focalInputs(observation)}<p class="bread-model-label">Same linear model; unrevealed inputs use their background mean, 0:</p>${modelEquation(data)}<p class="bread-baseline">${math(`${valueMarkupTextEmpty()}<mo>=</mo>${prediction([0,0])}<mo>=</mo><mn>0</mn>`,'The empty coalition value is y hat of zero, zero, which is zero')}</p><div class="bread-order">${renderDifference(observation,first,0)}${renderDifference(observation,second,1 << first)}</div>`;
}

function valueMarkupTextEmpty() {
  return `<msub><mi>v</mi><mi>x</mi></msub>${parentheses('<mo lspace="0" rspace="0">∅</mo>')}`;
}

function averageEquation(observation,feature) {
  const first = marginal(observation,feature,0);
  const second = marginal(observation,feature,1 << (1-feature));
  const result = observation.shapValues[feature];
  const half = '<mfrac><mn>1</mn><mn>2</mn></mfrac>';
  return `<section class="bread-order-step"><h4>${feature === 0 ? 'Bread' : 'Peace'}</h4><div class="bread-average-equation" role="math" aria-label="The SHAP value for ${labels[feature]} equals one half times ${signed(first)} plus one half times ${signed(second)} equals ${signed(result)}"><math aria-hidden="true"><mrow><msub><mi>ϕ</mi><mtext>${labels[feature]}</mtext></msub>${parentheses('<msub><mi>v</mi><mi>x</mi></msub>')}<mo>=</mo>${half}<mo>×</mo>${parentheses(signedNumeral(first))}</mrow></math><math aria-hidden="true"><mrow><mo>+</mo>${half}<mo>×</mo>${parentheses(signedNumeral(second))}<mo>=</mo>${signedNumeral(result)}</mrow></math></div></section>`;
}

export function breadPeaceDecompositionMarkup() {
  const credit = feature => `<msub><mi>ϕ</mi><mtext>${feature}</mtext></msub>${parentheses('<msub><mi>v</mi><mi>x</mi></msub>')}`;
  return `<span class="bread-decomposition-identity" role="math" aria-label="y hat of x equals y hat of zero, zero plus the SHAP contribution for bread plus the SHAP contribution for peace"><math aria-hidden="true"><mrow>${yHat}${parentheses('<mi>x</mi>')}<mo>=</mo>${prediction([0,0])}</mrow></math><math aria-hidden="true"><mrow><mo>+</mo>${credit('bread')}</mrow></math><math aria-hidden="true"><mrow><mo>+</mo>${credit('peace')}</mrow></math></span>`;
}

function renderShap(data, observation) {
  return `<h3>Average the two orders</h3>${focalInputs(observation)}<div class="bread-order-averages">${averageEquation(observation,0)}${averageEquation(observation,1)}</div><div class="bread-decomposition" role="group" aria-label="Baseline plus both SHAP contributions equals the predicted standardized vote"><span>Baseline<strong>0</strong></span><span>Bread<strong class="${observation.shapValues[0] < 0 ? 'negative' : ''}">${signed(observation.shapValues[0])}</strong></span><span>Peace<strong class="${observation.shapValues[1] < 0 ? 'negative' : ''}">${signed(observation.shapValues[1])}</strong></span><span>Prediction<strong>${format(observation.prediction)}</strong></span></div><p class="input-note">Two orders, each with weight 1/2. Predictions are in SD of vote share.</p>${breadPeaceDecompositionMarkup()}`;
}

function renderRecap(data, observation) {
  const valueBody = `<msub><mi>v</mi><mi>x</mi></msub>${parentheses('<mi>S</mi>')}`;
  const valueOfS = math(valueBody,'v sub x of S');
  const included = '<msub><mi>x</mi><mi>S</mi></msub>';
  const excludedIndex = '<mrow><mo>−</mo><mi>S</mi></mrow>';
  const meanExcluded = `<msub><mover accent="true"><mi>x</mi><mo>¯</mo></mover>${excludedIndex}</msub>`;
  const meanPrediction = `${yHat}${parentheses(`${included}<mo>,</mo>${meanExcluded}`)}`;
  const equation = (body,label) => `<div class="recap-general-equation" role="math" aria-label="${label}"><math aria-hidden="true"><mrow>${valueBody}<mo>=</mo></mrow></math><math aria-hidden="true"><mrow>${body}</mrow></math></div>`;
  const means = labels.map((_,index) => mean(data.background.map(row => row.values[index])));
  const rows = [0,1,2,3].map(mask => {
    const inputs = observation.values.map((value,index) => mask & (1 << index) ? value : means[index]);
    const members = labels.filter((_,index) => mask & (1 << index));
    const group = members.length ? `<mo stretchy="false">{</mo>${members.map(label => `<mtext>${label}</mtext>`).join('<mo>,</mo>')}<mo stretchy="false">}</mo>` : '<mo>∅</mo>';
    const inputCells = inputs.map((value,index) => {
      const included = Boolean(mask & (1 << index));
      return `<td class="${included ? 'recap-fixed' : 'recap-mean'}"><span class="recap-input-name">${index === 0 ? 'Bread' : 'Peace'}</span>${math(numeral(value),`${labels[index]} equals ${format(value)}`)}<small>${included ? 'from this election' : 'background mean'}</small></td>`;
    }).join('');
    return `<tr><th scope="row">${math(group,`S contains ${members.length ? members.join(' and ') : 'no features'}`)}</th>${inputCells}<td class="recap-prediction">${math(`${prediction(inputs)}<mo>=</mo>${numeral(coalition(observation,mask).value)}`,`y hat of ${inputs.map(format).join(', ')} equals ${format(coalition(observation,mask).value)}`)}</td></tr>`;
  }).join('');
  return `<h3>What does ${valueOfS} mean?</h3><div class="recap-definitions"><div class="recap-general-definition"><p class="recap-rule">Any model: average predictions</p>${backgroundAverageMarkup()}<p class="recap-qualification">Keep included inputs at this observation’s values; use each background row for the other inputs.</p></div><div class="recap-generalization"><p class="recap-rule">Linear model: use mean excluded inputs</p>${equation(meanPrediction,'For a linear model, v sub x of S equals y hat evaluated at the observation’s included inputs and the overall background means of the excluded inputs.')}<p class="recap-legend">${math(meanExcluded,'x bar sub minus S')}: overall background means. This shortcut can fail for nonlinear models.</p></div></div><table class="coalition-recap-table"><caption>Bread and peace both have background mean 0.</caption><thead><tr><th scope="col">${math('<mi>S</mi>','S')}</th><th scope="col">Bread</th><th scope="col">Peace</th><th scope="col">${valueOfS}</th></tr></thead><tbody>${rows}</tbody></table>`;
}

export function renderBreadPeace(scene, data, observation) {
  if (scene === 'ols-model') return renderCorrelations(data);
  if (scene === 'ols-fits') return renderFullPrediction(data,observation);
  if (scene === 'ols-refit' || scene === 'ols-bread-value') return renderRefit(data,observation,scene === 'ols-bread-value');
  if (scene === 'ols-income-first') return renderOrder(data,observation,[0,1]);
  if (scene === 'ols-fatalities-first') return renderOrder(data,observation,[1,0]);
  if (scene === 'ols-shap') return renderShap(data,observation);
  if (scene === 'ols-recap') return renderRecap(data,observation);
  throw new RangeError(`Unknown OLS scene: ${scene}`);
}
