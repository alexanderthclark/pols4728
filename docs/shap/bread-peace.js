import { coalition, mean, marginal } from './calculation.mjs';
import { valueMarkup } from './formula.js';

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
    ['Between the two features', '<mi>ρ</mi><mo>=</mo><msub><mi>r</mi><mrow><mtext>bread</mtext><mo>,</mo><mtext>peace</mtext></mrow></msub>', data.refit.featureCorrelation],
    ['Bread and observed vote', '<msub><mi>r</mi><mrow><mtext>bread</mtext><mo>,</mo><mi>y</mi></mrow></msub>', data.refit.breadVoteCorrelation],
    ['Peace and observed vote', '<msub><mi>r</mi><mrow><mtext>peace</mtext><mo>,</mo><mi>y</mi></mrow></msub>', data.refit.peaceVoteCorrelation],
  ];
  return `<h3>Start with three correlations</h3><p class="bread-context">Bread, peace, and observed vote share y are centered and standardized: mean 0, SD 1.</p><div class="bread-correlations">${correlations.map(([label,symbol,value]) => `<section><p>${label}</p>${math(`${symbol}<mo>=</mo>${numeral(value)}`,`${label}: correlation ${format(value)}`)}</section>`).join('')}</div><p class="input-note">These are stipulated correlations for a stylized Bread and Peace example.</p>`;
}

function renderFits(data) {
  return `<h3>The three OLS fits</h3><div class="bread-fits">${[
    ['Bread-only OLS','bread',data.refit.breadOnlyCoefficient],
    ['Peace-only OLS','peace',data.refit.peaceOnlyCoefficient],
  ].map(([label,feature,coefficient]) => `<section><h4>${label}</h4>${math(`<msub>${yHat}<mtext>${feature}</mtext></msub><mo>=</mo>${numeral(coefficient)}<mo>×</mo><mtext>${feature}</mtext>`,`${label}: y hat equals ${format(coefficient)} times ${feature}`)}</section>`).join('')}<section class="bread-full-fit"><h4>Bivariate OLS · the model we explain</h4>${modelEquation(data)}</section></div><p class="input-note">All intercepts are zero. The univariate slopes equal the feature–vote correlations; the bivariate slopes adjust for the feature correlation.</p>`;
}

function focalInputs(observation) {
  return `<p class="bread-context">This election: bread = ${format(observation.values[0])}, peace = ${format(observation.values[1])}. Inputs to ŷ are written in that order.</p>`;
}

export function breadPeaceMeanPredictionMarkup(observation) {
  const bread = numeral(observation.values[0]);
  const predictionWithPeace = `${yHat}${parentheses(`${bread}<mo>,</mo><mtext>peace</mtext>`)}`;
  const breadValue = `<msub><mi>v</mi><mi>x</mi></msub>${parentheses('<mo stretchy="false">{</mo><mtext>bread</mtext><mo stretchy="false">}</mo>')}`;
  return `<span class="bread-mean-bridge"><span class="bread-mean-equation" role="math" aria-label="v sub x of bread is the average of y hat of ${format(observation.values[0])}, peace over background peace values, which equals y hat of ${format(observation.values[0])}, zero"><math aria-hidden="true"><mrow>${breadValue}<mo>=</mo><munder><mtext>average</mtext><mtext>background peace</mtext></munder>${predictionWithPeace}</mrow></math><math aria-hidden="true"><mrow><mo>=</mo>${prediction(point(observation,1))}</mrow></math></span></span>`;
}

function renderRefit(data, observation) {
  const fixed = coalition(observation,1).value;
  return `<h3>Which prediction does SHAP use?</h3>${focalInputs(observation)}<p class="bread-focal-prediction">${math(`${prediction(observation.values)}<mo>=</mo>${numeral(observation.prediction)}`,`The bivariate prediction for this election is ${format(observation.prediction)}`)}</p><div class="refit-comparison"><section class="refit-unused"><h4>Bread-only OLS</h4><p>Use the univariate coefficient, 0.75.</p><p class="refit-equation">0.75 × (${format(observation.values[0])})</p><div class="refit-result">${format(observation.reducedPrediction)}<small>Reduced model · not used for SHAP</small></div></section><section><h4>Bivariate OLS; peace averaged</h4><p>Keep both coefficients; average background peace.</p><p class="refit-equation">${math(prediction(point(observation,1)), 'The original bivariate model with this election’s bread held fixed and peace at its background mean.')}</p><div class="refit-result">${format(fixed)}<small>${valueMarkup(1,notation)}<br>The value in our SHAP game</small></div></section></div>`;
}

function renderDifference(observation, feature, beforeMask) {
  const afterMask = beforeMask | (1 << feature);
  const before = coalition(observation,beforeMask).value;
  const after = coalition(observation,afterMask).value;
  const contribution = after-before;
  const beforePoint = point(observation,beforeMask), afterPoint = point(observation,afterMask);
  const label = feature === 0 ? 'Bread' : 'Peace';
  const group = beforeMask ? `{${labels[1-feature]}}` : '∅';
  return `<section class="bread-order-step"><h4>${label} joins ${group}</h4><div class="bread-difference" role="math" aria-label="${label} marginal: y hat of ${afterPoint.join(', ')} minus y hat of ${beforePoint.join(', ')} equals ${format(after)} minus ${format(before)} equals ${signed(contribution)}"><math aria-hidden="true"><mrow>${prediction(afterPoint)}<mo>−</mo>${prediction(beforePoint)}</mrow></math><math aria-hidden="true" class="${contribution < 0 ? 'negative' : ''}"><mrow><mo>=</mo>${numeral(after)}<mo>−</mo>${before < 0 ? parentheses(numeral(before)) : numeral(before)}<mo>=</mo>${signedNumeral(contribution)}</mrow></math></div></section>`;
}

function renderOrder(data, observation, order) {
  const first = order[0], second = order[1];
  return `<h3>${first === 0 ? 'Bread → Peace' : 'Peace → Bread'}</h3>${focalInputs(observation)}<p class="bread-model-label">Use the same bivariate equation for every prediction:</p>${modelEquation(data)}<p class="bread-baseline">${math(`${valueMarkupTextEmpty()}<mo>=</mo>${prediction([0,0])}<mo>=</mo><mn>0</mn>`,'The empty coalition value is y hat of zero, zero, which is zero')}</p><div class="bread-order">${renderDifference(observation,first,0)}${renderDifference(observation,second,1 << first)}</div>`;
}

function valueMarkupTextEmpty() {
  return `<msub><mi>v</mi><mi>x</mi></msub>${parentheses('<mo lspace="0" rspace="0">∅</mo>')}`;
}

function averageEquation(observation,feature) {
  const first = marginal(observation,feature,0);
  const second = marginal(observation,feature,1 << (1-feature));
  const result = observation.shapValues[feature];
  const half = '<mfrac><mn>1</mn><mn>2</mn></mfrac>';
  return `<section class="bread-order-step"><h4>${feature === 0 ? 'Bread' : 'Peace'}</h4><div class="bread-average-equation" role="math" aria-label="The SHAP value for ${labels[feature]} equals one half times ${signed(first)} plus one half times ${signed(second)} equals ${signed(result)}"><math aria-hidden="true"><mrow><msub><mi>ϕ</mi><mtext>${labels[feature]}</mtext></msub>${parentheses('<mi>x</mi>')}<mo>=</mo>${half}<mo>×</mo>${parentheses(signedNumeral(first))}</mrow></math><math aria-hidden="true"><mrow><mo>+</mo>${half}<mo>×</mo>${parentheses(signedNumeral(second))}<mo>=</mo>${signedNumeral(result)}</mrow></math></div></section>`;
}

export function breadPeaceDecompositionMarkup() {
  const credit = feature => `<msub><mi>ϕ</mi><mtext>${feature}</mtext></msub>${parentheses('<mi>x</mi>')}`;
  return `<span class="bread-decomposition-identity">${math(`${yHat}${parentheses('<mi>x</mi>')}<mo>=</mo>${prediction([0,0])}<mo>+</mo>${credit('bread')}<mo>+</mo>${credit('peace')}`, 'y hat of x equals y hat of zero, zero plus the SHAP contribution for bread plus the SHAP contribution for peace')}</span>`;
}

function renderShap(data, observation) {
  return `<h3>Average the two orders</h3>${focalInputs(observation)}<div class="bread-order-averages">${averageEquation(observation,0)}${averageEquation(observation,1)}</div><div class="bread-decomposition" role="group" aria-label="Baseline plus both SHAP contributions equals the predicted standardized vote"><span>Baseline<strong>0</strong></span><span>Bread<strong class="${observation.shapValues[0] < 0 ? 'negative' : ''}">${signed(observation.shapValues[0])}</strong></span><span>Peace<strong class="${observation.shapValues[1] < 0 ? 'negative' : ''}">${signed(observation.shapValues[1])}</strong></span><span>Prediction<strong>${format(observation.prediction)}</strong></span></div>`;
}

export function renderBreadPeace(scene, data, observation) {
  if (scene === 'ols-model') return renderCorrelations(data);
  if (scene === 'ols-fits') return renderFits(data);
  if (scene === 'ols-refit') return renderRefit(data,observation);
  if (scene === 'ols-income-first') return renderOrder(data,observation,[0,1]);
  if (scene === 'ols-fatalities-first') return renderOrder(data,observation,[1,0]);
  if (scene === 'ols-shap') return renderShap(data,observation);
  throw new RangeError(`Unknown OLS scene: ${scene}`);
}
