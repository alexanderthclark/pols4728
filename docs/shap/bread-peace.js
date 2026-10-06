import { coalition, hybridValues, mean, marginal } from './calculation.mjs';
import { valueMarkup, joiningMarkup } from './formula.js';

const labels = ['income', 'fatalities'];
const notation = {labels, names:['real income growth', 'war fatalities']};
const yHat = '<mover accent="true"><mi>y</mi><mo>^</mo></mover>';
const format = value => new Intl.NumberFormat('en-US', {maximumFractionDigits:3}).format(Math.abs(value) < 1e-10 ? 0 : value).replace('-', '−');
const signed = value => Math.abs(value) < 1e-10 ? '0' : `${value > 0 ? '+' : '−'}${format(Math.abs(value))}`;
const setName = mask => mask ? `{${labels.filter((_, index) => mask & (1 << index)).join(', ')}}` : '∅';
const math = (content, label) => `<math aria-label="${label}"><mrow>${content}</mrow></math>`;

export function validateBreadPeace(data) {
  if (data.features?.length !== 2 || data.background?.length !== 8 || !data.observations?.length) throw new Error('The OLS warm-up needs two features and eight rows.');
  const columns = [0,1].map(index => data.background.map(row => row.values[index]));
  columns.push(data.background.map(row => row.outcome));
  for (const column of columns) {
    if (Math.abs(mean(column)) > 1e-9 || Math.abs(mean(column.map(value => value ** 2)) - 1) > 1e-9) throw new Error('The OLS columns must have mean zero and SD one.');
  }
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
    income:format(observation.values[0]), fatalities:format(observation.values[1]),
    prediction:format(observation.prediction), reducedPrediction:format(observation.reducedPrediction),
    incomeValue:format(coalition(observation,1).value), fatalitiesValue:format(coalition(observation,2).value),
    incomeMarginal:signed(marginal(observation,0,0)), incomeShap:signed(observation.shapValues[0]),
    fatalitiesShap:signed(observation.shapValues[1]),
  };
}

export function breadPeaceInputTable(data, observation, mask, beforeMask) {
  const group = coalition(observation,mask);
  const before = beforeMask === undefined ? null : coalition(observation,beforeMask);
  return `<table class="input-table"><caption>All 8 equally weighted reference rows · vote predictions in SD</caption><thead><tr><th scope="col">Row</th>${data.features.map((feature,index) => `<th scope="col">${feature.shortLabel}<small>${mask & (1 << index) ? 'Included<br>fixed to election' : 'Excluded<br>from background'}</small></th>`).join('')}${before ? `<th scope="col">${math(yHat, 'y hat')}<small>Before</small></th><th scope="col">${math(yHat, 'y hat')}<small>After</small></th>` : `<th scope="col">${math(yHat, 'y hat')}<small>Prediction</small></th>`}</tr></thead><tbody>${data.background.map((row,rowIndex) => {
    const hybrid = hybridValues(row.values, observation.values,mask);
    return `<tr><td>${rowIndex+1}</td>${hybrid.map((value,index) => {
      const fixed = mask & (1 << index);
      const newlyFixed = before && fixed && !(beforeMask & (1 << index));
      const changed = newlyFixed && value !== row.values[index];
      return `<td class="${fixed ? newlyFixed ? 'newly-fixed' : 'fixed' : 'unknown'}">${changed ? `<span class="cell-original">${format(row.values[index])}</span><span class="replacement-arrow" aria-label="replaced by"> → </span>` : ''}<span class="cell-value">${format(value)}</span></td>`;
    }).join('')}${before ? `<td>${format(before.predictions[rowIndex])}</td>` : ''}<td>${format(group.predictions[rowIndex])}</td></tr>`;
  }).join('')}</tbody><tfoot><tr><th scope="row" colspan="3">Average of all 8</th>${before ? `<td>${format(before.value)}</td>` : ''}<td>${format(group.value)}</td></tr></tfoot></table>`;
}

function modelEquation() {
  return `<div class="prediction-equation bread-equation" role="math" aria-label="y hat of x equals 0.5 times income minus 0.5 times fatalities"><math aria-hidden="true"><mrow>${yHat}<mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn><mo>×</mo><mtext>income</mtext></mrow></math><math aria-hidden="true"><mrow><mo>−</mo><mn>0.5</mn><mo>×</mo><mtext>fatalities</mtext></mrow></math></div>`;
}

function trainingTable(data) {
  return `<table class="input-table bread-training"><caption>8 synthetic training rows · income, fatalities, and observed y standardized</caption><thead><tr><th scope="col">Row</th><th scope="col">Income</th><th scope="col">Fatalities</th><th scope="col"><var>y</var><small>Outcome</small></th><th scope="col">${math(yHat,'y hat')}<small>Prediction</small></th></tr></thead><tbody>${data.background.map((row,index) => `<tr><td>${index+1}</td><td>${format(row.values[0])}</td><td>${format(row.values[1])}</td><td>${format(row.outcome)}</td><td>${format(row.modelPrediction)}</td></tr>`).join('')}</tbody></table>`;
}

function renderRefit(data, observation) {
  const fixed = coalition(observation,1).value;
  return `<h3>Exclude fatalities: two calculations</h3><div class="refit-comparison"><section><h4>Refit income-only OLS</h4><p>Income coefficient becomes <strong>0.75</strong>.</p><p class="refit-equation">0.5 + (−0.5) × (−0.5) = 0.75</p><p class="refit-note">Original coefficient + correlation × omitted coefficient</p><div class="refit-result">${format(observation.reducedPrediction)}<small>Reduced-model prediction<br>0.75 × income</small></div></section><section><h4>Average full-model predictions</h4><p>Income coefficient stays <strong>0.5</strong>.</p><p class="refit-equation">0.5 × income − 0.5 × 0</p><p class="refit-note">Background fatalities average to zero.</p><div class="refit-result">${format(fixed)}<small>${valueMarkup(1,notation)}<br>Original model, averaged</small></div></section></div><p class="input-note">The SHAP calculation uses the original fitted model throughout. The left column is a comparison with a different model.</p>`;
}

function renderRows(data, observation, mask, beforeMask) {
  const before = beforeMask === undefined ? null : coalition(observation,beforeMask);
  const group = coalition(observation,mask);
  return `${before ? joiningMarkup(beforeMask,0,notation) : `<h3 class="coalition-heading">${valueMarkup(mask,notation)}</h3>`}<p class="formula-table-context">${before ? 'S = ∅ · i = income · after − before' : 'S = ∅ · no columns fixed to this election'}</p><p class="focal-values"><var>x</var>: income = ${format(observation.values[0])} · fatalities = ${format(observation.values[1])}</p>${breadPeaceInputTable(data,observation,mask,beforeMask)}<div class="table-marginal"><span>${before ? 'Income’s marginal contribution' : 'Baseline: average prediction'}</span><span class="${before && group.value - before.value < 0 ? 'negative' : ''}">${before ? `${format(group.value)} − ${format(before.value)} = ${signed(group.value-before.value)}` : format(group.value)}</span></div><p class="input-note">${before ? 'Income comes from x. Fatalities still come from each background row.' : 'These are model outputs ŷ, not observed outcomes y or prediction errors.'}</p>`;
}

function renderShap(data, observation) {
  const contribution = observation.shapValues[0];
  return `<h3>Four groups; two revealing orders</h3><table class="weights-table bread-coalitions"><caption>Prediction averages for this election</caption><thead><tr><th scope="col">Fixed feature group S</th><th scope="col">v<sub>x</sub>(S)</th></tr></thead><tbody>${observation.coalitions.map(group => `<tr><th scope="row">${setName(group.mask)}</th><td>${format(group.value)}</td></tr>`).join('')}</tbody></table><table class="weights-table bread-orders"><caption>Income’s marginal in each order · weight 1/2</caption><thead><tr><th scope="col">Order</th><th scope="col">Before → After</th><th scope="col">Marginal</th></tr></thead><tbody>${[0,2].map((mask,index) => `<tr><th scope="row">${index === 0 ? 'Income → Fatalities' : 'Fatalities → Income'}</th><td>${format(coalition(observation,mask).value)} → ${format(coalition(observation,mask|1).value)}</td><td class="${contribution < 0 ? 'negative' : ''}">${signed(marginal(observation,0,mask))}</td></tr>`).join('')}</tbody></table><p class="bread-average">ϕ<sub>income</sub>(x) = (1/2) × (${signed(contribution)}) + (1/2) × (${signed(contribution)}) = ${signed(contribution)}</p><div class="bread-decomposition" role="group" aria-label="Baseline plus both SHAP contributions equals the predicted standardized vote"><span>Baseline<strong>0</strong></span><span>Income<strong class="${observation.shapValues[0] < 0 ? 'negative' : ''}">${signed(observation.shapValues[0])}</strong></span><span>Fatalities<strong class="${observation.shapValues[1] < 0 ? 'negative' : ''}">${signed(observation.shapValues[1])}</strong></span><span>Prediction<strong>${format(observation.prediction)}</strong></span></div>`;
}

export function renderBreadPeace(scene, data, observation) {
  if (scene === 'ols-model') return `<h3>Stylized Bread and Peace · fit OLS once</h3>${modelEquation()}<p class="bread-correlation">Feature correlation ρ = −0.5 · inputs and observed y: mean 0, SD 1</p>${trainingTable(data)}<p class="input-note">Income growth and war fatalities follow Hibbs’s model. These rows, outcomes, and coefficients are constructed for teaching.</p>`;
  if (scene === 'ols-prediction') return `<h3>Explain one election’s prediction</h3>${modelEquation()}<div class="observation-card"><span class="date">Observation x · ${observation.name}</span><dl><dt>Income growth</dt><dd>${format(observation.values[0])} SD</dd><dt>War fatalities</dt><dd>${format(observation.values[1])} SD</dd></dl><div class="prediction-number">${format(observation.prediction)}<span class="prediction-label">ŷ(x) · SD of incumbent-party vote share</span></div></div><p class="input-note">A negative standardized fatality value means fewer fatalities than the reference average.</p>`;
  if (scene === 'ols-refit') return renderRefit(data,observation);
  if (scene === 'ols-background') return renderRows(data,observation,0);
  if (scene === 'ols-income') return renderRows(data,observation,1,0);
  if (scene === 'ols-shap') return renderShap(data,observation);
  throw new RangeError(`Unknown OLS scene: ${scene}`);
}
