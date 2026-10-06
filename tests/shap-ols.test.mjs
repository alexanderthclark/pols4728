import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

const data = JSON.parse(await readFile(new URL('../docs/shap/bread-peace.json',import.meta.url),'utf8'));
const tolerance = 1e-9;
const mean = values => values.reduce((sum,value) => sum + value,0) / values.length;
const covariance = (a,b) => mean(a.map((value,index) => (value-mean(a))*(b[index]-mean(b))));
const close = (actual,expected,label) => assert.ok(Number.isFinite(actual) && Math.abs(actual-expected) < tolerance,`${label}: ${actual} != ${expected}`);
const bread = data.background.map(row => row.values[0]);
const peace = data.background.map(row => row.values[1]);
const outcomes = data.background.map(row => row.outcome);
const varianceBread = covariance(bread,bread);
const variancePeace = covariance(peace,peace);
const cross = covariance(bread,peace);
const breadOutcome = covariance(bread,outcomes);
const peaceOutcome = covariance(peace,outcomes);
// Solve the two OLS normal equations independently of Python and saved coefficients.
const determinant = varianceBread*variancePeace - cross**2;
const fitted = [
  (breadOutcome*variancePeace - peaceOutcome*cross)/determinant,
  (peaceOutcome*varianceBread - breadOutcome*cross)/determinant,
];
const predict = values => fitted[0]*values[0] + fitted[1]*values[1];

test('synthetic bread, peace, and observed vote are standardized and yield the declared full OLS fit', () => {
  assert.equal(data.model.fitted,true);
  assert.match(data.dataSource,/Synthetic/);
  assert.equal(data.background.length,8);
  assert.equal(new Set(data.background.map(row => row.id)).size,8);
  for (const [index,column] of [bread,peace,outcomes].entries()) {
    close(mean(column),0,`column ${index} mean`);
    close(covariance(column,column),1,`column ${index} variance`);
  }
  close(cross,.5,'feature correlation');
  close(data.model.intercept,0,'centered intercept');
  fitted.forEach((coefficient,index) => {
    close(coefficient,[.5,.5][index],`full OLS slope ${index}`);
    close(data.model.coefficients[index],coefficient,`stored OLS slope ${index}`);
  });
  const residuals = outcomes.map((value,index) => value-predict(data.background[index].values));
  close(mean(residuals),0,'OLS residual mean');
  close(covariance(residuals,bread),0,'OLS residual orthogonal to bread');
  close(covariance(residuals,peace),0,'OLS residual orthogonal to peace');
  close(covariance(residuals,residuals),.25,'remaining outcome variance');
  assert.ok(residuals.some(value => Math.abs(value) > .1),'Observed y and fitted y hat must be distinct');
  data.background.forEach(row => close(row.modelPrediction,predict(row.values),`${row.id} fitted prediction`));
});

test('correlations recover three OLS fits while SHAP keeps the bivariate coefficients fixed', () => {
  const reducedSlope = breadOutcome/varianceBread;
  const correlation = cross/Math.sqrt(varianceBread*variancePeace);
  const outcomeCorrelation = breadOutcome/Math.sqrt(varianceBread*covariance(outcomes,outcomes));
  close(reducedSlope,.75,'reduced OLS slope');
  close(reducedSlope,outcomeCorrelation,'univariate standardized slope equals correlation');
  close(reducedSlope,fitted[0]+correlation*fitted[1],'omitted-variable identity');
  close(data.refit.breadOnlyCoefficient,reducedSlope,'saved reduced slope');
  close(data.refit.breadVoteCorrelation,outcomeCorrelation,'saved vote-bread correlation');
  const peaceCorrelation = peaceOutcome/Math.sqrt(variancePeace*covariance(outcomes,outcomes));
  close(peaceCorrelation,.75,'vote-peace correlation');
  close(data.refit.peaceVoteCorrelation,peaceCorrelation,'saved vote-peace correlation');
  close(data.refit.peaceOnlyCoefficient,peaceOutcome/variancePeace,'peace-only OLS slope');
  close(peaceCorrelation,fitted[1]+correlation*fitted[0],'peace omitted-variable identity');
  const fromCorrelations = [
    (outcomeCorrelation-correlation*peaceCorrelation)/(1-correlation**2),
    (peaceCorrelation-correlation*outcomeCorrelation)/(1-correlation**2),
  ];
  fromCorrelations.forEach((coefficient,index) => close(coefficient,fitted[index],`correlations recover bivariate slope ${index}`));
  close(data.refit.featureCorrelation,correlation,'saved feature correlation');
  close(data.refit.omittedVariableTerm,.25,'saved omitted-variable term');
  for (const observation of data.observations) {
    const fixedModelAverage = mean(data.background.map(row => predict([observation.values[0],row.values[1]])));
    close(fixedModelAverage,.5*observation.values[0],`${observation.id} mean substitution`);
    close(observation.coalitions[1].value,fixedModelAverage,`${observation.id} bread-only coalition`);
    close(observation.reducedPrediction,reducedSlope*observation.values[0],`${observation.id} refitted prediction`);
    close(observation.reducedPrediction-fixedModelAverage,.25*observation.values[0],`${observation.id} two operations differ`);
  }
});

test('all two-feature coalitions and orders recover centered linear SHAP and its prediction decomposition', () => {
  for (const observation of data.observations) {
    const groups = Array.from({length:4},(_,mask) => {
      const predictions = data.background.map(row => predict(row.values.map((value,index) => mask & (1 << index) ? observation.values[index] : value)));
      return {value:mean(predictions),predictions};
    });
    assert.deepEqual(observation.coalitions.map(group => group.mask),[0,1,2,3]);
    observation.coalitions.forEach((group,mask) => {
      close(group.value,groups[mask].value,`${observation.id} group ${mask}`);
      assert.equal(group.predictions.length,8);
      group.predictions.forEach((value,index) => close(value,groups[mask].predictions[index],`${observation.id} group ${mask} row ${index}`));
    });
    close(observation.baseValue,0,`${observation.id} centered baseline`);
    close(observation.prediction,predict(observation.values),`${observation.id} full prediction`);
    assert.deepEqual(new Set(observation.orders.map(order => order.features.join(','))),new Set(['0,1','1,0']));
    for (let feature=0;feature<2;feature++) {
      const otherMask = 1 << (1-feature);
      const firstMarginal = groups[1 << feature].value - groups[0].value;
      const secondMarginal = groups[3].value - groups[otherMask].value;
      close(firstMarginal,secondMarginal,`${observation.id} additive marginal invariant`);
      close(observation.shapValues[feature],.5*firstMarginal+.5*secondMarginal,`${observation.id} two-order average`);
      close(observation.shapValues[feature],fitted[feature]*observation.values[feature],`${observation.id} centered closed form`);
    }
    for (const order of observation.orders) {
      let mask=0;
      order.features.forEach((feature,index) => {
        const after = mask | (1 << feature);
        close(order.marginals[index],groups[after].value-groups[mask].value,`${observation.id} order marginal`);
        mask=after;
      });
      close(order.marginals.reduce((sum,value) => sum+value,0),observation.prediction,`${observation.id} order telescopes`);
    }
    close(observation.shapValues.reduce((sum,value) => sum+value,0),observation.prediction,`${observation.id} SHAP reconstruction`);
  }
  close(data.validation.maxShapAgreementError,0,'official exact SHAP agreement');
  close(data.validation.maxBaselineAgreementError,0,'official SHAP baseline agreement');
});
