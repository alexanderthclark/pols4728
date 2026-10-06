import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

const data = JSON.parse(await readFile(new URL('../docs/shap/bread-peace.json',import.meta.url),'utf8'));
const tolerance = 1e-9;
const mean = values => values.reduce((sum,value) => sum + value,0) / values.length;
const covariance = (a,b) => mean(a.map((value,index) => (value-mean(a))*(b[index]-mean(b))));
const close = (actual,expected,label) => assert.ok(Number.isFinite(actual) && Math.abs(actual-expected) < tolerance,`${label}: ${actual} != ${expected}`);
const income = data.background.map(row => row.values[0]);
const fatalities = data.background.map(row => row.values[1]);
const outcomes = data.background.map(row => row.outcome);
const varianceIncome = covariance(income,income);
const varianceFatalities = covariance(fatalities,fatalities);
const cross = covariance(income,fatalities);
const incomeOutcome = covariance(income,outcomes);
const fatalitiesOutcome = covariance(fatalities,outcomes);
// Solve the two OLS normal equations independently of Python and saved coefficients.
const determinant = varianceIncome*varianceFatalities - cross**2;
const fitted = [
  (incomeOutcome*varianceFatalities - fatalitiesOutcome*cross)/determinant,
  (fatalitiesOutcome*varianceIncome - incomeOutcome*cross)/determinant,
];
const predict = values => fitted[0]*values[0] + fitted[1]*values[1];

test('synthetic income, fatalities, and observed vote are standardized and yield the declared full OLS fit', () => {
  assert.equal(data.model.fitted,true);
  assert.match(data.dataSource,/Synthetic/);
  assert.equal(data.background.length,8);
  assert.equal(new Set(data.background.map(row => row.id)).size,8);
  for (const [index,column] of [income,fatalities,outcomes].entries()) {
    close(mean(column),0,`column ${index} mean`);
    close(covariance(column,column),1,`column ${index} variance`);
  }
  close(cross,-.5,'feature correlation');
  close(data.model.intercept,0,'centered intercept');
  fitted.forEach((coefficient,index) => {
    close(coefficient,[.5,-.5][index],`full OLS slope ${index}`);
    close(data.model.coefficients[index],coefficient,`stored OLS slope ${index}`);
  });
  const residuals = outcomes.map((value,index) => value-predict(data.background[index].values));
  close(mean(residuals),0,'OLS residual mean');
  close(covariance(residuals,income),0,'OLS residual orthogonal to income');
  close(covariance(residuals,fatalities),0,'OLS residual orthogonal to fatalities');
  close(covariance(residuals,residuals),.25,'remaining outcome variance');
  assert.ok(residuals.some(value => Math.abs(value) > .1),'Observed y and fitted y hat must be distinct');
  data.background.forEach(row => close(row.modelPrediction,predict(row.values),`${row.id} fitted prediction`));
});

test('correlations recover three OLS fits while SHAP keeps the bivariate coefficients fixed', () => {
  const reducedSlope = incomeOutcome/varianceIncome;
  const correlation = cross/Math.sqrt(varianceIncome*varianceFatalities);
  const outcomeCorrelation = incomeOutcome/Math.sqrt(varianceIncome*covariance(outcomes,outcomes));
  close(reducedSlope,.75,'reduced OLS slope');
  close(reducedSlope,outcomeCorrelation,'univariate standardized slope equals correlation');
  close(reducedSlope,fitted[0]+correlation*fitted[1],'omitted-variable identity');
  close(data.refit.incomeOnlyCoefficient,reducedSlope,'saved reduced slope');
  close(data.refit.incomeVoteCorrelation,outcomeCorrelation,'saved vote-income correlation');
  const fatalitiesCorrelation = fatalitiesOutcome/Math.sqrt(varianceFatalities*covariance(outcomes,outcomes));
  close(fatalitiesCorrelation,-.75,'vote-fatalities correlation');
  close(data.refit.fatalitiesVoteCorrelation,fatalitiesCorrelation,'saved vote-fatalities correlation');
  close(data.refit.fatalitiesOnlyCoefficient,fatalitiesOutcome/varianceFatalities,'fatalities-only OLS slope');
  close(fatalitiesCorrelation,fitted[1]+correlation*fitted[0],'fatalities omitted-variable identity');
  const fromCorrelations = [
    (outcomeCorrelation-correlation*fatalitiesCorrelation)/(1-correlation**2),
    (fatalitiesCorrelation-correlation*outcomeCorrelation)/(1-correlation**2),
  ];
  fromCorrelations.forEach((coefficient,index) => close(coefficient,fitted[index],`correlations recover bivariate slope ${index}`));
  close(data.refit.featureCorrelation,correlation,'saved feature correlation');
  close(data.refit.omittedVariableTerm,.25,'saved omitted-variable term');
  for (const observation of data.observations) {
    const fixedModelAverage = mean(data.background.map(row => predict([observation.values[0],row.values[1]])));
    close(fixedModelAverage,.5*observation.values[0],`${observation.id} mean substitution`);
    close(observation.coalitions[1].value,fixedModelAverage,`${observation.id} income-only coalition`);
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
