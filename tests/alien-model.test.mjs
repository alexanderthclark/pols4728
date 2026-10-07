import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fitRule,predict,summarize,simulate,illustration} from '../docs/alien/model.mjs';
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-12,`${actual} ≠ ${expected}`);
test('five-game illustration fits all training labels but misclassifies a one-point loss',()=>{
 const rule=fitRule(illustration);assert.equal(rule.cutoff,-1.5);
 for(const [k,s] of illustration)assert.equal(predict(rule,k-s),Number(k>s));
 assert.equal(predict(rule,-1),1);assert.equal(predict(rule,-1.5),1);
});
test('one-class samples use infinite cutoffs and predict their observed class everywhere',()=>{
 const wins=fitRule([[82.5,48],[86.5,54],[70.5,15],[66.5,5],[71.5,17]]),losses=fitRule([[1,2],[0,3]]);
 assert.equal(wins.cutoff,-Infinity);assert.equal(losses.cutoff,Infinity);
 assert.equal(wins.differences.length,5);assert.ok(wins.differences.every(x=>x>0));
 assert.equal(predict(wins,-100),1);assert.equal(predict(losses,100),0);
});
test('the published simulation reproduces the notes and its MSE decomposition',()=>{
 const fixture=JSON.parse(readFileSync(new URL('../docs/alien/samples.json',import.meta.url)));
 assert.equal(fixture.games.length,1000);
 const result=summarize(fixture.games.map(fitRule),0.5);
 assert.equal(result.wins,532);assert.equal(result.losses,468);
 near(result.mean,.532);near(result.bias,-.468);near(result.biasSquared,.219024);near(result.variance,.248976);near(result.mse,.468);
 near(result.mse,result.biasSquared+result.variance+result.noise);
});
test('decomposition also holds at a loss input and in new sample sizes',()=>{
 for(const n of [5,10,25,100,500]){
 const rules=simulate(1000,n,45);
 for(const x of [-20.5,-.5,.5,20.5]){
 const r=summarize(rules,x);near(r.mse,r.biasSquared+r.variance);assert.equal(r.noise,0);assert.equal(r.truth,Number(x>0));
 }
 }
});
test('synthetic scores use the stated support and produce no ties',()=>{
 const rules=simulate(100,5,77);assert.deepEqual(rules,simulate(100,5,77));
 for(const {games} of rules)for(const [k,s]of games){assert.ok(k>=.5&&k<=100.5);assert.ok(s>=0&&s<=100);assert.equal(k%1,.5);assert.equal(s%1,0);assert.notEqual(k,s);}
});
