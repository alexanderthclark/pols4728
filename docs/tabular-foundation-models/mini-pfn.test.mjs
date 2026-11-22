import test from 'node:test';
import assert from 'node:assert/strict';
import {miniPfnResults as run} from './mini-pfn-results.mjs';
import {toyState} from './bayes.mjs';
import {renderMechanism,recordedTrainingTasks} from './mechanism.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<2e-5,`${a} != ${b}`);
const dot=(x,y)=>x.reduce((sum,v,i)=>sum+v*y[i],0);
const linear=(x,name)=>run.parameters[`${name}.weight`].map((w,i)=>dot(w,x)+run.parameters[`${name}.bias`][i]);
const mlp=(x,name)=>linear(linear(x,`${name}.0`).map(v=>Math.max(0,v)),`${name}.2`);

test('saved weights independently reproduce both context-dependent attention traces',()=>{
  assert.equal(run.metadata.is_tabpfn,false);
  assert.equal(run.metadata.training_status,'completed');
  assert.deepEqual(run.examples.A.query,run.examples.B.query);
  for(const example of Object.values(run.examples)){
    assert.equal(example.state_dict_sha256,run.metadata.state_dict_sha256);
    const h=example.context.map(row=>mlp(row,'context_encoder'));
    const query=mlp(example.query,'query_encoder');
    const q=linear(query,'to_q'), keys=h.map(row=>linear(row,'to_k')), values=h.map(row=>linear(row,'to_v'));
    const scores=keys.map(k=>dot(q,k)/Math.sqrt(run.metadata.width));
    const max=Math.max(...scores),exp=scores.map(s=>Math.exp(s-max)),sum=exp.reduce((a,b)=>a+b,0),attention=exp.map(v=>v/sum);
    attention.forEach((v,i)=>near(v,example.attention[i]));
    const pooled=q.map((_,j)=>attention.reduce((s,a,i)=>s+a*values[i][j],0));
    pooled.forEach((v,j)=>near(v,example.pooled[j]));
    assert.equal(query.length,32);
    assert.equal(pooled.length,32);
    assert.equal([...query,...pooled].length,64);
    const logit=mlp([...query,...pooled],'decoder')[0];
    near(logit,example.logit);
    near(1/(1+Math.exp(-logit)),example.probability);
  }
  assert.ok(run.examples.A.probability>.5);
  assert.ok(run.examples.B.probability<.5);
});

test('training snapshots use the recorded first update, not the final checkpoint',()=>{
  const u=run.first_update;
  assert.deepEqual(recordedTrainingTasks.first.context,u.context);
  assert.deepEqual(recordedTrainingTasks.first.query,u.query);
  assert.equal(recordedTrainingTasks.first.query_outcome,u.query_outcome);
  const attr=(svg,name)=>Number(svg.match(new RegExp(`${name}="([^"]+)"`))[1]);
  for(const mobile of [false,true]){
    const before=renderMechanism(2,0,mobile),after=renderMechanism(2,1,mobile);
    near(attr(before,'data-query-probability'),u.first_query_probability_before);
    near(attr(after,'data-query-probability'),u.first_query_probability_after);
    near(attr(before,'data-batch-log-loss'),u.batch_log_loss_before);
    near(attr(after,'data-batch-log-loss'),u.batch_log_loss_after);
    assert.equal(attr(before,'data-batch-size'),256);
    assert.equal(attr(after,'data-batch-size'),256);
    assert.notEqual(attr(before,'data-query-probability'),run.examples.A.probability);
    assert.notEqual(attr(after,'data-query-probability'),run.examples.A.probability);
    // At intermediate scroll positions only measured states may be reported.
    for(const progress of [.1,.3,.5,.7,.9]){
      const svg=renderMechanism(2,progress,mobile);
      assert.ok([u.first_query_probability_before,u.first_query_probability_after].includes(attr(svg,'data-query-probability')));
    }
    assert.equal(renderMechanism(2,0,mobile),before,'Backward scrolling restores the pre-update snapshot.');
  }
});

test('teaching outputs remain distinct from their exact Bayesian targets',()=>{
  near(run.examples.A.exact_bayes_probability,toyState(4).prediction);
  near(run.examples.B.exact_bayes_probability,1-toyState(4).prediction);
  near(run.evaluation.ignore_context_log_loss,Math.log(2));
  assert.ok(run.evaluation.model_log_loss<run.evaluation.ignore_context_log_loss);
  assert.notEqual(run.examples.A.probability,run.examples.A.exact_bayes_probability);
});
