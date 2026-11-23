import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import katex from '../docs/attention/vendor/katex/katex.min.js';
import {formatWeight} from '../docs/attention/trained-scenes.mjs';
import {createGuidedTrainedScenes} from '../docs/attention/trained-tour-scenes.mjs';

const original=JSON.parse(readFileSync(new URL('../docs/attention/data/infinite-jest-attention.json',import.meta.url),'utf8'));
const investigation=JSON.parse(readFileSync(new URL('../docs/attention/data/infinite-jest-investigation.json',import.meta.url),'utf8'));
const near=(actual,expected,tolerance,message)=>assert.ok(Math.abs(actual-expected)<=tolerance,`${message}: ${actual} versus ${expected}`);
const clamp=(value,minimum=0,maximum=1)=>Math.max(minimum,Math.min(maximum,value));
const escapeText=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const drawing={
  C:{blue:'#234e70',teal:'#007f73',rust:'#a54f32',ink:'#242424',muted:'#606060',rule:'#dededb'},
  m:tex=>katex.renderToString(tex,{throwOnError:true,strict:'error',output:'htmlAndMathml'}),
  text:(value,...args)=>`<text data-args="${escapeText(args.join(','))}">${escapeText(value)}</text>`,
  line:(...args)=>`<line data-args="${escapeText(args.join(','))}"/>`,
  rect:(x,y,width,height,...args)=>`<rect x="${x}" y="${y}" width="${width}" height="${height}" data-args="${escapeText(args.join(','))}"/>`,
  circle:(...args)=>`<circle data-args="${escapeText(args.join(','))}"/>`,
  clamp,
  lerp:(from,to,phase)=>from+(to-from)*phase,
  ease:value=>{const phase=clamp(value);return phase*phase*(3-2*phase);},
};
const scoresIn=value=>{
  if(!value||typeof value!=='object')return [];
  if(Array.isArray(value.logits)&&'positive_probability' in value)return [value];
  return Object.values(value).flatMap(scoresIn);
};
const tokensRetained=tokens=>{
  assert.equal(tokens[0],'[CLS]');assert.equal(tokens.at(-1),'[SEP]');
  assert.ok(tokens.includes("'"),'tokenized contractions must retain the apostrophe');
};

test('the investigation uses the original checkpoint and independently reproduces its published results',()=>{
  assert.equal(investigation.schema_version,1);
  assert.equal(investigation.model.id,original.model.id);
  assert.equal(investigation.model.revision,original.model.revision);
  assert.equal(investigation.model.architecture,'DistilBertForSequenceClassification');
  assert.deepEqual(investigation.model.labels,['NEGATIVE','POSITIVE']);
  assert.equal(investigation.model.layers,6);assert.equal(investigation.model.heads,12);
  const settings=investigation.provenance.settings;
  assert.equal(settings.device,'cpu');assert.equal(settings.dtype,'float32');
  assert.equal(settings.model_mode,'eval');assert.equal(settings.attention_implementation,'eager');
  assert.equal(settings.padding,false);assert.equal(settings.truncation,false);
  assert.equal(investigation.provenance.checks.published_originals.length,2);
  original.examples.forEach((example,index)=>{
    const observed=investigation.behavior[index?'full':'negative'];
    assert.equal(observed.text,example.text);assert.deepEqual(observed.tokens,example.tokens);
    example.logits.forEach((logit,label)=>near(observed.score.logits[label],logit,1e-5,'reproduced original logit'));
    const check=investigation.provenance.checks.published_originals.find(item=>item.text===example.text);
    assert.ok(check&&check.max_logit_error<1e-5&&check.max_attention_error<1e-5);
  });
});

test('every reported prediction and probability is consistent with its measured logits',()=>{
  const scores=scoresIn(investigation);
  assert.ok(scores.length>50,'include validation outcomes as well as selected examples');
  for(const score of scores){
    assert.equal(score.logits.length,2);assert.ok(score.logits.every(Number.isFinite));
    const margin=score.logits[1]-score.logits[0];
    near(score.margin,margin,2e-6,'positive-minus-negative margin');
    near(score.positive_probability,1/(1+Math.exp(-margin)),2e-7,'probability from logits');
    assert.equal(score.prediction,margin>0?'POSITIVE':'NEGATIVE');
  }
});

test('the two selected reads address actual token positions in the complete measured matrix',()=>{
  const phrase=investigation.phrase,n=phrase.tokens.length;
  assert.equal(phrase.layer,2);assert.equal(phrase.head,9);
  assert.equal(phrase.text,investigation.behavior.short.text);
  assert.deepEqual(phrase.tokens,investigation.behavior.short.tokens);
  tokensRetained(phrase.tokens);assert.equal(n,11);
  assert.equal(phrase.attention.length,n);
  for(const row of phrase.attention){
    assert.equal(row.length,n);assert.ok(row.every(value=>Number.isFinite(value)&&value>=0&&value<=1));
    near(row.reduce((sum,value)=>sum+value,0),1,2e-6,'complete attention row');
  }
  assert.deepEqual(phrase.edges.map(edge=>[edge.destination_token,edge.source_token]),[['t','just'],['just','t']]);
  for(const edge of phrase.edges){
    assert.equal(phrase.tokens[edge.destination_index],edge.destination_token);
    assert.equal(phrase.tokens[edge.source_index],edge.source_token);
    near(edge.weight,phrase.attention[edge.destination_index][edge.source_index],1e-9,'selected source coefficient');
  }
  near(phrase.edges[0].weight,.7736082,1e-6,'t reads just');
  near(phrase.edges[1].weight,.8649251,1e-6,'just reads t');
  assert.equal(investigation.provenance.checks.removed_edges_zero,true);
  assert.equal(investigation.provenance.checks.intervened_row_sums_match_method,true);
  near(investigation.provenance.checks.all_ones_mask_max_logit_error,0,1e-6,'unmodified mask control');
});

test('the phrase intervention preserves both the short-review reversal and the full-review non-reversal',()=>{
  const {short,full}=investigation.phrase;
  assert.equal(short.before.prediction,'POSITIVE');assert.equal(short.after.prediction,'NEGATIVE');
  assert.equal(full.before.prediction,'POSITIVE');assert.equal(full.after.prediction,'POSITIVE');
  near(short.after.positive_probability,.100165,1e-5,'short review after message removal');
  near(full.after.positive_probability,.9994966,1e-6,'full review after the same intervention');
  for(const result of [short,full])near(result.delta_margin,result.after.margin-result.before.margin,1e-9,'intervention margin change');
  assert.equal(full.text,investigation.behavior.full.text);
  assert.equal(full.renormalized,false);
  const checks=investigation.validation.phrase;
  assert.equal(checks.cases.length,5);
  for(const [kind,count] of [['short','short_flips'],['full','full_flips'],['renormalized_short','renormalized_short_flips']]){
    const flips=checks.cases.filter(entry=>entry[kind].before.prediction!==entry[kind].after.prediction).length;
    assert.equal(flips,checks[count]);
    for(const entry of checks.cases){
      tokensRetained(entry[kind].tokens);
      assert.equal(entry[kind].renormalized,kind==='renormalized_short');
      assert.deepEqual(entry[kind].edges.map(edge=>[edge.destination_token,edge.source_token]),[['t','just'],['just','t']]);
    }
  }
  assert.equal(checks.short_flips,5);assert.equal(checks.full_flips,0);
});

test('the brightness comparison holds one head fixed and includes transformed-value magnitude',()=>{
  const brightness=investigation.brightness;
  assert.equal(brightness.text,original.examples[0].text);
  const [apostrophe,t]=brightness.messages;
  assert.equal(apostrophe.id,'apostrophe');assert.equal(t.id,'t');
  for(const message of brightness.messages){
    assert.equal(message.layer,6);assert.equal(message.head,10);
    assert.equal(message.destination_index,0);assert.equal(message.destination_token,'[CLS]');
    assert.equal(brightness.tokens[message.source_index],message.source_token);
    near(message.weight,original.examples[0].attentions[5][9][0][message.source_index],1e-9,'same measured head and destination');
    assert.ok(message.value_norm>=0&&message.projected_value_norm>=0);
    near(message.message_norm,message.weight*message.projected_value_norm,2e-6,'weighted projected-value norm');
    near(message.delta_margin,message.after.margin-brightness.baseline.margin,1e-9,'message removal effect');
  }
  assert.ok(apostrophe.weight/t.weight>20);
  assert.ok(Math.abs(apostrophe.message_norm/t.message_norm-1)<.05,'the two measured messages have similar magnitudes');
  near(apostrophe.delta_margin,.011164,1e-5,'apostrophe message removal');
});

test('state and value transfers use an aligned token and distinguish fixed layer-5 attention',()=>{
  const context=investigation.context;
  assert.equal(context.layer,5);assert.equal(context.state_after_block,4);assert.equal(context.source_token,'not');
  assert.equal(context.negative.tokens.length,context.positive.tokens.length);
  assert.equal(context.negative.not_index,context.positive.not_index);
  for(const example of [context.negative,context.positive])assert.equal(example.tokens[example.not_index],'not');
  assert.equal(context.directions.length,2);
  for(const transfer of context.directions){
    assert.notEqual(transfer.recipient,transfer.donor);
    assert.deepEqual(transfer.before,context[transfer.recipient].score);
    assert.equal(transfer.values_after.prediction,context[transfer.donor].score.prediction);
    assert.equal(transfer.state_after.prediction,context[transfer.donor].score.prediction);
    near(transfer.values_delta_margin,transfer.values_after.margin-transfer.before.margin,1e-9,'value transfer effect');
    near(transfer.state_delta_margin,transfer.state_after.margin-transfer.before.margin,1e-9,'complete-state transfer effect');
    near(transfer.attention_max_abs_change,0,1e-7,'layer-5 attention stays fixed during value replacement');
  }
  near(context.directions[0].values_after.margin,4.906172,1e-5,'positive donor values into negative recipient');
  near(context.directions[1].values_after.margin,-3.402029,1e-5,'negative donor values into positive recipient');
  for(const value of Object.values(context.controls))near(value,0,1e-6,'embedding, own-state, and final-token controls');
});

test('validation retains all eight additional pairs and the unsuccessful direction',()=>{
  const validation=investigation.validation.context;
  assert.equal(validation.pairs,8);assert.equal(validation.cases.length,8);
  assert.equal(new Set(validation.cases.map(entry=>entry.id)).size,8);
  const directions=validation.cases.flatMap(entry=>entry.directions);
  assert.equal(directions.length,validation.interventions);assert.equal(directions.length,16);
  for(const result of directions)assert.equal(result.flipped,result.before.prediction!==result.after.prediction);
  assert.equal(directions.filter(result=>result.flipped).length,validation.flips);
  assert.equal(validation.flips,15);
  assert.equal(directions.filter(result=>!result.flipped).length,1,'do not silently discard the failed validation direction');
});

test('all eight scenes render strict mathematics and restore measured states in both motion modes',()=>{
  const scenes=createGuidedTrainedScenes(drawing,original,investigation);
  assert.deepEqual(scenes.map(scene=>scene.id),[
    'trained-predictions','trained-phrase-test','trained-phrase-routing','trained-message-removal',
    'trained-context-restored','trained-brightness-test','trained-state-swap','trained-value-swap',
  ]);
  for(const scene of scenes){
    assert.equal(scene.discretePhases,true,'measured states must not be replaced by a generic interpolated scene phase');
    if(scene.equation)drawing.m(scene.equation);
    for(const reducedMotion of [false,true]){
      const phases=[0,.13,.3,.42,.58,.67,.82,1];
      const forward=phases.map(phase=>scene.draw(phase,{reducedMotion}));
      for(const svg of forward)assert.doesNotMatch(svg,/NaN|undefined|\[object Object\]/,scene.id);
      assert.ok(new Set(forward).size>1,`${scene.id} must progress as the page scrolls`);
      for(let index=phases.length-1;index>=0;index--){
        assert.equal(scene.draw(phases[index],{reducedMotion}),forward[index],`${scene.id} restores phase ${phases[index]}`);
      }
    }
  }
  const wrong=structuredClone(investigation);wrong.model.revision='different-checkpoint';
  assert.throws(()=>createGuidedTrainedScenes(drawing,original,wrong),/checkpoint/);
});

test('the matrix-to-message diagram preserves every measured source and destination cell',()=>{
  const scene=createGuidedTrainedScenes(drawing,original,investigation).find(scene=>scene.id==='trained-phrase-routing');
  const {tokens,attention}=investigation.phrase;
  for(const phase of [0,.4,1]){
    const svg=scene.draw(phase);
    assert.equal((svg.match(/data-attention-cell=/g)||[]).length,tokens.length**2);
    tokens.forEach((destination,i)=>tokens.forEach((source,j)=>{
      const title=`Layer 2 head 9. Destination ${i+1} ${destination}, source ${j+1} ${source}: ${formatWeight(attention[i][j])}`;
      assert.ok(svg.includes(`<title>${escapeText(title)}</title>`),title);
    }));
    assert.match(svg,/\[CLS\]/);assert.match(svg,/\[SEP\]/);
  }
});

test('scrolling only displays measured probabilities and attention percentages, never intermediate predictions',()=>{
  const {behavior,phrase,brightness,context}=investigation;
  const negativeTransfer=context.directions.find(direction=>direction.recipient==='negative');
  const scoreSets={
    'trained-predictions':[behavior.negative.score,behavior.full.score],
    'trained-phrase-test':[behavior.full.score,behavior.short.score,behavior.without_just.score],
    'trained-phrase-routing':[],
    'trained-message-removal':[phrase.short.before,phrase.short.after],
    'trained-context-restored':[phrase.short.before,phrase.short.after,phrase.full.before,phrase.full.after],
    'trained-brightness-test':[brightness.baseline,brightness.messages[0].after],
    'trained-state-swap':[context.negative.score,context.positive.score,...context.directions.map(direction=>direction.state_after)],
    'trained-value-swap':[negativeTransfer.before,negativeTransfer.values_after],
  };
  for(const scene of createGuidedTrainedScenes(drawing,original,investigation)){
    const probabilities=scoreSets[scene.id].map(score=>score.positive_probability);
    const permittedPercentages=new Set(probabilities.flatMap(value=>[2,4].map(digits=>Number((value*100).toFixed(digits)))));
    if(['trained-phrase-routing','trained-message-removal'].includes(scene.id)){
      phrase.edges.forEach(edge=>permittedPercentages.add(Number((100*edge.weight).toFixed(1))));
    }
    if(scene.id==='trained-brightness-test')brightness.messages.forEach(message=>permittedPercentages.add(Number((100*message.weight).toFixed(1))));
    for(const reducedMotion of [false,true])for(let step=0;step<=20;step++){
      const svg=scene.draw(step/20,{reducedMotion});
      for(const [,value] of svg.matchAll(/data-positive-probability="([^"]+)"/g)){
        assert.ok(probabilities.includes(Number(value)),`${scene.id} fabricated a probability ${value} at phase ${step/20}`);
      }
      for(const [,value] of svg.matchAll(/(\d+(?:\.\d+)?)%/g)){
        assert.ok(permittedPercentages.has(Number(value)),`${scene.id} displays an unmeasured percentage ${value}%`);
      }
    }
  }
});

test('intervention readouts follow the supplied results rather than hardcoded teaching outcomes',()=>{
  const changed=structuredClone(investigation);
  const replacement=p=>{
    const margin=Math.log(p/(1-p));
    return {logits:[0,margin],margin,positive_probability:p,prediction:p>.5?'POSITIVE':'NEGATIVE'};
  };
  changed.phrase.short.after=replacement(.271828);
  changed.phrase.short.delta_margin=changed.phrase.short.after.margin-changed.phrase.short.before.margin;
  const transfer=changed.context.directions.find(direction=>direction.recipient==='negative');
  transfer.values_after=replacement(.64123);
  transfer.values_delta_margin=transfer.values_after.margin-transfer.before.margin;
  const scenes=createGuidedTrainedScenes(drawing,original,changed);
  for(const [id,value] of [['trained-message-removal',.271828],['trained-value-swap',.64123]]){
    const scene=scenes.find(candidate=>candidate.id===id);
    assert.ok(scene.draw(1).includes(`data-positive-probability="${value}"`));
    assert.ok(scene.draw(1).includes(`${(100*value).toFixed(2)}%`));
  }
});
