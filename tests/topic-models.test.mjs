import assert from 'node:assert/strict';
import test from 'node:test';
import {pelotonExample, topicModelsVisual} from '../docs/text/topic-models.mjs';

test('the Peloton illustration preserves excerpts and clearly distinguishes invented shares from fitted results',()=>{
  assert.deepEqual(pelotonExample.excerpts,['Cadence 80 to 100.','Resistance 20 to 40.','Grab those weights.']);
  assert.equal(pelotonExample.topics.reduce((sum,topic)=>sum+topic.share,0),1);
  const markup=topicModelsVisual(1,1,false);
  for(const excerpt of pelotonExample.excerpts)assert.ok(markup.includes(excerpt));
  assert.match(markup,/Illustrative topics and proportions/);
  assert.match(markup,/These are not fitted results/);
  const widths=[...markup.matchAll(/data-key="mixture-segment-\d"[^>]*width="([\d.]+)"/g)].map(match=>Number(match[1]));
  assert.deepEqual(widths,[436.8,187.2]);
  assert.equal(widths.reduce((sum,width)=>sum+width,0),624);
});

test('adjacent stages share exact visual endpoints and preserve every keyed object',()=>{
  assert.equal(topicModelsVisual(0,0),topicModelsVisual(1,.5));
  assert.equal(topicModelsVisual(1,1),topicModelsVisual(2,1.5));
  const keys=markup=>[...markup.matchAll(/data-key="([^"]+)"/g)].map(match=>match[1]);
  assert.deepEqual(keys(topicModelsVisual(0,0)),keys(topicModelsVisual(2,2)));
  assert.equal(new Set(keys(topicModelsVisual(2,2))).size,keys(topicModelsVisual(2,2)).length);
});

test('scroll progress changes mixture geometry and draws metadata without inventing a numerical effect',()=>{
  const start=topicModelsVisual(1,.55),middle=topicModelsVisual(1,.75),end=topicModelsVisual(1,1);
  assert.notEqual(start,middle);assert.notEqual(middle,end);
  assert.equal(middle,topicModelsVisual(1,.75));
  const stm=topicModelsVisual(2,2);
  assert.match(stm,/data-metadata-progress="1"/);
  assert.match(stm,/stroke-dashoffset="0"/);
  assert.match(stm,/data-share="0.7"/);assert.match(stm,/data-share="0.3"/);
  assert.match(stm,/No numerical metadata effect is estimated/);
  assert.equal(topicModelsVisual(2,1.6,false),stm);
  assert.equal(topicModelsVisual(1,.6,false),end);
  assert.throws(()=>topicModelsVisual(3,3),RangeError);
});
