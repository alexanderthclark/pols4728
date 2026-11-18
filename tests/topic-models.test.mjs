import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';
const require=createRequire(import.meta.url);
globalThis.katex=require('../docs/word2vec/vendor/katex/katex.min.js');
const {pelotonExample,topicModelsVisual}=await import('../docs/text/topic-models.mjs');
const {dirichletVisual}=await import('../docs/text/dirichlet-visual.mjs');
const {ldaMechanismVisual}=await import('../docs/text/lda-mechanism.mjs');
const {chapters}=await import('../docs/text/lesson.mjs');
const quote="So I love Seated Baby Cradle, because it's an opportunity for me to practice holding my body in a way that acknowledges every part of me as being precious.";

test('the expanded explanation builds probability vectors before LDA generation, inference, and STM',()=>{
  const chapter=chapters.at(-1);
  assert.equal(chapter.id,'topic-models');assert.equal(chapter.application,true);
  assert.deepEqual(chapter.stages.map(s=>s.visual),['simplex','dirichlet','concentration','topic-words','document-mixture','token-topic','token-word','inference','stm-prevalence','stm-content']);
  assert.match(chapter.stages[0].readingText,/simplex/i);
  assert.match(chapter.stages[1].readingText,/Dirichlet/);
  assert.match(chapter.stages[8].readingText,/logistic.normal/i);
});

test('the actual model walkthrough preserves the full requested quote and correct class attribution',()=>{
  assert.ok(pelotonExample.excerpts.includes(quote));
  const markup=topicModelsVisual(4,4,false);
  assert.ok(markup.includes(quote),'The complete quotation must remain accessible in the rendered model.');
  assert.match(markup,/Chelsea Jackson Roberts/);
  assert.match(markup,/Seated Baby Cradle/);
  assert.ok(chapters.at(-1).stages[4].readingText.includes(quote));
  assert.match(chapters.at(-1).stages[4].readingText,/Focused Flow/);
});

test('the lesson routes progress and reduced motion into every mathematical scene',()=>{
  for(let stage=0;stage<10;stage++){
    for(const motion of [true,false])for(const delta of [-.4,0,.3]){
      const p=stage+delta;
      assert.equal(topicModelsVisual(stage,p,motion),stage<3?dirichletVisual(stage,p,motion):ldaMechanismVisual(stage-3,p-3,motion));
    }
  }
  assert.throws(()=>topicModelsVisual(10,10),RangeError);
  assert.throws(()=>topicModelsVisual(-1,-1),RangeError);
});
