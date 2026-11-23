import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import katex from '../docs/attention/vendor/katex/katex.min.js';
import {createTrainedScenes,formatWeight,normalizeTrainedData} from '../docs/attention/trained-scenes.mjs';

const data=JSON.parse(readFileSync(new URL('../docs/attention/data/infinite-jest-attention.json',import.meta.url),'utf8'));
const texts=[
  "Good and Infinite Jest don't belong in the same sentence",
  "Infinite Jest wasn't just good. It was spectacular.",
];
const near=(actual,expected,tolerance,message)=>assert.ok(Math.abs(actual-expected)<=tolerance,`${message}: ${actual} versus ${expected}`);

test('measured attention comes from the pinned sentiment checkpoint in evaluation mode',()=>{
  assert.equal(data.schema_version,1);
  assert.equal(data.model.id,'distilbert/distilbert-base-uncased-finetuned-sst-2-english');
  assert.equal(data.model.revision,'714eb0fa89d2f80546fda750413ed43d93601a13');
  assert.equal(data.model.architecture,'DistilBertForSequenceClassification');
  assert.equal(data.model.num_layers,6);
  assert.equal(data.model.num_heads,12);
  assert.equal(data.model.hidden_size,768);
  assert.equal(data.model.readout,'[CLS]');
  assert.deepEqual(data.model.labels,['NEGATIVE','POSITIVE']);
  assert.equal(data.inference.model_mode,'eval');
  assert.equal(data.inference.dtype,'float32');
  assert.equal(data.inference.attention_implementation,'eager');
  assert.equal(data.inference.padding,false);
  assert.equal(data.inference.truncation,false);
  assert.deepEqual(data.inference.attention_axes,['layer','head','destination_token','source_token']);
});

test('actual token IDs and offsets retain the unchanged sentences and all special tokens',()=>{
  assert.deepEqual(data.examples.map(example=>example.text),texts);
  const tokenToId=new Map(),idToToken=new Map();
  for(const example of data.examples){
    const {tokens,token_ids:ids,token_offsets:offsets,attention_mask:mask,text}=example;
    assert.equal(tokens.length,ids.length);
    assert.equal(tokens.length,offsets.length);
    assert.equal(tokens.length,mask.length);
    assert.equal(tokens[0],'[CLS]');assert.equal(ids[0],101);
    assert.equal(tokens.at(-1),'[SEP]');assert.equal(ids.at(-1),102);
    assert.deepEqual(offsets[0],[0,0]);assert.deepEqual(offsets.at(-1),[0,0]);
    assert.ok(mask.every(value=>value===1),'no padding token may silently disappear from the measured matrix');
    assert.ok(tokens.some(token=>token.startsWith('##')),'retain actual WordPiece fragments');
    assert.ok(tokens.includes("'"),'retain tokenized punctuation');
    let previousEnd=0;
    tokens.forEach((token,i)=>{
      const id=ids[i];assert.ok(Number.isInteger(id)&&id>=0);
      if(tokenToId.has(token))assert.equal(id,tokenToId.get(token));
      if(idToToken.has(id))assert.equal(token,idToToken.get(id));
      tokenToId.set(token,id);idToToken.set(id,token);
      if(i===0||i===tokens.length-1)return;
      const [start,end]=offsets[i];
      assert.ok(Number.isInteger(start)&&Number.isInteger(end)&&start>=previousEnd&&end>start&&end<=text.length);
      assert.equal(text.slice(previousEnd,start).trim(),'','token offsets must not omit input text');
      assert.equal(token.replace(/^##/,''),text.slice(start,end).toLowerCase());
      previousEnd=end;
    });
    assert.equal(text.slice(previousEnd).trim(),'');
  }
});

test('recorded probabilities and predicted labels agree with the exported logits',()=>{
  for(const example of data.examples){
    assert.equal(example.logits.length,2);assert.equal(example.probabilities.length,2);
    assert.ok(example.logits.every(Number.isFinite));
    const max=Math.max(...example.logits),exps=example.logits.map(value=>Math.exp(value-max)),normalizer=exps.reduce((a,b)=>a+b,0);
    example.probabilities.forEach((value,i)=>{
      assert.ok(Number.isFinite(value)&&value>=0&&value<=1);
      near(value,exps[i]/normalizer,2e-7,`softmax probability ${i}`);
    });
    near(example.probabilities.reduce((a,b)=>a+b,0),1,2e-7,'class probabilities sum to one');
    const index=example.logits[1]>example.logits[0]?1:0;
    assert.equal(example.prediction.index,index);
    assert.equal(example.prediction.label,data.model.labels[index]);
    assert.equal(example.prediction.probability,example.probabilities[index]);
  }
});

test('every measured head contains a complete normalized destination-by-source matrix',()=>{
  for(const example of data.examples){
    const n=example.tokens.length;
    assert.equal(example.attentions.length,6);
    for(const layer of example.attentions){
      assert.equal(layer.length,12);
      for(const head of layer){
        assert.equal(head.length,n);
        for(const row of head){
          assert.equal(row.length,n);
          assert.ok(row.every(value=>Number.isFinite(value)&&value>=0&&value<=1));
          near(row.reduce((a,b)=>a+b,0),1,2e-6,'attention row sum including special tokens');
        }
      }
    }
    assert.ok(example.validation.maximum_attention_row_sum_error<2e-6);
  }
});

test('trained-model scenes render strict mathematics and restore measured states when scrolling back',()=>{
  const drawing={C:{blue:'#234e70',teal:'#007f73',rust:'#a54f32',ink:'#242424',muted:'#606060'},
    m:tex=>katex.renderToString(tex,{throwOnError:true,strict:'error',output:'htmlAndMathml'}),
    text:(text,...args)=>`<text data-args="${args.join(',')}">${text}</text>`,
    line:(...args)=>`<line data-args="${args.join(',')}"/>`,
    rect:(...args)=>`<rect data-args="${args.join(',')}"/>`};
  const scenes=createTrainedScenes(drawing,data),measured=normalizeTrainedData(data);
  assert.equal(scenes.length,3);
  for(const scene of scenes){
    drawing.m(scene.equation);const initial=scene.draw(0);
    for(const phase of [.1,.33,.5,.75,1])assert.doesNotMatch(scene.draw(phase),/NaN|undefined|\[object Object\]/);
    assert.notEqual(scene.draw(1),initial,scene.id);
    assert.equal(scene.draw(0),initial,scene.id);
  }
  measured.examples.forEach((example,i)=>{
    assert.equal(example.prediction,data.examples[i].prediction.label);
    const svg=scenes[i+1].draw(1),row=example.attentions[5][0][example.tokens.indexOf('[CLS]')];
    assert.match(svg,/\[CLS\]/);assert.match(svg,/\[SEP\]/);
    const largest=[...row].sort((a,b)=>b-a).slice(0,3);
    for(const value of largest)assert.ok(svg.includes(formatWeight(value)),'displayed top weights must come from layer 6, head 1, [CLS]');
  });
});
