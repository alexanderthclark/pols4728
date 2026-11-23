import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import katex from '../docs/attention/vendor/katex/katex.min.js';
import {formatWeight} from '../docs/attention/trained-scenes.mjs';
import {createGuidedTrainedScenes} from '../docs/attention/trained-tour-scenes.mjs';

const data=JSON.parse(readFileSync(new URL('../docs/attention/data/infinite-jest-attention.json',import.meta.url),'utf8'));
const near=(actual,expected,tolerance,message)=>assert.ok(Math.abs(actual-expected)<=tolerance,`${message}: ${actual} versus ${expected}`);
const weight=(example,layer,head,destination,source)=>example.attentions[layer-1][head-1][example.tokens.indexOf(destination)][example.tokens.indexOf(source)];
const clamp=(value,minimum=0,maximum=1)=>Math.max(minimum,Math.min(maximum,value));
const escapeText=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const drawing={
  C:{blue:'#234e70',teal:'#007f73',rust:'#a54f32',ink:'#242424',muted:'#606060'},
  m:tex=>katex.renderToString(tex,{throwOnError:true,strict:'error',output:'htmlAndMathml'}),
  text:(value,...args)=>`<text data-args="${escapeText(args.join(','))}">${escapeText(value)}</text>`,
  line:(...args)=>`<line data-args="${escapeText(args.join(','))}"/>`,
  rect:(x,y,width,height,...args)=>`<rect x="${x}" y="${y}" width="${width}" height="${height}" data-args="${escapeText(args.join(','))}"/>`,
  circle:(...args)=>`<circle data-args="${escapeText(args.join(','))}"/>`,
  clamp,
  lerp:(from,to,phase)=>from+(to-from)*phase,
  ease:value=>{const phase=clamp(value);return phase*phase*(3-2*phase);},
};
const sceneById=(scenes,id)=>{
  const scene=scenes.find(candidate=>candidate.id===id);
  assert.ok(scene,`missing guided observation ${id}`);
  return scene;
};

test('the guided next-token observation holds for every eligible content position in both reviews',()=>{
  for(const example of data.examples){
    const matrix=example.attentions[1][9];
    // Exclude [CLS], [SEP], and the last ordinary token: the stated pattern
    // concerns an ordinary destination with a following ordinary source.
    for(let destination=1;destination<example.tokens.length-2;destination++){
      const row=matrix[destination],source=destination+1;
      assert.ok(row[source]>.9999,`${example.tokens[destination]} should read its next token ${example.tokens[source]} with >99.99% weight`);
      assert.equal(row.indexOf(Math.max(...row)),source);
    }
  }
});

test('the two first-layer heads read opposite pieces of the same contraction',()=>{
  const [negative,positive]=data.examples;
  near(weight(negative,1,3,"'",'don'),.9920895099639893,1e-7,'negative contraction: previous-piece coefficient');
  near(weight(negative,1,11,"'",'t'),.9878438711166382,1e-7,'negative contraction: following-piece coefficient');
  // Check the same relation at a different location and with a different stem;
  // this guards against accidentally rendering a fixed column from review 1.
  near(weight(positive,1,3,"'",'wasn'),.9924139380455017,1e-7,'positive contraction: previous-piece coefficient');
  near(weight(positive,1,11,"'",'t'),.9837672710418701,1e-7,'positive contraction: following-piece coefficient');
});

test('the same late head reads good in both reviews despite their opposite predictions',()=>{
  const [negative,positive]=data.examples;
  near(weight(negative,5,8,'[CLS]','good'),.46756666898727417,1e-7,'negative review: good coefficient');
  near(weight(positive,5,8,'[CLS]','good'),.625940203666687,1e-7,'positive review: good coefficient');
  assert.ok(negative.probabilities[1]<.01);
  assert.ok(positive.probabilities[1]>.99);
});

test('the surprising final readout example is the actual CLS-to-apostrophe coefficient',()=>{
  const negative=data.examples[0];
  assert.equal(data.model.architecture,'DistilBertForSequenceClassification');
  assert.equal(data.model.readout,'[CLS]');
  near(weight(negative,6,10,'[CLS]',"'"),.8731127381324768,1e-7,'final apostrophe coefficient');
  const row=negative.attentions[5][9][0];
  near(row.reduce((sum,value)=>sum+value,0),1,2e-6,'full final-layer row, including special tokens');
  assert.equal(row.indexOf(Math.max(...row)),negative.tokens.indexOf("'"));
});

test('guided scenes render strict mathematics and restore the same measured states in reverse',()=>{
  const scenes=createGuidedTrainedScenes(drawing,data);
  assert.equal(new Set(scenes.map(scene=>scene.id)).size,scenes.length,'scene identities must remain distinct');
  for(const scene of scenes){
    drawing.m(scene.equation);
    const phases=[0,.1,.35,.5,.7,.9,1];
    const forward=phases.map(phase=>scene.draw(phase));
    for(const output of forward)assert.doesNotMatch(output,/NaN|undefined|\[object Object\]/,scene.id);
    assert.ok(new Set(forward).size>1,`${scene.id} must show a change during its scroll progression`);
    for(let index=phases.length-1;index>=0;index--){
      assert.equal(scene.draw(phases[index]),forward[index],`${scene.id} must restore phase ${phases[index]}`);
    }
  }
});

test('the opening matrix retains every measured cell and every special-token row',()=>{
  const scene=sceneById(createGuidedTrainedScenes(drawing,data),'trained-local-routing');
  const example=data.examples[0],matrix=example.attentions[1][9];
  for(const phase of [0,.5,1]){
    const svg=scene.draw(phase);
    assert.equal((svg.match(/<title>/g)||[]).length,example.tokens.length**2);
    example.tokens.forEach((destination,i)=>example.tokens.forEach((source,j)=>{
      const description=`Layer 2 head 10. Destination ${i+1} ${destination}, source ${j+1} ${source}: ${formatWeight(matrix[i][j])}`;
      assert.ok(svg.includes(`<title>${escapeText(description)}</title>`),description);
    }));
    assert.match(svg,/\[CLS\]/);assert.match(svg,/\[SEP\]/);
  }
});

test('the final summary retains each separate CLS row and every measured source column',()=>{
  const scene=sceneById(createGuidedTrainedScenes(drawing,data),'trained-readout-route');
  const example=data.examples[0],destination=example.tokens.indexOf('[CLS]');
  const svg=scene.draw(1);
  example.attentions[5].forEach((matrix,head)=>example.tokens.forEach((source,j)=>{
    const description=`Layer 6 head ${head+1}. Destination ${destination+1} [CLS], source ${j+1} ${source}: ${formatWeight(matrix[destination][j])}`;
    assert.ok(svg.includes(`<title>${escapeText(description)}</title>`),description);
  }));
  assert.match(svg,/Earlier \[CLS\]/,'the diagram must preserve the incoming residual route');
});

test('grouped good bars use original weights and preserve the mass of all other positions',()=>{
  const scene=sceneById(createGuidedTrainedScenes(drawing,data),'trained-sentiment-context');
  const svg=scene.draw(1);
  const groups=[...svg.matchAll(/<g clip-path="url\(#([^\)]+)\)">([\s\S]*?)<\/g>/g)];
  assert.equal(groups.length,2);
  groups.forEach(([,clip,contents],index)=>{
    const example=data.examples[index],row=example.attentions[4][7][0],good=example.tokens.indexOf('good');
    const rest=row.reduce((sum,value,source)=>sum+(source===good?0:value),0);
    assert.ok(contents.includes(`good ${formatWeight(row[good])}; all other positions ${formatWeight(rest)}`));
    const clipMarkup=svg.match(new RegExp(`<clipPath id="${clip}">([\\s\\S]*?)<\\/clipPath>`));
    assert.ok(clipMarkup);
    const fullWidth=Number(clipMarkup[1].match(/width="([^"]+)"/)[1]);
    const segments=[...contents.matchAll(/<rect x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"/g)];
    assert.equal(segments.length,2);
    const [goodBar,restBar]=segments;
    near(Number(goodBar[3])/fullWidth,row[good],1e-12,'good occupies its original share of the absolute scale');
    near(Number(restBar[3])/fullWidth,rest,1e-12,'other positions retain their original combined mass');
    near(Number(restBar[1]),Number(goodBar[1])+Number(goodBar[3]),1e-10,'the two bars meet without gaps or overlap');
    near((Number(goodBar[3])+Number(restBar[3]))/fullWidth,1,2e-6,'the complete measured row is retained');
  });
});

test('displayed weights change with the supplied data instead of hardcoded teaching numbers',()=>{
  const changed=structuredClone(data);
  const replaceWeight=(example,layer,head,destination,source,newWeight)=>{
    const row=example.attentions[layer-1][head-1][example.tokens.indexOf(destination)];
    const column=example.tokens.indexOf(source),oldOther=row.reduce((sum,value,index)=>sum+(index===column?0:value),0);
    row.forEach((value,index)=>{row[index]=index===column?newWeight:value*(1-newWeight)/oldOther;});
  };
  replaceWeight(changed.examples[0],5,8,'[CLS]','good',.413);
  replaceWeight(changed.examples[0],6,10,'[CLS]',"'",.327);
  const scenes=createGuidedTrainedScenes(drawing,changed);
  const goodScene=sceneById(scenes,'trained-sentiment-context');
  assert.ok(goodScene.draw(1).includes('good 0.413000; all other positions 0.587000'));
  assert.match(goodScene.detail,/41\.3%/);
  const apostropheScene=sceneById(scenes,'trained-contextual-state');
  assert.match(apostropheScene.draw(1),/32\.7%/);
  assert.match(apostropheScene.detail,/32\.7%/);
});
