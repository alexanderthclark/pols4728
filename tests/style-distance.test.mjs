import assert from 'node:assert/strict';
import test from 'node:test';
import { passages, triplet, margin, squaredDistance, tripletLoss, embeddingState, stateAt, projectVector } from '../docs/style-distance/model.mjs';
import { lesson } from '../docs/style-distance/lesson.mjs';

test('the selected triplet distinguishes opposite claims from different styles', () => {
  const byId=Object.fromEntries(passages.map(p=>[p.id,p]));
  // Assert the named roles independently: the symmetric map alone would also
  // give the old B/D/A triplet the same numerical distances.
  assert.deepEqual(triplet,{anchor:'A',positive:'C',negative:'B'});
  assert.equal(byId.A.text,'the race is not to the swift');
  assert.equal(byId.B.text,'…success or failure in competitive activities exhibits no tendency to be commensurate with innate capacity…');
  assert.equal(byId.C.text,'the swift shall win the race');
  assert.equal(byId.D.text,'Competitive success accrues to the participant exhibiting the greatest velocity.');
  for(const id of ['A','B']) assert.equal(byId[id].content,'uncertain');
  for(const id of ['C','D']) assert.equal(byId[id].content,'certain');
  for(const id of ['A','C']) assert.equal(byId[id].style,'biblical');
  for(const id of ['B','D']) assert.equal(byId[id].style,'academic');
  const a=byId[triplet.anchor],p=byId[triplet.positive],n=byId[triplet.negative];
  assert.equal(a.style,p.style);
  assert.notEqual(a.content,p.content);
  assert.equal(a.content,n.content);
  assert.notEqual(a.style,n.style);
  assert.equal(new Set(passages.map(p=>`${p.content}:${p.style}`)).size,4);
});

test('triplet hinge uses squared distance and the stated margin', () => {
  assert.equal(squaredDistance([1,2],[4,6]),25);
  assert.equal(tripletLoss([0,0],[2,0],[1,0],0.1),3.1);
  assert.equal(tripletLoss([0,0],[1,0],[2,0],0.1),0);
  assert.equal(margin,0.1);
  for(let step=0;step<=200;step++) {
    const s=embeddingState(step/200);
    const positive=(s.vectors.A[0]-s.vectors.C[0])**2+(s.vectors.A[1]-s.vectors.C[1])**2;
    const negative=(s.vectors.A[0]-s.vectors.B[0])**2+(s.vectors.A[1]-s.vectors.B[1])**2;
    const independentlyComputed=Math.max(0,positive-negative+0.1);
    assert.ok(Math.abs(s.positiveDistance-positive)<1e-12);
    assert.ok(Math.abs(s.negativeDistance-negative)<1e-12);
    assert.ok(Math.abs(s.loss-independentlyComputed)<1e-12);
    assert.ok(Number.isFinite(s.loss)&&s.loss>=0);
    // All coordinates remain within the fixed displayed range.
    assert.ok(Object.values(s.vectors).flat().every(value=>Math.abs(value)<1.5));
  }
});

test('the schematic changes every embedding while retaining identity and axis metric', () => {
  const initial=stateAt(7),final=stateAt(9);
  assert.ok(Math.abs(initial.positiveDistance-5.29)<1e-12);
  assert.ok(Math.abs(initial.negativeDistance-0.3136)<1e-12);
  assert.ok(Math.abs(initial.loss-5.0764)<1e-12);
  assert.ok(Math.abs(final.positiveDistance-0.4096)<1e-12);
  assert.ok(Math.abs(final.negativeDistance-3.61)<1e-12);
  assert.ok(initial.positiveDistance>initial.negativeDistance);
  assert.ok(final.positiveDistance+margin<=final.negativeDistance);
  assert.ok(initial.loss>0);
  assert.equal(final.loss,0);
  assert.deepEqual(Object.keys(initial.vectors),Object.keys(final.vectors));
  for(const p of passages) assert.notDeepEqual(initial.vectors[p.id],final.vectors[p.id]);
  assert.deepEqual(stateAt(-1),stateAt(0));
  assert.deepEqual(stateAt(Infinity),stateAt(0));
});

test('desktop and mobile projection preserve Euclidean distance with equal axis scales', () => {
  for(const frame of [{x:465,y:135,width:485,height:300},{x:55,y:156,width:390,height:204}]) {
    const scale=Math.min(frame.width,frame.height)/3;
    for(const progress of [0,0.5,1]) {
      const vectors=embeddingState(progress).vectors;
      for(const id of ['B','C','D']) {
        const actual=squaredDistance(projectVector(vectors.A,frame),projectVector(vectors[id],frame));
        const expected=squaredDistance(vectors.A,vectors[id])*scale*scale;
        assert.ok(Math.abs(actual-expected)<1e-8);
      }
    }
  }
});

test('scroll interpolation is continuous, reversible, and independent of rendering history', () => {
  for(const p of [0,2.8,3,4.9,6.5,7,7.4,8.5,9,10.9,11]) {
    const expected=lesson.render(p);
    lesson.render(11);lesson.render(0);
    assert.equal(lesson.render(p),expected);
    for(const axis of [0,1]) {
      const below=stateAt(p-1e-5).vectors.D[axis],above=stateAt(p+1e-5).vectors.D[axis];
      assert.ok(Math.abs(above-below)<0.0001);
    }
  }
  assert.notEqual(lesson.render(7.3),lesson.render(7.7));
  assert.notEqual(lesson.render(8.2),lesson.render(8.3));
});

test('the diagram connects the named anchor to C and B rather than the old symmetric triplet', () => {
  for(const compact of [false,true]) for(const position of [4,5,6,8,9]) {
    const svg=lesson.render(position,{compact});
    assert.match(svg,/data-edge="A-C"/);
    assert.match(svg,/data-edge="A-B"/);
    assert.doesNotMatch(svg,/data-edge="B-D"/);
  }
});

test('diagram transitions never overlay the text from different visual modes', () => {
  for(const compact of [false,true]) for(let step=0;step<=110;step++) {
    const svg=lesson.render(step/10,{compact});
    assert.equal([...svg.matchAll(/data-diagram-mode=/g)].length,1);
    const mode=svg.match(/data-diagram-mode="([^"]+)"/)[1];
    assert.equal(svg.includes('Three passages enter the same encoder'),mode==='encoder'&&!compact);
    assert.equal(svg.includes('The trainable parameters receive the update'),mode==='update'&&!compact);
  }
  assert.match(lesson.render(2.5),/data-diagram-mode="encoder" opacity="0.00"/);
  assert.match(lesson.render(6.5),/data-diagram-mode="update" opacity="0.00"/);
});

test('lesson has a complete readable route and an honest data provenance', () => {
  assert.equal(lesson.scenes.length,12);
  assert.equal(new Set(lesson.scenes.map(s=>s.id)).size,12);
  assert.ok(lesson.sources.some(s=>s.url==='https://aclanthology.org/2025.naacl-long.436/'));
  assert.ok(lesson.sources.some(s=>/Orwell/.test(s.label)),'the academic rendering is attributed to Orwell');
  assert.match(lesson.dataNote,/not from a trained language model/);
  assert.match(lesson.scenes[10].note,/No held-out scores/);
  for(const [index,scene]of lesson.scenes.entries()) {
    assert.ok(scene.title&&scene.body,`scene ${index} has its text route`);
    for(const compact of [false,true]) {
      const svg=lesson.render(index,{compact});
      assert.doesNotMatch(svg,/NaN|Infinity|<foreignObject|<script|setInterval/);
      assert.match(svg,/<title>/);
      assert.ok([...svg.matchAll(/font-size="([\d.]+)"/g)].every(match=>Number(match[1])>=20));
      for(const [tag]of svg.matchAll(/<text\b[^>]*>/g)) assert.equal([...tag.matchAll(/font-size=/g)].length,1);
    }
  }
});
