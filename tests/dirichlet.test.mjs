import assert from 'node:assert/strict';
import test from 'node:test';
import {createRequire} from 'node:module';
import {simplexPoint,simplexVertices,simplexMesh,dirichletLogDensity,dirichletMean,categoricalProbability,exampleMixture,concentrationAt} from '../docs/text/dirichlet-model.mjs';
const require=createRequire(import.meta.url);
globalThis.katex=require('../docs/word2vec/vendor/katex/katex.min.js');
const {dirichletVisual}=await import('../docs/text/dirichlet-visual.mjs');
const close=(actual,expected,tolerance=1e-10)=>assert.ok(Math.abs(actual-expected)<tolerance,`${actual} differs from ${expected}`);

test('the simplex represents valid categorical probabilities with exact barycentric geometry',()=>{
  [[1,0,0],[0,1,0],[0,0,1]].forEach((theta,k)=>assert.deepEqual(simplexPoint(theta),simplexVertices[k]));
  assert.deepEqual(simplexPoint(exampleMixture),[288,271]);
  const mean=simplexPoint([1/3,1/3,1/3]);
  close(mean[0],340);close(mean[1],715/3);
  close([0,1,2].reduce((sum,k)=>sum+categoricalProbability(exampleMixture,k),0),1);
  close(categoricalProbability(exampleMixture,2),.3);
  assert.throws(()=>simplexPoint([.7,.2,.3]),RangeError);
  assert.throws(()=>simplexPoint([1.2,-.2,0]),RangeError);
  assert.throws(()=>categoricalProbability(exampleMixture,3),RangeError);
});

test('Dirichlet density agrees with closed forms and its concentration changes shape while preserving the mean',()=>{
  const center=[1/3,1/3,1/3],corner=[.98,.01,.01];
  close(Math.exp(dirichletLogDensity(exampleMixture,[1,1,1])),2);
  close(Math.exp(dirichletLogDensity(exampleMixture,[2,2,2])),120*.5*.2*.3);
  close(dirichletLogDensity(center,[1,1,1]),dirichletLogDensity(corner,[1,1,1]));
  assert.ok(dirichletLogDensity(corner,[.3,.3,.3])>dirichletLogDensity(center,[.3,.3,.3]));
  assert.ok(dirichletLogDensity(corner,[8,8,8])<dirichletLogDensity(center,[8,8,8]));
  for(const a of [.3,1,8])dirichletMean([a,a,a]).forEach(value=>close(value,1/3));
  assert.deepEqual(dirichletMean([5,2,3]),exampleMixture);
  assert.throws(()=>dirichletLogDensity(center,[0,1,1]),RangeError);
});

test('the density mesh uses interior centroids and the scroll schedule holds the uniform case',()=>{
  const mesh=simplexMesh(20);
  assert.equal(mesh.length,400);
  for(const {centroid,vertices} of mesh){
    assert.ok(centroid.every(p=>p>0));close(centroid.reduce((sum,p)=>sum+p,0),1);
    vertices.forEach(theta=>assert.doesNotThrow(()=>simplexPoint(theta)));
    for(const a of [.3,1,8])assert.ok(Number.isFinite(dirichletLogDensity(centroid,[a,a,a])));
  }
  close(concentrationAt(0),.3);close(concentrationAt(.2),.3);
  close(concentrationAt(.5),1);close(concentrationAt(.6),1);close(concentrationAt(.7),1);close(concentrationAt(1),8);
  const integral=mesh.reduce((sum,{centroid})=>sum+Math.exp(dirichletLogDensity(centroid,[2,2,2]))/(2*mesh.length),0);
  close(integral,1,.005);
  const fine=simplexMesh(40);
  const refined=fine.reduce((sum,{centroid})=>sum+Math.exp(dirichletLogDensity(centroid,[2,2,2]))/(2*fine.length),0);
  assert.ok(Math.abs(refined-1)<Math.abs(integral-1)/3);
});

test('the pinned simplex has exact boundaries, reversible geometry, and complete reduced-motion states',()=>{
  assert.equal(dirichletVisual(0,0),dirichletVisual(1,.5));
  assert.equal(dirichletVisual(1,1),dirichletVisual(2,1.5));
  const middle=dirichletVisual(2,1.8);
  assert.equal(middle,dirichletVisual(2,1.8));
  assert.notEqual(middle,dirichletVisual(2,2));
  assert.equal(dirichletVisual(2,1.6,false),dirichletVisual(2,2));
  const keys=markup=>[...markup.matchAll(/data-key="([^"]+)"/g)].map(match=>match[1]);
  assert.deepEqual(keys(dirichletVisual(0)),keys(dirichletVisual(2)));
  assert.equal(new Set(keys(middle)).size,keys(middle).length);
  for(const stage of [0,1,2])for(const delta of [-.5,-.3,0]){
    const markup=dirichletVisual(stage,stage+delta);
    assert.doesNotMatch(markup,/NaN|Infinity|undefined|katex-error/);
    assert.match(markup,/class="katex"/);
  }
  assert.match(dirichletVisual(1),/a probability vector/);
  assert.match(dirichletVisual(1),/one topic/);
  assert.match(dirichletVisual(2),/mean stays one third/);
});
