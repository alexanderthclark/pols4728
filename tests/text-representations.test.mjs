import assert from 'node:assert/strict';
import test from 'node:test';
import {documents, tokenize, vocabulary, counts, termStats, tfidf, cosine, displayTerms, documentFrequency, referenceCount, preprocessPreview, explorerTerms} from '../docs/text/model.mjs';

function close(actual, expected, tolerance=1e-12) {
  assert.ok(Math.abs(actual-expected)<tolerance, `${actual} ≈ ${expected}`);
}
test('preview tokenization case-folds and preserves repetitions',()=>{
  assert.deepEqual(tokenize('The SENATE, the budget! 2026'),['the','senate','the','budget']);
  assert.deepEqual(tokenize('... 123'),[]);
  for(const doc of documents){
    assert.deepEqual(tokenize(doc.text),doc.previewTokens);
    assert.deepEqual(preprocessPreview(doc.text),doc.previewRetainedTokens);
    assert.deepEqual(doc.tokens,doc.previewRetainedTokens);
  }
});
test('the published bags use two full classes and the audited 127-class vocabulary',()=>{
  assert.deepEqual(documents.map(d=>d.label),['Cycling','Yoga']);
  assert.match(documents[0].instructor,/Ally Love/);
  assert.match(documents[1].instructor,/Chelsea Jackson Roberts/);
  assert.equal(referenceCount,127);
  assert.equal(vocabulary.length,2948);
  assert.equal(new Set(vocabulary).size,vocabulary.length);
  assert.deepEqual(counts.map(row=>row.reduce((a,b)=>a+b,0)),[1490,2154]);
  for(const [r,row] of counts.entries()){
    assert.equal(row.length,vocabulary.length);
    assert.ok(row.every(v=>Number.isInteger(v)&&v>=0));
    assert.equal(row.reduce((a,b)=>a+b,0),documents[r].retainedTokenCount);
    assert.ok(documents[r].tokens.length<documents[r].retainedTokenCount,'Preview is only a short excerpt.');
  }
  assert.equal(documentFrequency.length,vocabulary.length);
  assert.ok(documentFrequency.every(df=>Number.isInteger(df)&&df>=3&&df<=127));
  assert.deepEqual(displayTerms,['take','resistance','confidence','inhale','earth','tabletop']);
  assert.deepEqual(counts.map(row=>displayTerms.map(t=>row[vocabulary.indexOf(t)])),[[11,42,28,2,0,0],[41,0,0,105,54,30]]);
  assert.deepEqual(displayTerms.map(t=>termStats(t,0).df),[127,110,25,77,4,6]);
  assert.equal(explorerTerms.length,24);
  assert.ok(explorerTerms.every(t=>vocabulary.includes(t)));
});
test('IDF uses the reference corpus rather than the two displayed rows',()=>{
  assert.deepEqual(termStats('earth',1),{tf:54,df:4,N:127,idf:Math.log(127/4),weight:54*Math.log(127/4)});
  assert.equal(termStats('earth',0).weight,0);
  for(const r of [0,1])assert.equal(termStats('take',r).weight,0);
  close(termStats('resistance',0).weight,42*Math.log(127/110));
  close(termStats('inhale',1).weight,105*Math.log(127/77));
  assert.ok(termStats('confidence',0).tf<termStats('resistance',0).tf);
  assert.ok(termStats('confidence',0).weight>termStats('resistance',0).weight);
  assert.ok(termStats('earth',1).tf<termStats('inhale',1).tf);
  assert.ok(termStats('earth',1).weight>termStats('inhale',1).weight);
  assert.throws(()=>termStats('not-in-the-vocabulary',0),RangeError);
  assert.throws(()=>termStats('earth',-1),RangeError);
  assert.throws(()=>termStats('earth',2),RangeError);
});
test('every TF–IDF coordinate matches raw count times the saved document-frequency weight',()=>{
  tfidf.forEach((row,r)=>row.forEach((value,c)=>close(value,counts[r][c]*Math.log(127/documentFrequency[c]))));
});
test('the full-class cosine comparison reflects the changed weights',()=>{
  close(cosine([1,2],[2,4]),1);
  close(cosine([1,0],[0,1]),0);
  close(cosine([1,0],[-1,0]),-1);
  assert.equal(cosine([0,0],[2,1]),0);
  assert.throws(()=>cosine([1],[1,2]),RangeError);
  close(cosine(counts[0],counts[1]),.23183257,1e-8);
  close(cosine(tfidf[0],tfidf[1]),.02151493,1e-8);
  assert.ok(cosine(tfidf[0],tfidf[1])<cosine(counts[0],counts[1]));
});
