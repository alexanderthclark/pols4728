import test from 'node:test';
import assert from 'node:assert/strict';
import {positionForScroll,sceneAt} from '../docs/fine-tuning-shared/shell.mjs';
test('scroll position progresses continuously and retraces exactly',()=>{
 const points=[0,225,450,675,900,675,450,225,0];
 assert.deepEqual(points.map(y=>positionForScroll(y,0,900,5)),[0,.25,.5,.75,1,.75,.5,.25,0]);
});
test('covers and reading material clamp to first and final visual states',()=>{
 assert.equal(positionForScroll(-100,300,900,14),0);
 assert.equal(positionForScroll(99999,300,900,14),13);
 assert.equal(sceneAt(4.49,14),4);assert.equal(sceneAt(4.5,14),5);
});
