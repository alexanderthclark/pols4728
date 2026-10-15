import assert from 'node:assert/strict';
import test from 'node:test';
import { scrollSceneAt } from '../docs/pca/scroll-timeline.mjs';

const close = (actual, expected) => assert.ok(
  Math.abs(actual - expected) < 1e-10,
  `${actual} differs from ${expected}`,
);

test('scroll timeline holds the first and last scenes outside the sequence', () => {
  const first = { index: 0, progress: 0, local: 0 };
  const last = { index: 18, progress: 1, local: 1 };
  assert.deepEqual(scrollSceneAt(-500, 115, 19), first);
  assert.deepEqual(scrollSceneAt(0, 115, 19), first);
  assert.deepEqual(scrollSceneAt(19 * 115, 115, 19), last);
  assert.deepEqual(scrollSceneAt(10000, 115, 19), last);
  assert.deepEqual(scrollSceneAt(-1, 100, 1), first);
  assert.deepEqual(scrollSceneAt(100, 100, 1), { index: 0, progress: 1, local: 1 });
});

test('each scene starts exactly on its boundary and the previous scene completes first', () => {
  for (let index = 0; index < 19; index++) {
    assert.deepEqual(scrollSceneAt(index * 115, 115, 19), { index, progress: 0, local: 0 });
    if (index > 0) {
      const previous = scrollSceneAt(index * 115 - .001, 115, 19);
      assert.equal(previous.index, index - 1);
      assert.equal(previous.progress, 1);
      close(previous.local, 1 - .001 / 115);
    }
  }
});

test('local position is linear while animation pauses then eases between 12% and 82%', () => {
  for (const [local, expected] of [
    [0, 0], [.06, 0], [.12, 0], [.295, .15625], [.47, .5],
    [.645, .84375], [.82, 1], [.9, 1], [.99, 1],
  ]) {
    const scene = scrollSceneAt((3 + local) * 1000, 1000, 19);
    assert.equal(scene.index, 3);
    close(scene.local, local);
    close(scene.progress, expected);
  }
});

test('scrolling in either direction returns identical scene states without history', () => {
  const positions = [-100, 0, 10, 12, 29.5, 47, 64.5, 82, 99, 100, 112, 147, 182, 199, 200, 247, 282, 300, 1000];
  const forward = positions.map(position => scrollSceneAt(position, 100, 3));
  const backward = positions.toReversed().map(position => scrollSceneAt(position, 100, 3)).reverse();
  assert.deepEqual(backward, forward);
  assert.ok(forward.every(({ index, progress, local }) =>
    index >= 0 && index < 3 && progress >= 0 && progress <= 1 && local >= 0 && local <= 1));
});

test('a change in scene height preserves states at the same proportional scroll distance', () => {
  for (const distance of [0, .06, .47, .9, 1, 1.295, 5.645, 18.9, 19, 30]) {
    const short = scrollSceneAt(distance * 500, 500, 19);
    const tall = scrollSceneAt(distance * 1000, 1000, 19);
    assert.equal(short.index, tall.index);
    close(short.local, tall.local);
    close(short.progress, tall.progress);
  }
});

test('invalid scene dimensions, counts, and offsets fail explicitly', () => {
  for (const height of [0, -1, NaN, Infinity, -Infinity, undefined, '100']) {
    assert.throws(() => scrollSceneAt(0, height, 19), RangeError);
  }
  for (const count of [0, -1, 1.5, NaN, Infinity, undefined, '19', Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => scrollSceneAt(0, 100, count), RangeError);
  }
  for (const offset of [NaN, Infinity, -Infinity, undefined, '100']) {
    assert.throws(() => scrollSceneAt(offset, 100, 19), RangeError);
  }
});
