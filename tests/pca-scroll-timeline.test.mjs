import assert from 'node:assert/strict';
import test from 'node:test';
import { scrollLocalAtProgress, scrollSceneAt } from '../docs/pca/scroll-timeline.mjs';

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

test('inverse progress clamps to the start and finish of the active scroll interval', () => {
  assert.equal(scrollLocalAtProgress(-5), .12);
  assert.equal(scrollLocalAtProgress(0), .12);
  assert.equal(scrollLocalAtProgress(1), .82);
  assert.equal(scrollLocalAtProgress(5), .82);
  close(scrollLocalAtProgress(.5), .47);
});

test('inverse progress round trips across scenes and proportional viewport heights', () => {
  const progressValues = [0, 1e-12, 1e-9, 1e-6,
    ...Array.from({ length: 99 }, (_, index) => (index + 1) / 100),
    1 - 1e-6, 1 - 1e-9, 1 - 1e-12, 1];
  let previousLocal = -Infinity;
  for (const progress of progressValues) {
    const local = scrollLocalAtProgress(progress);
    assert.ok(local >= previousLocal, `Local scroll position must increase at ${progress}`);
    previousLocal = local;
    for (const count of [1, 3, 19]) {
      for (const index of new Set([0, Math.floor(count / 2), count - 1])) {
        for (const height of [1, 100, 736 * 1.15, 1234.5]) {
          const state = scrollSceneAt((index + local) * height, height, count);
          assert.equal(state.index, index);
          assert.ok(Math.abs(state.progress - progress) < 2e-14,
            `Progress ${progress} did not survive scene ${index}/${count} at height ${height}: ${state.progress}`);
          close(state.local, local);
        }
      }
    }
  }
});

test('tiny positive progress survives an inverse round trip without snapping to the start', () => {
  for (const progress of [1e-20, 1e-16, 1e-12, 1e-9]) {
    const local = scrollLocalAtProgress(progress);
    assert.ok(local > .12);
    for (const height of [115, 736, 1000, 1e6]) {
      const state = scrollSceneAt(local * height, height, 1);
      assert.ok(Math.abs(state.progress / progress - 1) < 1e-6,
        `Tiny progress ${progress} changed to ${state.progress} at height ${height}`);
    }
  }
});

test('inverse progress rejects nonfinite and nonnumeric input', () => {
  for (const progress of [NaN, Infinity, -Infinity, undefined, '0.5', null]) {
    assert.throws(() => scrollLocalAtProgress(progress), RangeError);
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
