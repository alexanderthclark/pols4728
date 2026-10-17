import assert from 'node:assert/strict';
import test from 'node:test';
import { speechFrame, speechPath, speechWords } from '../docs/text/speech-motion.mjs';

function countsOf(words) {
  return Object.fromEntries([...new Set(speechWords)].map(word => [word, words.filter(value => value === word).length]));
}

test('the fragment retains all ten words and counts its repeated words independently', () => {
  assert.equal(speechWords.join(' '), 'government of the people by the people for the people');
  const first = speechFrame(0);
  assert.equal(first.caughtTotal, 0);
  assert.equal(first.pendingTotal, 10);
  assert.deepEqual(first.tokens, []);
  assert.equal(first.bagFullness, 0);

  const last = speechFrame(1);
  assert.deepEqual(last.caughtCounts, { government: 1, of: 1, the: 3, people: 3, by: 1, for: 1 });
  assert.equal(last.caughtTotal, 10);
  assert.equal(last.pendingTotal, 0);
  assert.deepEqual(last.tokens, []);
  assert.equal(last.bagFullness, 1);
  assert.equal(last.mouthOpen, 0);
  assert.equal(last.catcherLean, 0);
});

test('every word progresses from pending to airborne to caught, without loss or duplication', () => {
  const previousState = speechWords.map(() => 'pending');
  const observed = speechWords.map(() => new Set());
  let previousCaught = 0;
  let mostAirborne = 0;
  for (let step = 0; step <= 2000; step += 1) {
    const frame = speechFrame(step / 2000);
    assert.equal(frame.total, 10);
    assert.equal(frame.pendingTotal + frame.tokens.length + frame.caughtTotal, frame.total);
    assert.equal(new Set(frame.tokens.map(token => token.index)).size, frame.tokens.length);
    assert.ok(frame.caughtTotal >= previousCaught);
    assert.deepEqual(frame.caughtCounts, countsOf(speechWords.slice(0, frame.caughtTotal)));
    assert.equal(Object.values(frame.caughtCounts).reduce((sum, count) => sum + count, 0), frame.caughtTotal);
    assert.equal(frame.bagFullness, frame.caughtTotal / frame.total);
    mostAirborne = Math.max(mostAirborne, frame.tokens.length);

    for (const [index, word] of speechWords.entries()) {
      const token = frame.tokens.find(value => value.index === index);
      const state = index < frame.caughtTotal ? 'caught' : token ? 'airborne' : 'pending';
      if (token) assert.equal(token.word, word);
      const allowedNext = { pending: ['pending', 'airborne'], airborne: ['airborne', 'caught'], caught: ['caught'] };
      assert.ok(allowedNext[previousState[index]].includes(state), `${index}: ${previousState[index]} → ${state}`);
      observed[index].add(state);
      previousState[index] = state;
    }
    previousCaught = frame.caughtTotal;
  }
  assert.equal(mostAirborne, 3);
  for (const states of observed) assert.deepEqual([...states], ['pending', 'airborne', 'caught']);
});

test('each separate word rises from the mouth, travels forward, and descends into the bag', () => {
  const paths = speechWords.map(() => []);
  for (let step = 0; step <= 10000; step += 1) {
    const frame = speechFrame(step / 10000);
    for (const token of frame.tokens) paths[token.index].push(token);
  }
  for (const path of paths) {
    const first = path[0];
    const last = path.at(-1);
    assert.ok(Math.abs(first.x - speechPath.mouth[0]) < 1);
    assert.ok(Math.abs(first.y - speechPath.mouth[1]) < 1);
    assert.ok(Math.abs(last.x - speechPath.bag[0]) < 1);
    assert.ok(Math.abs(last.y - speechPath.bag[1]) < 1);
    assert.ok(Math.min(...path.map(token => token.y)) < 170, 'the arc leaves clear space above the figures');
    assert.ok(path[1].y < first.y, 'words initially rise');
    assert.ok(last.y > path.at(-2).y, 'words descend toward the bag');
    for (let index = 1; index < path.length; index += 1) {
      assert.ok(path[index].x > path[index - 1].x, 'a word never backs up or loops');
    }
  }
});

test('all poses and token properties remain finite and within their intended ranges', () => {
  for (const reducedMotion of [false, true]) {
    for (let step = 0; step <= 1000; step += 1) {
      const frame = speechFrame(step / 1000, { reducedMotion });
      assert.ok(frame.mouthOpen >= 0 && frame.mouthOpen <= 1);
      assert.ok(Math.abs(frame.gesture) <= 1);
      assert.ok(Math.abs(frame.catcherLean) <= 1);
      assert.ok(Number.isInteger(frame.pendingTotal) && frame.pendingTotal >= 0);
      for (const token of frame.tokens) {
        for (const key of ['x', 'y', 'rotation', 'opacity', 'scale']) assert.ok(Number.isFinite(token[key]), key);
        assert.ok(token.x >= 0 && token.x <= 1000);
        assert.ok(token.y >= 0 && token.y <= 500);
        assert.ok(token.opacity >= 0 && token.opacity <= 1);
        assert.ok(token.scale > 0 && token.scale <= 1);
        assert.ok(Math.abs(token.rotation) <= 5);
      }
    }
  }
});

test('reverse scrolling and other calls cannot change the same scene', () => {
  const positions = Array.from({ length: 101 }, (_, index) => index / 100);
  const forward = positions.map(p => speechFrame(p));
  const reverse = [...positions].reverse().map(p => speechFrame(p)).reverse();
  assert.deepEqual(reverse, forward);
  const changed = speechFrame(0.55);
  changed.caughtCounts.people = 999;
  changed.tokens[0].x = -100;
  assert.deepEqual(speechFrame(0.55), forward[55]);
  assert.deepEqual(speechFrame(-2), speechFrame(0));
  assert.deepEqual(speechFrame(2), speechFrame(1));
  for (const invalid of [NaN, Infinity, -Infinity, undefined, '0.5']) assert.throws(() => speechFrame(invalid), RangeError);
});

test('reduced motion holds one readable pose per step and preserves every count', () => {
  const options = { reducedMotion: true };
  assert.equal(speechFrame(0, options).tokens.length, 0);
  for (let index = 0; index < 10; index += 1) {
    const early = speechFrame((index + 0.2) / 10, options);
    const late = speechFrame((index + 0.8) / 10, options);
    const { progress: earlyProgress, ...earlyPose } = early;
    const { progress: lateProgress, ...latePose } = late;
    assert.deepEqual(earlyPose, latePose);
    assert.equal(early.caughtTotal, index);
    assert.deepEqual(early.caughtCounts, countsOf(speechWords.slice(0, index)));
    assert.equal(early.tokens.length, 1);
    assert.equal(early.tokens[0].index, index);
    assert.equal(early.tokens[0].opacity, 1);
    assert.equal(early.tokens[0].rotation, 0);
    assert.equal(early.mouthOpen + early.gesture + early.catcherLean, 0);
    assert.equal(early.pendingTotal + early.tokens.length + early.caughtTotal, 10);
  }
  const final = speechFrame(1, options);
  assert.deepEqual(final.caughtCounts, speechFrame(1).caughtCounts);
  assert.equal(final.caughtTotal, 10);
  assert.equal(final.tokens.length, 0);
});
