import assert from 'node:assert/strict';
import test from 'node:test';
import { clamp01, easeInOut, phase, lerp, createTimeline } from '../docs/quiz/motion.mjs';

const closeTo = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} should equal ${expected}`);

function setup({ duration = 1000, cancelWorks = true } = {}) {
  let time = 0;
  let nextId = 0;
  const frames = new Map();
  const renders = [];
  const states = [];
  const timeline = createTimeline({
    duration,
    render: progress => renders.push(progress),
    onState: state => states.push(state),
    now: () => time,
    requestFrame(callback) { const id = nextId++; frames.set(id, callback); return id; },
    cancelFrame(id) { if (cancelWorks) frames.delete(id); },
  });
  return {
    timeline, renders, states,
    pending: () => frames.size,
    elapse(milliseconds) { time += milliseconds; },
    advance(milliseconds = 0) {
      time += milliseconds;
      for (const [id, callback] of [...frames]) {
        frames.delete(id);
        callback(time);
      }
    },
  };
}

test('math helpers clamp phases and ease symmetrically without changing interpolation units', () => {
  assert.deepEqual([-Infinity, -1, 0, 0.4, 1, 2, Infinity].map(clamp01), [0, 0, 0, 0.4, 1, 1, 1]);
  assert.deepEqual([-1, 0, 0.5, 1, 2].map(easeInOut), [0, 0, 0.5, 1, 1]);
  closeTo(easeInOut(0.25), 0.15625);
  for (const p of [0.1, 0.2, 0.6, 0.95]) closeTo(easeInOut(p) + easeInOut(1 - p), 1);
  for (const [p, expected] of [[0, 0], [0.2, 0], [0.4, 0.5], [0.6, 1], [1, 1]]) {
    closeTo(phase(p, 0.2, 0.6), expected);
  }
  assert.equal(phase(0.5), 0.5);
  assert.equal(phase(0.49, 0.5, 0.5), 0);
  assert.equal(phase(0.5, 0.5, 0.5), 1);
  assert.throws(() => phase(0.5, 0.7, 0.3), /at least/);
  assert.equal(lerp(100, 300, 0.25), 150);
  assert.equal(lerp(10, -10, 0.5), 0);
});

test('play renders its starting position, advances, and completes once without another frame', () => {
  const ui = setup();
  assert.deepEqual(ui.timeline.getState(), { status: 'idle', progress: 0 });
  assert.equal(ui.pending(), 0);
  ui.timeline.play();
  assert.deepEqual(ui.renders, [0]);
  assert.equal(ui.pending(), 1);
  ui.advance(250);
  assert.deepEqual(ui.timeline.getState(), { status: 'playing', progress: 0.25 });
  ui.advance(900);
  assert.deepEqual(ui.renders, [0, 0.25, 1]);
  assert.deepEqual(ui.timeline.getState(), { status: 'complete', progress: 1 });
  assert.equal(ui.pending(), 0);
  ui.timeline.finish();
  ui.timeline.pause();
  ui.timeline.resume();
  ui.advance(10000);
  assert.deepEqual(ui.renders, [0, 0.25, 1]);
  const snapshot = ui.timeline.getState();
  snapshot.progress = 0;
  ui.states[0].status = 'cancelled';
  assert.deepEqual(ui.timeline.getState(), { status: 'complete', progress: 1 });
});

test('pause includes time since the last frame and resume excludes time spent paused', () => {
  const ui = setup();
  ui.timeline.play();
  ui.advance(200);
  ui.elapse(50);
  ui.timeline.pause();
  assert.deepEqual(ui.timeline.getState(), { status: 'paused', progress: 0.25 });
  assert.equal(ui.pending(), 0);
  ui.advance(10000);
  ui.timeline.resume();
  ui.advance(250);
  assert.deepEqual(ui.timeline.getState(), { status: 'playing', progress: 0.5 });
  ui.advance(500);
  assert.deepEqual(ui.timeline.getState(), { status: 'complete', progress: 1 });
});

test('uncancellable queued callbacks cannot revive a paused or cancelled run', () => {
  const ui = setup({ cancelWorks: false });
  ui.timeline.play();
  ui.elapse(100);
  ui.timeline.pause();
  const pausedRenders = ui.renders.length;
  ui.advance(1000);
  assert.equal(ui.renders.length, pausedRenders);
  ui.timeline.resume();
  ui.timeline.cancel();
  ui.advance(1000);
  assert.deepEqual(ui.timeline.getState(), { status: 'cancelled', progress: 0.1 });
  assert.equal(ui.pending(), 0);
});

test('a new play invalidates an old callback and can restart after cancellation or completion', () => {
  const ui = setup({ cancelWorks: false });
  ui.timeline.play();
  ui.advance(100);
  ui.timeline.cancel();
  ui.timeline.play({ from: 0.4 });
  ui.advance(100);
  assert.deepEqual(ui.renders, [0, 0.1, 0.4, 0.5]);
  assert.equal(ui.pending(), 1);
  ui.advance(500);
  assert.equal(ui.timeline.getState().status, 'complete');
  ui.timeline.play();
  assert.deepEqual(ui.timeline.getState(), { status: 'playing', progress: 0 });
  ui.advance(100);
  closeTo(ui.renders.at(-1), 0.1);
});

test('finish invalidates queued work and also works without browser clocks or frame APIs', () => {
  const ui = setup({ cancelWorks: false });
  ui.timeline.play();
  ui.advance(100);
  ui.timeline.finish();
  ui.advance(200);
  assert.deepEqual(ui.renders, [0, 0.1, 1]);
  const renders = [];
  const timeline = createTimeline({
    render: p => renders.push(p),
    requestFrame: undefined,
    cancelFrame: undefined,
    now: () => { throw new Error('clock should not be read'); },
  });
  timeline.finish();
  timeline.finish();
  assert.deepEqual(renders, [1]);
  assert.deepEqual(timeline.getState(), { status: 'complete', progress: 1 });
});

test('duration and start values are bounded, and zero duration completes immediately', () => {
  for (const duration of [-1, Infinity, NaN]) {
    assert.throws(() => createTimeline({ duration, render() {} }), /finite and nonnegative/);
  }
  assert.throws(() => createTimeline(), /render must/);
  const ui = setup();
  assert.throws(() => ui.timeline.play({ from: NaN }), /from must be finite/);
  assert.deepEqual(ui.timeline.getState(), { status: 'idle', progress: 0 });
  ui.timeline.play({ from: -2 });
  assert.equal(ui.renders.at(-1), 0);
  ui.timeline.play({ from: 2 });
  assert.deepEqual(ui.timeline.getState(), { status: 'complete', progress: 1 });
  assert.equal(ui.pending(), 0);
  const instant = setup({ duration: 0 });
  instant.timeline.play();
  assert.deepEqual(instant.renders, [1]);
  assert.equal(instant.pending(), 0);
});

test('clock regressions do not reverse an animation, and pausing after its deadline completes it', () => {
  const ui = setup();
  ui.timeline.play();
  ui.advance(500);
  ui.advance(-200);
  assert.equal(ui.renders.at(-1), 0.5);
  ui.elapse(1000);
  ui.timeline.pause();
  assert.deepEqual(ui.timeline.getState(), { status: 'complete', progress: 1 });
  assert.equal(ui.pending(), 0);
});
