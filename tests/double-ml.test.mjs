import assert from 'node:assert/strict';
import test from 'node:test';
import {
  calculateSummary, clamp, groups, lerp, phase, residualizedRows,
  rows, scrollState, smoothstep, summary
} from '../docs/double-ml/model.mjs';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} ≠ ${expected}`);

test('the 32 students retain identities, treatment counts, and the constructed ten-point effect', () => {
  assert.equal(rows.length, 32);
  assert.equal(new Set(rows.map(row => row.id)).size, 32);
  for (const x of [40, 50, 60, 70]) {
    const students = rows.filter(row => row.x === x);
    assert.equal(students.length, 8);
    assert.equal(students.filter(row => row.d === 1).length, x < 60 ? 2 : 6);
    for (const row of students) {
      assert.equal(row.y, 50 + 40 * Number(x >= 60) + 10 * row.d);
      assert.equal(row.y - 10 * row.d, x < 60 ? 50 : 90);
    }
  }
});

test('the observed outcome prediction includes the expected tutoring contribution', () => {
  for (const tier of ['low', 'high']) {
    const students = rows.filter(row => row.tier === tier);
    const observedMean = students.reduce((sum, row) => sum + row.y, 0) / students.length;
    const treatmentRate = students.reduce((sum, row) => sum + row.d, 0) / students.length;
    const baseline = tier === 'low' ? 50 : 90;
    assert.equal(observedMean, students[0].m);
    assert.equal(treatmentRate, students[0].e);
    assert.equal(students[0].m, baseline + 10 * treatmentRate);
  }
});

test('the four displayed pairs use student counts in the residual slope', () => {
  assert.deepEqual(groups.map(({ count, r, t }) => ({ count, r, t })), [
    { count: 12, r: -2.5, t: -0.25 },
    { count: 4, r: 7.5, t: 0.75 },
    { count: 4, r: -7.5, t: -0.75 },
    { count: 12, r: 2.5, t: 0.25 }
  ]);
  assert.equal(groups.reduce((sum, group) => sum + group.count * group.r * group.t, 0), 60);
  assert.equal(groups.reduce((sum, group) => sum + group.count * group.t ** 2, 0), 6);
  assert.equal(summary.sumTR, 60);
  assert.equal(summary.sumT2, 6);
  assert.equal(summary.slope, 10);
  for (const row of rows) assert.equal(row.r, 10 * row.t);
});

test('the raw gap is thirty and subtracting only the outcome prediction leaves 7.5', () => {
  assert.equal(summary.treatedCount, 16);
  assert.equal(summary.controlCount, 16);
  assert.equal(summary.treatedMean, 90);
  assert.equal(summary.controlMean, 60);
  assert.equal(summary.rawGap, 30);
  assert.equal(summary.treatedResidualMean, 3.75);
  assert.equal(summary.controlResidualMean, -3.75);
  assert.equal(summary.outcomeOnlyGap, 7.5);
  assert.deepEqual(calculateSummary([...rows].reverse()), summary);
});

test('vertical and horizontal residualization can be scrolled independently while preserving students', () => {
  const start = residualizedRows(0, 0);
  const middle = residualizedRows(0.5, 0.25);
  const finish = residualizedRows(1, 1);
  assert.deepEqual(start.map(row => row.id), rows.map(row => row.id));
  assert.deepEqual(finish, rows);
  for (let index = 0; index < rows.length; index += 1) {
    const original = rows[index];
    assert.equal(start[index].r, original.y);
    assert.equal(start[index].t, original.d);
    assert.equal(middle[index].r, original.y - 0.5 * original.m);
    assert.equal(middle[index].t, original.d - 0.25 * original.e);
  }
  assert.deepEqual(residualizedRows(-1, 2), residualizedRows(0, 1));
  assert.equal(rows[0].r, -2.5, 'rendering intermediate frames cannot alter source residuals');
});

test('scroll progress is continuous, reversible, and independent of elapsed time', () => {
  const steps = [{ top: 1000, height: 800, id: 'a' }, { top: 1800, height: 1200, id: 'b' }];
  const at = distance => scrollState(1000 - 800 * 0.55 + distance, steps, 800);
  assert.deepEqual(at(-100), { index: 0, id: 'a', progress: 0, motionProgress: 0, overall: 0 });
  close(at(200).progress, 0.25);
  close(at(400).motionProgress, 0.5);
  close(at(401).progress - at(400).progress, 1 / 800);
  assert.equal(at(800).id, 'b');
  assert.equal(at(800).progress, 0);
  close(at(1400).progress, 0.5);
  const before = at(200);
  at(1600);
  assert.deepEqual(at(200), before, 'reverse scrolling reconstructs exactly the previous pose');
  assert.equal(at(4000).progress, 1);
  assert.equal(at(4000).overall, 1);
});

test('reduced motion changes poses at the same reading position and remains reversible', () => {
  const steps = [{ top: 0, height: 1000, id: 'one' }];
  const at = progress => scrollState(progress * 1000, steps, 800, { anchor: 0, reducedMotion: true });
  assert.equal(at(0.49).motionProgress, 0);
  assert.equal(at(0.5).motionProgress, 1);
  assert.equal(at(0.7).motionProgress, 1);
  assert.equal(at(0.2).motionProgress, 0);
  assert.equal(at(0.7).progress, 0.7, 'reading progress is preserved without continuous movement');
  assert.equal(phase(-0.1, true), 0);
  assert.equal(phase(1.1, true), 1);
});

test('section gaps retain the completed diagram and an empty lesson has no active section', () => {
  const steps = [{ top: 100, height: 100, id: 'first' }, { top: 300, height: 100, id: 'next' }];
  const state = scrollState(250, steps, 800, { anchor: 0 });
  assert.equal(state.id, 'first');
  assert.equal(state.progress, 1);
  assert.deepEqual(scrollState(0, [], 800), { index: -1, id: null, progress: 0, motionProgress: 0, overall: 0 });
  assert.throws(() => scrollState(0, steps, 0), RangeError);
  assert.throws(() => scrollState(0, [{ top: 0, height: 0 }], 800), RangeError);
});

test('interpolation stays within the two meaningful poses', () => {
  assert.equal(clamp(-1), 0);
  assert.equal(clamp(2), 1);
  assert.equal(lerp(50, 60, 0.5), 55);
  assert.equal(lerp(50, 60, 2), 60);
  assert.equal(smoothstep(0), 0);
  assert.equal(smoothstep(0.5), 0.5);
  assert.equal(smoothstep(1), 1);
});
