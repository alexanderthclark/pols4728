import assert from 'node:assert/strict';
import test from 'node:test';
import { ELECTORAL_MAP } from '../docs/shapley/electoral-map-data.mjs';
import { electoralResults } from '../docs/shapley/electoral-college.mjs';

// Independent planar shoelace calculation; holes have the opposite winding.
function area(rings) {
  return Math.abs(rings.reduce((total, ring) => {
    let twice = 0;
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i], b = ring[(i + 1) % ring.length];
      twice += a[0] * b[1] - b[0] * a[1];
    }
    return total + twice / 2;
  }, 0));
}
const phases = ['geographic', 'electoral', 'adjusted'];
const closeTo = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);

test('the map represents all 50 states and DC with the exact voting-power data', () => {
  const { states } = ELECTORAL_MAP;
  assert.equal(states.length, 51);
  assert.equal(new Set(states.map(state => state.abbreviation)).size, 51);
  const expected = new Map(electoralResults.map(state => [state.abbreviation, state]));
  for (const state of states) {
    const value = expected.get(state.abbreviation);
    assert.ok(value, state.abbreviation);
    assert.equal(state.votes, value.votes);
    closeTo(state.adjustedVotes, 538 * value.share);
    closeTo(state.adjustedVotes, state.votes * value.relativePower);
  }
  assert.equal(states.reduce((sum, state) => sum + state.votes, 0), 538);
  closeTo(states.reduce((sum, state) => sum + state.adjustedVotes, 0), 538);
});

test('state areas encode votes and adjusted votes on a common quantitative scale', () => {
  const { states } = ELECTORAL_MAP;
  const electoralArea = states.reduce((sum, state) => sum + area(state.electoral), 0);
  const adjustedArea = states.reduce((sum, state) => sum + area(state.adjusted), 0);
  assert.ok(electoralArea > 0);
  // A single scale prevents the second morph from magnifying the small premium.
  assert.ok(Math.abs(adjustedArea / electoralArea - 1) < 0.0002);
  for (const state of states) {
    const electoralShare = area(state.electoral) / electoralArea;
    const adjustedShare = area(state.adjusted) / adjustedArea;
    assert.ok(Math.abs(electoralShare / (state.votes / 538) - 1) < 0.001, `${state.abbreviation}: electoral area`);
    assert.ok(Math.abs(adjustedShare / (state.adjustedVotes / 538) - 1) < 0.001, `${state.abbreviation}: adjusted area`);
    const expectedRatio = state.adjustedVotes / state.votes;
    assert.ok(Math.abs(area(state.adjusted) / area(state.electoral) / expectedRatio - 1) < 0.001, `${state.abbreviation}: area premium`);
  }
  const ca = states.find(state => state.abbreviation === 'CA');
  const wy = states.find(state => state.abbreviation === 'WY');
  assert.ok(area(ca.adjusted) > area(ca.electoral));
  assert.ok(area(wy.adjusted) < area(wy.electoral));
});

test('every state preserves its rings and vertex correspondence for reversible morphs', () => {
  for (const state of ELECTORAL_MAP.states) {
    for (const phase of phases) {
      const rings = state[phase];
      assert.ok(rings.length > 0, `${state.abbreviation}: ${phase}`);
      assert.equal(rings.length, state.geographic.length);
      assert.ok(area(rings) > 0);
      rings.forEach((ring, index) => {
        assert.ok(ring.length >= 3);
        assert.equal(ring.length, state.geographic[index].length);
        ring.forEach(point => assert.ok(point.length === 2 && point.every(Number.isFinite)));
      });
      assert.ok(state.labels[phase].every(Number.isFinite));
    }
  }
});

test('shared geographic border vertices remain joined in both cartograms', () => {
  const seen = new Map();
  let shared = 0;
  for (const state of ELECTORAL_MAP.states) {
    state.geographic.forEach((ring, ringIndex) => ring.forEach((point, pointIndex) => {
      const key = point.join(',');
      const prior = seen.get(key);
      if (prior && prior.abbreviation !== state.abbreviation) {
        shared++;
        for (const phase of ['electoral', 'adjusted']) {
          assert.deepEqual(state[phase][ringIndex][pointIndex], prior[phase], `${state.abbreviation}: ${phase} border`);
        }
      } else {
        seen.set(key, { abbreviation: state.abbreviation, electoral: state.electoral[ringIndex][pointIndex], adjusted: state.adjusted[ringIndex][pointIndex] });
      }
    }));
  }
  assert.ok(shared > 100, 'the cartogram retains a substantial shared state-border topology');
});
