import assert from 'node:assert/strict';
import test from 'node:test';
import { exactWeightedVotingPower, ELECTORAL_STATES, ELECTORAL_QUOTA, ELECTORAL_TOTAL, electoralResults } from '../docs/shapley/electoral-college.mjs';

const closeTo = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-14, `${actual} should equal ${expected}`);

// An independent reference: visit every order and credit its first quota crossing.
function enumeratedPower(weights, quota) {
  const numerators = weights.map(() => 0n);
  let denominator = 0n;
  function visit(prefix, remaining) {
    if (remaining.length) {
      for (const player of remaining) visit([...prefix, player], remaining.filter(other => other !== player));
      return;
    }
    denominator++;
    let total = 0;
    for (const player of prefix) {
      const next = total + weights[player];
      if (total < quota && next >= quota) numerators[player]++;
      total = next;
    }
  }
  visit([], weights.map((_, player) => player));
  return { numerators, denominator };
}

test('exact coalition counting matches independent enumeration of orders', () => {
  for (const [weights, quota] of [
    [[2, 1, 1], 3], [[3, 2, 1, 1], 4], [[3, 2, 1, 1], 5],
    [[5, 2, 2, 1, 0], 6], [[1, 1, 1, 1, 1], 3], [[1, 1], 3],
  ]) {
    const expected = enumeratedPower(weights, quota);
    const actual = exactWeightedVotingPower(weights, quota);
    assert.equal(actual.totalOrders, expected.denominator);
    assert.deepEqual(actual.values.map(value => value.numerator), expected.numerators);
    assert.ok(actual.values.every(value => value.denominator === expected.denominator));
  }
  const { shares } = exactWeightedVotingPower([2, 1, 1], 3);
  [2 / 3, 1 / 6, 1 / 6].forEach((expected, player) => closeTo(shares[player], expected));
});

test('equal players, a dictator, and a dummy have their expected voting power', () => {
  const equal = exactWeightedVotingPower([1, 1, 1, 1, 1], 3);
  equal.shares.forEach(share => closeTo(share, 1 / 5));
  assert.ok(equal.values.every(value => value.numerator === equal.values[0].numerator));
  assert.deepEqual(exactWeightedVotingPower([3, 1, 0], 3).shares, [1, 0, 0]);
  assert.deepEqual(exactWeightedVotingPower([0, 0], 1).shares, [0, 0]);
});

test('the 51-bloc Electoral College calculation reproduces every stored exact result', () => {
  assert.equal(ELECTORAL_STATES.length, 51);
  assert.equal(new Set(ELECTORAL_STATES.map(state => state.abbreviation)).size, 51);
  assert.equal(ELECTORAL_STATES.reduce((sum, state) => sum + state.votes, 0), ELECTORAL_TOTAL);
  assert.equal(ELECTORAL_QUOTA, 270);
  const computed = exactWeightedVotingPower(ELECTORAL_STATES.map(state => state.votes), ELECTORAL_QUOTA);
  assert.equal(computed.values.reduce((sum, value) => sum + value.numerator, 0n), computed.totalOrders);
  const sameWeight = new Map();
  computed.values.forEach((value, player) => {
    const state = electoralResults[player];
    assert.equal(value.numerator.toString(), state.numerator, state.name);
    assert.equal(value.denominator.toString(), state.denominator, state.name);
    closeTo(value.share, state.share);
    closeTo(state.voteShare, state.votes / ELECTORAL_TOTAL);
    closeTo(state.relativePower, state.share / state.voteShare);
    if (sameWeight.has(state.votes)) assert.equal(value.numerator, sameWeight.get(state.votes));
    sameWeight.set(state.votes, value.numerator);
  });
  closeTo(electoralResults.reduce((sum, state) => sum + state.share, 0), 1);
});

test('California and Texas retain independently calculated exact rational shares', () => {
  for (const [abbreviation, numerator, denominator, share, relativePower] of [
    ['CA', 111603651861685256303n, 1033014834748665568800n, 0.10803683365189874, 1.0763669723096578],
    ['TX', 29994201953488676597n, 387380563030749588300n, 0.07742825741906878, 1.041410062286475],
  ]) {
    const state = electoralResults.find(state => state.abbreviation === abbreviation);
    assert.equal(BigInt(state.numerator) * denominator, numerator * BigInt(state.denominator));
    closeTo(state.share, share);
    closeTo(state.relativePower, relativePower);
    assert.ok(state.share > state.voteShare);
  }
  const wyoming = electoralResults.find(state => state.abbreviation === 'WY');
  assert.ok(wyoming.share < wyoming.voteShare);
  closeTo(wyoming.relativePower, 0.9688086213323499);
});

test('invalid voting games are rejected before exact counting', () => {
  for (const weights of [[], [-1, 2], [0.5, 1], [Infinity, 1]]) {
    assert.throws(() => exactWeightedVotingPower(weights, 1), /voting weights/);
  }
  for (const quota of [0, -1, 1.5, Infinity]) {
    assert.throws(() => exactWeightedVotingPower([1, 1], quota), /quota/);
  }
});
