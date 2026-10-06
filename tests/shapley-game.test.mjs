import assert from 'node:assert/strict';
import test from 'node:test';
import { defineGame, majority, matchingOrders } from '../docs/shapley/game.mjs';

const closeTo = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} should equal ${expected}`);

test('majority voting enumerates the full three-player coalition lattice', () => {
  assert.equal(majority.nodes.length, 8);
  assert.equal(majority.edges.length, 12);
  assert.equal(majority.orders.length, 6);
  assert.deepEqual(majority.nodes.map(node => node.value), [0, 0, 0, 1, 0, 1, 1, 1]);
  for (const order of majority.orders) {
    assert.equal(order.path.length, 3);
    assert.equal(order.path[0].from, 0);
    assert.equal(order.path.at(-1).to, 7);
    assert.equal(new Set(order.path.map(edge => edge.player)).size, 3);
    for (let i = 1; i < order.path.length; i++) assert.equal(order.path[i].from, order.path[i - 1].to);
  }
});

test('every weight agrees with the frequency of its exact joining edge', () => {
  for (const edge of majority.edges) {
    const matches = matchingOrders(majority, edge.id);
    assert.equal(matches.length, edge.count, edge.id);
    closeTo(matches.length / majority.orders.length, edge.weight);
    const expected = majority.orders.filter(order => order.order.slice(0, order.order.indexOf(edge.playerIndex))
      .reduce((mask, playerIndex) => mask | (1 << playerIndex), 0) === edge.from);
    assert.deepEqual(matches.map(order => order.label), expected.map(order => order.label));
  }
  assert.throws(() => matchingOrders(majority, 'missing'), /Unknown joining edge/);
});

test('each voter’s joining edges partition all six orders', () => {
  majority.players.forEach((_, playerIndex) => {
    const edges = majority.edges.filter(edge => edge.playerIndex === playerIndex);
    assert.deepEqual(edges.map(edge => matchingOrders(majority, edge.id).length).sort(), [1, 1, 2, 2]);
    const labels = edges.flatMap(edge => matchingOrders(majority, edge.id).map(order => order.label));
    assert.equal(labels.length, 6);
    assert.equal(new Set(labels).size, 6);
    closeTo(edges.reduce((sum, edge) => sum + edge.weight, 0), 1);
    const average = majority.orders.reduce((sum, order) => sum + order.path.find(edge => edge.playerIndex === playerIndex).delta, 0) / 6;
    closeTo(majority.shares[playerIndex], average);
    closeTo(majority.shares[playerIndex], 1 / 3);
  });
});

test('A’s two zero-contribution edges select distinct pairs of paths', () => {
  assert.equal(majority.edges.find(edge => edge.id === '0-1').delta, 0);
  assert.equal(majority.edges.find(edge => edge.id === '6-7').delta, 0);
  assert.deepEqual(matchingOrders(majority, '0-1').map(order => order.label), ['A → B → C', 'A → C → B']);
  assert.deepEqual(matchingOrders(majority, '6-7').map(order => order.label), ['B → C → A', 'C → B → A']);
  assert.deepEqual(matchingOrders(majority, '2-3').map(order => order.label), ['B → A → C']);
  assert.deepEqual(matchingOrders(majority, '4-5').map(order => order.label), ['C → A → B']);
});

test('factorial factors count independent orders before and after a joining voter', () => {
  const arrangementCounts = [1, 1, 2]; // 0!, 1!, and 2! for this three-voter game.
  for (const edge of majority.edges) {
    const matches = matchingOrders(majority, edge.id);
    const split = matches.map(({ order }) => {
      const at = order.indexOf(edge.playerIndex);
      return { before: order.slice(0, at).join(','), after: order.slice(at + 1).join(',') };
    });
    const before = new Set(split.map(order => order.before));
    const after = new Set(split.map(order => order.after));
    const size = majority.nodes[edge.from].size;
    assert.equal(before.size, arrangementCounts[size], `${edge.id}: orders before`);
    assert.equal(after.size, arrangementCounts[majority.players.length - size - 1], `${edge.id}: orders after`);
    const pairs = new Set(split.map(order => `${order.before}|${order.after}`));
    for (const prefix of before) for (const suffix of after) assert.ok(pairs.has(`${prefix}|${suffix}`));
    assert.equal(before.size * after.size, edge.count);
    closeTo(before.size * after.size / majority.totalOrders, edge.weight);
  }
});

test('path frequencies support a game with unequal voting power', () => {
  const votes = { A: 2, B: 1, C: 1 };
  const weighted = defineGame({ id: 'weighted-vote', name: 'Weighted voting', players: ['A', 'B', 'C'],
    value: members => Number(members.reduce((sum, player) => sum + votes[player], 0) >= 3) });
  [2 / 3, 1 / 6, 1 / 6].forEach((expected, i) => closeTo(weighted.shares[i], expected));
  closeTo(weighted.shares.reduce((sum, share) => sum + share, 0), weighted.gain);
  for (const edge of weighted.edges) {
    assert.equal(matchingOrders(weighted, edge.id).length, edge.count);
    assert.deepEqual(matchingOrders(weighted, edge.id).map(order => order.label), matchingOrders(majority, edge.id).map(order => order.label));
  }
});
