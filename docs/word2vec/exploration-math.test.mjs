import assert from 'node:assert/strict';
import {
  vectors, candidates, dot, norm, unit, add, scale, cosine,
  query, rank, cosmulScore, rankCosmul,
} from './math.js';

const close = (actual, expected, tolerance = 1e-12) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
};
const vectorClose = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, i) => close(value, expected[i]));
};
const original = JSON.stringify(vectors);
const { water, dirt, rain, mud, clean, dry } = vectors;

close(dot([1, 2, 3], [4, -1, 2]), 8);
close(norm([3, 4]), 5);
vectorClose(unit([3, 4]), [0.6, 0.8]);
vectorClose(add([1, 2], [3, -4]), [4, -2]);
vectorClose(scale([1, -2], 3), [3, -6]);
close(cosine([1, 0], [0, 1]), 0);
close(cosine([1, 0], [-1, 0]), -1);
close(cosine(water, dirt), cosine(dirt, water));
close(cosine(water, dirt), cosine(scale(water, 7), scale(dirt, 0.2)));
assert.throws(() => unit([0, 0]), /no direction/);
assert.throws(() => cosine([0, 0], dirt), /no direction/);
assert.throws(() => dot([1, 2], [3]), /dimensions/);
assert.throws(() => norm([1, NaN]), /finite/);

// Every toy pair satisfies the exact identity for unit-vector distance.
for (const a of Object.values(vectors)) {
  close(norm(unit(a)), 1);
  for (const b of Object.values(vectors)) {
    const difference = add(unit(a), scale(unit(b), -1));
    close(dot(difference, difference), 2 * (1 - cosine(a, b)));
  }
}

// Source normalization is scale-invariant; raw addition retains length weights.
const normalizedQuery = query([dirt, rain]);
const rawQuery = query([dirt, rain], [], { normalizeSources: false });
vectorClose(query([scale(dirt, 5), scale(rain, 0.2)]), normalizedQuery);
vectorClose(rawQuery, [2.6, 1.6]);
assert.ok(cosine(normalizedQuery, rawQuery) < 0.99);
const exclude = ['dirt', 'rain'];
assert.equal(rank(normalizedQuery, candidates, exclude)[0].word, 'puddle');
assert.equal(rank(rawQuery, candidates, exclude)[0].word, 'mud');

// A named query explicitly excludes its inputs; a bare vector has no such
// identity information. The ranking function makes that policy visible.
assert.equal(rank(dirt)[0].word, 'dirt');
assert.ok(rank(dirt, candidates, ['dirt']).every(({ word }) => word !== 'dirt'));
assert.deepEqual(rank(normalizedQuery), rank(scale(normalizedQuery, 2)));
vectorClose(query([water, dirt], [clean, dry]),
  add(add(unit(water), unit(dirt)), scale(add(unit(clean), unit(dry)), -1)));
vectorClose(query([water, dirt], [water]), unit(dirt));
vectorClose(query([water], [water]), [0, 0]);
assert.throws(() => rank(query([water], [water])), /no direction/);

// Hand-calculated shifted-cosine examples verify the product/ratio, including
// the stabilizer for an empty negative list and a nearly vanishing denominator.
const e1 = [1, 0], e2 = [0, 1];
close(cosmulScore(e1, [e1, e2]), 0.5 / (1 + 1e-6));
close(cosmulScore(e1, [e1, e2], [e2]), 0.5 / (0.5 + 1e-6));
close(cosmulScore(e1, [e1], [scale(e1, -1)]), 1e6);
assert.ok(cosmulScore(e1, [e1], [scale(e1, -1)]) > 1,
  'A multiplicative score is not a probability.');
close(cosmulScore(mud, [water, dirt], [dry]),
  cosmulScore(scale(mud, 9), [scale(water, 3), scale(dirt, 0.4)], [scale(dry, 7)]));
close(cosmulScore(mud, [water, dirt]), cosmulScore(mud, [dirt, water]));
assert.deepEqual(rankCosmul([water]).map(item => item.word), rank(water).map(item => item.word));
assert.ok(rankCosmul([water, dirt], [], candidates, ['water', 'dirt'])
  .every(({ word }) => word !== 'water' && word !== 'dirt'));
for (const ranking of [rank(normalizedQuery), rankCosmul([water, dirt], [dry])]) {
  assert.ok(ranking.every(({ score }) => Number.isFinite(score)));
  assert.ok(ranking.every(({ score }, index) => index === 0 || ranking[index - 1].score >= score));
}
assert.throws(() => cosmulScore(mud, []), /positive source/);
assert.equal(JSON.stringify(vectors), original, 'All stored vectors must remain unchanged.');
assert.ok(Object.isFrozen(vectors) && Object.values(vectors).every(Object.isFrozen));

console.log('Word-vector math checks passed: vector algebra, cosine invariance, distance identity, signed queries, raw versus normalized composition, explicit exclusions, multiplicative scoring, and fixed stored data.');
console.log(JSON.stringify({
  normalizedDirtPlusRain: { query: normalizedQuery, top: rank(normalizedQuery, candidates, exclude)[0] },
  rawDirtPlusRain: { query: rawQuery, top: rank(rawQuery, candidates, exclude)[0] },
  waterPlusDirt: rank(query([water, dirt]), candidates, ['water', 'dirt']).slice(0, 3),
  multiplicativeWaterAndDirt: rankCosmul([water, dirt], [], candidates, ['water', 'dirt']).slice(0, 3),
}, null, 2));
