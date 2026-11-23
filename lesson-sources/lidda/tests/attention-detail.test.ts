import assert from "node:assert/strict";
import test from "node:test";
const detailUrl = new URL("../lib/attention-detail.ts", import.meta.url).href;
const { traceAttention, traceWholeModel, DETAIL_VALUES }: typeof import("../lib/attention-detail") = await import(detailUrl);
const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-12, `${a} differs from ${b}`);

test("every query row is normalized and the worked dot products are reproducible", () => {
  const first = traceAttention(0, 0);
  [0.16, 0.74, 0.72, 1.32].forEach((expected, j) => close(first.dots[j], expected));
  for (const h of [0, 1]) for (const r of [0, 1, 2, 3]) {
    const row = traceAttention(h, r);
    close(row.weights.reduce((a, b) => a + b, 0), 1);
    assert.ok(row.weights.every(a => a > 0));
    row.exponents.forEach((value, j) => close(value, Math.exp(row.dots[j] / Math.SQRT2)));
  }
});

test("incoming credit totals eight row units and agrees with the mean readout", () => {
  const whole = traceWholeModel();
  close(whole.incoming.reduce((a, b) => a + b, 0), 8);
  close(whole.credit.reduce((a, b) => a + b, 0), 1);
  close(whole.pooled, whole.credit.reduce((total, a, j) => total + a * DETAIL_VALUES[j], 0));
  assert.equal((whole.probability * 100).toFixed(1), "41.1");
  assert.deepEqual(whole.credit.map(a => (a * 100).toFixed(1)), ["17.6", "24.3", "30.5", "27.6"]);
});

test("invalid head or query selections are rejected", () => {
  for (const h of [-1, 2, .5, NaN]) assert.throws(() => traceAttention(h, 0), RangeError);
  for (const r of [-1, 4, .5, NaN]) assert.throws(() => traceAttention(0, r), RangeError);
});
