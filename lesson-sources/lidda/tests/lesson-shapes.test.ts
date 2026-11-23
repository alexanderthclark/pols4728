import assert from "node:assert/strict";
import test from "node:test";
import { traceLessonShapes } from "../lib/lesson-shapes.ts";
import { traceWholeModel } from "../lib/attention-detail.ts";

const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);
const close = (actual: number, expected: number) =>
  assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} differs from ${expected}`);

test("padding preserves both numerical attention heads and only valid rows sum to one", () => {
  const lesson = traceLessonShapes();
  const original = traceWholeModel();
  assert.equal(lesson.slotCount, 5);
  assert.equal(lesson.heads.length, 2);
  lesson.heads.forEach((matrix, head) => {
    assert.equal(matrix.length, 5);
    matrix.forEach((row) => assert.equal(row.length, 5));
    for (let query = 0; query < 4; query += 1) {
      assert.deepEqual(matrix[query].slice(0, 4), original.traces[head][query].weights);
      close(sum(matrix[query]), 1);
      assert.equal(matrix[query][4], 0);
    }
    assert.deepEqual(matrix[4], [0, 0, 0, 0, 0]);
  });
});

test("column totals conserve eight units and give unit-sum credit with no padded credit", () => {
  const lesson = traceLessonShapes();
  assert.equal(lesson.incoming.length, 5);
  assert.equal(lesson.credit.length, 5);
  const incomingFromHeads = Array.from({ length: 5 }, (_, source) =>
    lesson.heads.reduce((headSum, matrix) =>
      headSum + matrix.reduce((querySum, row) => querySum + row[source], 0), 0,
    ),
  );
  incomingFromHeads.forEach((incoming, source) => {
    close(lesson.incoming[source], incoming);
    close(lesson.credit[source], incoming / 8);
  });
  close(sum(lesson.incoming), 8);
  close(sum(lesson.credit), 1);
  assert.ok(lesson.credit.slice(0, 4).every((share) => share > 0 && share < 1));
  assert.equal(lesson.incoming[4], 0);
  assert.equal(lesson.credit[4], 0);
});

test("flattening a five-by-two representation retains ten slots in row-major order", () => {
  const lesson = traceLessonShapes();
  assert.equal(lesson.valueWidth, 2);
  assert.equal(lesson.flatSlots.length, lesson.slotCount * lesson.valueWidth);
  assert.deepEqual(lesson.flatSlots.slice(0, 8), [
    { position: 1, coordinate: 1, padded: false },
    { position: 1, coordinate: 2, padded: false },
    { position: 2, coordinate: 1, padded: false },
    { position: 2, coordinate: 2, padded: false },
    { position: 3, coordinate: 1, padded: false },
    { position: 3, coordinate: 2, padded: false },
    { position: 4, coordinate: 1, padded: false },
    { position: 4, coordinate: 2, padded: false },
  ]);
  assert.equal(new Set(lesson.flatSlots.slice(0, 8).map((slot) =>
    `${slot.position}:${slot.coordinate}`,
  )).size, 8);
  assert.deepEqual(lesson.flatSlots.slice(8), [
    { position: 5, coordinate: 1, padded: true },
    { position: 5, coordinate: 2, padded: true },
  ]);
});
