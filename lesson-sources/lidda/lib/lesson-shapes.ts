import { traceWholeModel } from "./attention-detail.ts";
/** Two illustrative heads and one masked PAD slot for a four-touch journey. */
export function traceLessonShapes(): {
  heads: number[][][];
  incoming: number[];
  credit: number[];
  slotCount: 5;
} {
  const whole = traceWholeModel();
  const heads = whole.traces.map((rows) => [
    ...rows.map((row) => [...row.weights, 0]),
    [0, 0, 0, 0, 0],
  ]);
  return { heads, incoming: [...whole.incoming, 0], credit: [...whole.credit, 0], slotCount: 5 };
}
