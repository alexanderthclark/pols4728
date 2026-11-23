import { traceWholeModel } from "./attention-detail.ts";

/**
 * Shapes for one five-slot teaching journey with four real touches.
 *
 * The attention weights and credit reuse the existing numerical example.
 * The two mixed-output coordinates are symbolic slots, not numerical values
 * derived from that example's scalar-value classifier. Padding receives zero
 * attention and credit; its two output coordinates are retained as zero slots
 * when the fixed-size representation is flattened.
 */
export function traceLessonShapes(): {
  heads: number[][][];
  incoming: number[];
  credit: number[];
  slotCount: 5;
  valueWidth: 2;
  flatSlots: { position: number; coordinate: number; padded: boolean }[];
} {
  const whole = traceWholeModel();
  const heads = whole.traces.map((rows) => [
    ...rows.map((row) => [...row.weights, 0]),
    [0, 0, 0, 0, 0],
  ]);

  return {
    heads,
    incoming: [...whole.incoming, 0],
    credit: [...whole.credit, 0],
    slotCount: 5,
    valueWidth: 2,
    // Mathematical labels are one-based; head-matrix array indices are zero-based.
    flatSlots: Array.from({ length: 10 }, (_, index) => ({
      position: Math.floor(index / 2) + 1,
      coordinate: index % 2 + 1,
      padded: index >= 8,
    })),
  };
}
