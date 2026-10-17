// An explicitly schematic embedding map, not measured language-model output.
// Every state uses the same two coordinate axes and the same passage identities.
export const passages = Object.freeze([
  { id: 'A', content: 'budget', style: 'plain', shape: 'circle', short: 'Budget · plain',
    text: 'The council cut the budget.', contentCode: -1, styleCode: -1 },
  { id: 'B', content: 'budget', style: 'bureaucratic', shape: 'circle', short: 'Budget · bureaucratic',
    text: 'The council implemented a reduction in budgetary allocation.', contentCode: -1, styleCode: 1 },
  { id: 'C', content: 'bridge', style: 'plain', shape: 'square', short: 'Bridge · plain',
    text: 'The bridge will open next week.', contentCode: 1, styleCode: -1 },
  { id: 'D', content: 'bridge', style: 'bureaucratic', shape: 'square', short: 'Bridge · bureaucratic',
    text: 'The bridge is scheduled to commence operations next week.', contentCode: 1, styleCode: 1 },
]);

export const triplet = Object.freeze({ anchor: 'B', positive: 'D', negative: 'A' });
export const margin = 0.1;
export const initialScale = Object.freeze({ content: 1.15, style: 0.28 });
export const finalScale = Object.freeze({ content: 0.32, style: 0.95 });
export const clamp = (value, low = 0, high = 1) => Math.min(high, Math.max(low, value));
export const mix = (start, end, progress) => start + (end - start) * progress;

export function safePosition(value, maximum = 11) {
  return clamp(Number.isFinite(value) ? value : 0, 0, maximum);
}

export function squaredDistance(a, b) {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
}

export function projectVector(vector, frame) {
  // Equal pixels per unit on both axes preserve the Euclidean geometry even
  // when the available plotting rectangle is wider than it is tall.
  const scale = Math.min(frame.width, frame.height) / 3;
  return [frame.x + frame.width / 2 + vector[0] * scale,
    frame.y + frame.height / 2 - vector[1] * scale];
}

export function tripletLoss(anchor, positive, negative, alpha = margin) {
  return Math.max(0, squaredDistance(anchor, positive) - squaredDistance(anchor, negative) + alpha);
}

export function embeddingState(progress = 0) {
  const t = clamp(progress);
  const content = mix(initialScale.content, finalScale.content, t);
  const style = mix(initialScale.style, finalScale.style, t);
  const vectors = Object.fromEntries(passages.map(passage => [passage.id,
    [passage.contentCode * content, passage.styleCode * style]]));
  const positiveDistance = squaredDistance(vectors[triplet.anchor], vectors[triplet.positive]);
  const negativeDistance = squaredDistance(vectors[triplet.anchor], vectors[triplet.negative]);
  return { progress: t, content, style, vectors, positiveDistance, negativeDistance,
    loss: tripletLoss(vectors.B, vectors.D, vectors.A) };
}

export function stateAt(position) {
  const value = safePosition(position);
  // Re-encoding changes continuously during the two scroll intervals after
  // the parameter-update explanation. These are illustrative coordinates;
  // interpolation is not presented as an optimizer's actual training trace.
  return embeddingState(clamp((value - 7) / 2));
}
