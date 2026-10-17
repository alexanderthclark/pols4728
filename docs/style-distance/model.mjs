// An explicitly schematic embedding map, not measured language-model output.
// Every state uses the same two coordinate axes and the same passage identities.
export const passages = Object.freeze([
  { id: 'A', content: 'uncertain', style: 'biblical', shape: 'circle', short: 'The race is not to the swift',
    text: 'the race is not to the swift', contentCode: -1, styleCode: -1,
    source: 'Ecclesiastes 9:11 · KJV' },
  { id: 'B', content: 'uncertain', style: 'academic', shape: 'circle', short: 'Ability does not ensure success',
    text: '…success or failure in competitive activities exhibits no tendency to be commensurate with innate capacity…',
    contentCode: -1, styleCode: 1, source: 'Orwell · excerpt from his parody' },
  { id: 'C', content: 'certain', style: 'biblical', shape: 'square', short: 'The swift shall win the race',
    text: 'the swift shall win the race', contentCode: 1, styleCode: -1,
    source: 'Constructed counterexample · not scripture' },
  { id: 'D', content: 'certain', style: 'academic', shape: 'square', short: 'Superior speed ensures success',
    text: 'Competitive success accrues to the participant exhibiting the greatest velocity.',
    contentCode: 1, styleCode: 1, source: 'New teaching paraphrase · not Orwell' },
]);

export const triplet = Object.freeze({ anchor: 'A', positive: 'C', negative: 'B' });
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
    loss: tripletLoss(vectors[triplet.anchor], vectors[triplet.positive], vectors[triplet.negative]) };
}

export function stateAt(position) {
  const value = safePosition(position);
  // Re-encoding changes continuously during the two scroll intervals after
  // the parameter-update explanation. These are illustrative coordinates;
  // interpolation is not presented as an optimizer's actual training trace.
  return embeddingState(clamp((value - 7) / 2));
}
