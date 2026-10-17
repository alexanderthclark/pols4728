// Scroll alone determines the scene, including when the reader moves backward.
export const speechWords = Object.freeze([
  'government', 'of', 'the', 'people', 'by', 'the', 'people', 'for', 'the', 'people',
]);

// All coordinates use the illustration's 1000 × 500 viewBox. Words follow one
// spacious arc, with time between them so they read as separate pieces of text.
export const speechPath = Object.freeze({
  mouth: Object.freeze([224, 257]),
  rise: Object.freeze([430, 45]),
  fall: Object.freeze([650, 85]),
  bag: Object.freeze([785, 285]),
});

const firstEmission = 0.025;
const wordInterval = 0.08;
const flightDuration = 0.23;

const clamp = value => Math.max(0, Math.min(1, value));
const smooth = value => {
  const x = clamp(value);
  return x * x * (3 - 2 * x);
};

function pointOnArc(t) {
  const s = 1 - t;
  return [0, 1].map(k =>
    s ** 3 * speechPath.mouth[k]
    + 3 * s ** 2 * t * speechPath.rise[k]
    + 3 * s * t ** 2 * speechPath.fall[k]
    + t ** 3 * speechPath.bag[k]);
}

function countsForPrefix(length) {
  const counts = Object.fromEntries([...new Set(speechWords)].map(word => [word, 0]));
  for (const word of speechWords.slice(0, length)) counts[word] += 1;
  return counts;
}

function frameWithCounts(progress, caughtTotal, tokens, pose = {}) {
  return {
    progress,
    mouthOpen: 0,
    gesture: 0,
    catcherLean: 0,
    ...pose,
    bagFullness: caughtTotal / speechWords.length,
    tokens,
    caughtWords: speechWords.slice(0, caughtTotal),
    caughtCounts: countsForPrefix(caughtTotal),
    caughtTotal,
    pendingTotal: speechWords.length - caughtTotal - tokens.length,
    total: speechWords.length,
  };
}

/**
 * Return a complete scene at a normalized scroll position.
 *
 * mouthOpen is 0–1; gesture and catcherLean are small normalized pose amounts.
 * The renderer chooses how many pixels/degrees those pose amounts represent.
 * Every word is either pending, airborne, or caught, including repeated words.
 * Reduced motion changes only at tenths of the story: one stationary readable
 * word accompanies each count, without mouth, arm, or catcher animation.
 */
export function speechFrame(progress, { reducedMotion = false } = {}) {
  if (!Number.isFinite(progress)) throw new RangeError('Progress must be a finite number.');
  const p = clamp(progress);

  if (reducedMotion) {
    const caughtTotal = Math.floor(p * speechWords.length);
    const tokens = p > 0 && caughtTotal < speechWords.length
      ? [{ index: caughtTotal, word: speechWords[caughtTotal], x: 490, y: 150, rotation: 0, opacity: 1, scale: 1 }]
      : [];
    return frameWithCounts(p, caughtTotal, tokens);
  }

  const tokens = [];
  let caughtTotal = 0;
  let mouthOpen = 0;
  let catcherLean = 0;

  speechWords.forEach((word, index) => {
    const emission = firstEmission + index * wordInterval;
    const capture = emission + flightDuration;
    const age = p - emission;

    // A modest open/close cycle reads as talking. The head stays upright.
    const syllable = age / 0.052;
    if (syllable > 0 && syllable < 1) {
      mouthOpen = Math.max(mouthOpen, 0.7 * Math.sin(Math.PI * syllable) ** 2);
    }

    // The catcher anticipates each arrival, then settles quickly afterward.
    const catchDistance = p < capture ? (capture - p) / 0.04 : (p - capture) / 0.014;
    if (catchDistance < 1) catcherLean = Math.max(catcherLean, 0.65 * (1 - smooth(catchDistance)));

    if (p >= capture) {
      caughtTotal += 1;
    } else if (age >= 0) {
      const t = age / flightDuration;
      const [x, y] = pointOnArc(t);
      tokens.push({
        index,
        word,
        x,
        y,
        rotation: (index % 2 ? 1 : -1) * 5 * Math.sin(Math.PI * t),
        opacity: smooth(t / 0.08) * smooth((1 - t) / 0.08),
        scale: 0.18 + 0.82 * smooth(t / 0.2) - 0.25 * smooth((t - 0.82) / 0.18),
      });
    }
  });

  const gestureEnvelope = smooth((p - firstEmission) / 0.06) * smooth((0.84 - p) / 0.09);
  const gesture = 0.65 * Math.sin(p * Math.PI * 4) * gestureEnvelope;
  return frameWithCounts(p, caughtTotal, tokens, { mouthOpen, gesture, catcherLean });
}
