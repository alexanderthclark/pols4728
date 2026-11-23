// Scroll position is the sole clock; reversing it restores every earlier state.
export function sceneAt(offset, extent, count, reduced = false) {
  const overall = Math.max(0, Math.min(1, offset / Math.max(1, extent)));
  const raw = overall * count;
  const index = Math.min(count - 1, Math.floor(raw));
  const fraction = raw - index;
  return {index, overall, fraction,
    progress: reduced ? 1 : Math.max(0, Math.min(1, (fraction - 0.15) / 0.62))};
}

export function narrativeOpacity(fraction, first = false, reduced = false) {
  if (reduced) return 1;
  const entry = first ? 1 : Math.min(1, fraction / 0.10);
  const exit = Math.min(1, (1 - fraction) / 0.09);
  return Math.max(0, Math.min(entry, exit));
}
