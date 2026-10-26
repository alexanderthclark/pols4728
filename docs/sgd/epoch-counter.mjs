// One epoch is one complete pass through the data, not one parameter update.
export function epochState(completedUpdates, totalEpochs = 300, batchesPerEpoch = 32) {
  if (!Number.isInteger(totalEpochs) || totalEpochs < 1 || !Number.isInteger(batchesPerEpoch) || batchesPerEpoch < 1) {
    throw new RangeError('Epoch and batch totals must be positive integers.');
  }
  if (!Number.isFinite(completedUpdates)) throw new TypeError('Update progress must be finite.');
  const totalUpdates = totalEpochs * batchesPerEpoch;
  if (!Number.isSafeInteger(totalUpdates)) throw new RangeError('The update total must be a safe integer.');
  // An interpolated update is only counted once the parameter step is complete.
  const usedUpdates = Math.floor(Math.max(0, Math.min(totalUpdates, completedUpdates)));
  const completedEpochs = Math.floor(usedUpdates / batchesPerEpoch);
  const finished = usedUpdates === totalUpdates;
  const completedBatches = finished ? batchesPerEpoch : usedUpdates % batchesPerEpoch;
  return {
    currentEpoch: finished ? totalEpochs : completedEpochs + 1,
    completedEpochs,
    batch: finished ? batchesPerEpoch : usedUpdates % batchesPerEpoch + 1,
    completedBatches,
    progress: completedBatches / batchesPerEpoch,
    finished,
    usedUpdates,
    totalEpochs,
    batchesPerEpoch,
    totalUpdates
  };
}

const rendered = new WeakMap();

// The track is decorative; every count remains selectable, native HTML text.
export function renderEpochCounter(element, state) {
  const value = epochState(state.usedUpdates, state.totalEpochs, state.batchesPerEpoch);
  const signature = `${value.usedUpdates}:${value.totalEpochs}:${value.batchesPerEpoch}`;
  if (rendered.get(element) === signature) return value;
  rendered.set(element, signature);
  const pad = number => String(number).padStart(2, '0');
  element.classList.add('epoch-counter');
  element.dataset.epoch = String(value.currentEpoch);
  element.dataset.batch = String(value.batch);
  element.dataset.completedEpochs = String(value.completedEpochs);
  element.dataset.completedBatches = String(value.completedBatches);
  element.dataset.completedUpdates = String(value.usedUpdates);
  element.dataset.finished = String(value.finished);
  element.style.setProperty('--epoch-progress', String(value.progress));
  element.setAttribute('aria-label', value.finished
    ? `Training complete: ${value.totalEpochs} epochs and ${value.totalUpdates.toLocaleString('en-US')} updates.`
    : `Epoch ${value.currentEpoch} of ${value.totalEpochs}: ${value.completedBatches} of ${value.batchesPerEpoch} batches used in this pass. ${value.completedEpochs} complete epochs.`);
  element.innerHTML = `<div class="epoch-counter__heading"><span class="epoch-counter__flag" aria-hidden="true"></span><span class="epoch-counter__label">Epoch</span></div>
    <div class="epoch-counter__count"><span class="epoch-counter__number">${pad(value.currentEpoch)}</span><span class="epoch-counter__total"> / ${pad(value.totalEpochs)}</span></div>
    <div class="epoch-counter__batch">${pad(value.completedBatches)} / ${pad(value.batchesPerEpoch)} ${value.finished ? 'complete' : 'batches used'}</div>
    <div class="epoch-counter__track" aria-hidden="true"><span class="epoch-counter__marker"></span></div>`;
  return value;
}
