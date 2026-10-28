/** Keep the complete reading version in place unless the animated lesson is ready. */
export async function startLesson({
  page = document,
  loadStory = () => import('./story.mjs?v=20261028-reading-fallback-5'),
  reportError = error => console.error('The animated lesson could not load; the complete reading version remains available.', error),
} = {}) {
  const root = page.querySelector('#chapters');
  const transcript = page.querySelector('#transcript');
  const explorer = page.querySelector('#explore');
  const readingChapters = [...transcript.querySelectorAll('details[data-chapter-id]')];
  let disposeStory;
  try {
    const {initializeStory} = await loadStory();
    disposeStory = initializeStory();
    // Transfer anchors only after initialization, so every ID remains unique.
    readingChapters.forEach(chapter => { chapter.id = `reading-${chapter.dataset.chapterId}`; });
    root.querySelectorAll('.track').forEach(track => { track.id = track.dataset.chapterId; });
    // The static text is a loading/print fallback, not an extra lesson section.
    transcript.hidden = true;
    explorer.hidden = false;
    return true;
  } catch (error) {
    disposeStory?.();
    root.replaceChildren();
    transcript.hidden = false;
    readingChapters.forEach(chapter => { chapter.id = chapter.dataset.chapterId; chapter.open = true; });
    page.querySelector('#compare').before(transcript);
    explorer.hidden = true;
    reportError(error);
    return false;
  }
}

if (typeof document !== 'undefined') {
  startLesson().then(() => {
    if (location.hash) {
      const target = document.getElementById(location.hash.slice(1));
      if (target) requestAnimationFrame(() => target.scrollIntoView());
    }
  });
}
