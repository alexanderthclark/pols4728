# Course design and authoring

- Read `design/README.md` before changing visual presentation or creating a new
  page, figure, or illustration. Use `design/guide.pdf` as the visual reference
  and `design/tokens.json` for palette roles. Apply print specifications to print
  documents and the guide's web guidance to interactive pages.
- Every course visual explainer must be a working scrollytelling experience
  unless the user explicitly requests otherwise. Prefer a pinned main graph or
  plot while brief text floats in and out with scrolling. Animate meaningful
  changes within the pinned visual, with reversible visual and text progression.
  Storyboards and prototypes must demonstrate the
  motion, not merely describe it. When depicting people, use the established
  round, torso-less figures. Use spare scene text and substantial explanatory
  animation. Begin explainers with a title scene, usually showing the torso-less
  figures acting out a metaphor for the lesson, such as adjusting a picture frame
  or hammering out a sine wave. Keep the title as real text and preserve existing
  deck and registered course-cover conventions. Make scrolling the complete
  default route and preferably the only storytelling input; do not
  require toggles, sliders, clicks, or hover to reach a teaching point. Reserve
  optional controls for purposeful exploration after the scroll sequence.
  Follow "Scrollytelling and interactive explainers" in `design/README.md`.
- Verify actual forward and backward scrolling on desktop and narrow mobile
  screens, along with keyboard access and reduced motion, before claiming the
  explainer is complete. Standalone lessons must not refer to their source
  assignment, quiz, or question numbers.
- Preserve the instructor's voice, substantive reasoning, assumptions, and
  qualifications. Use connected explanations and useful lists; introduce
  notation when it serves the argument.
- Use white backgrounds, readable type, precise quantitative geometry, direct
  labels, and visual cues that remain meaningful without color. Inspect new
  interactive pages on desktop and narrow mobile screens; support keyboard
  controls and reduced motion.
- Use approved artwork recorded in `design/assets.json`. Preserve proportions,
  monochrome colors, and clear space. Register new artwork with its role,
  description, and hash; keep image-generation prompts out of version control.
- Always include an opening title slide in slide decks unless the user explicitly
  asks to omit it. Preserve the title-slide convention when matching a deck.
- Never use Airbnb examples unless the user explicitly proposes them.
- Copy only requested course files from the private semester repository. Never
  merge its Git history or publish unrelated private materials.
- Keep the shared guide text, PDF, and tokens synchronized with `pols4728-2026f`.
  Update the private source when editing shared design rules, then use its
  `make design-export PUBLIC_DESIGN_REPO=/path/to/pols4728` command to refresh
  the public copy. Its `design-export-check` target checks for drift. Preserve
  the public introduction and artwork registry, and review and commit changes
  in each repository separately.
