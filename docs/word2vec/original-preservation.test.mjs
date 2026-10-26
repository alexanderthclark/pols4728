import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const testDirectory = dirname(fileURLToPath(import.meta.url));
// Local artifact layout uses ./dist; the published repository can keep this
// test beside index.html. An optional directory argument supports other layouts.
const possibleRoots = process.argv[2]
  ? [resolve(process.cwd(), process.argv[2])]
  : [join(testDirectory, 'dist'), testDirectory];
const siteRoot = possibleRoots.find(path => existsSync(join(path, 'index.html')));
assert.ok(siteRoot, `Could not find index.html in: ${possibleRoots.join(', ')}`);
const baseline = JSON.parse(readFileSync(join(testDirectory, 'preservation.json'), 'utf8'));
assert.equal(baseline.schemaVersion, 1);
const hash = value => createHash('sha256').update(value).digest('hex');

for (const [file, expected] of Object.entries(baseline.files)) {
  assert.equal(hash(readFileSync(join(siteRoot, file))), expected,
    `Original asset changed: ${file}. Append the exploration without rewriting this asset.`);
}

const html = readFileSync(join(siteRoot, 'index.html'), 'utf8');
const attributes = tag => Object.fromEntries([...tag.matchAll(/([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)]
  .map(([, name, doubleQuoted, singleQuoted]) => [name, doubleQuoted ?? singleQuoted]));
const classList = tag => (attributes(tag).class || '').split(/\s+/);

function section(predicate, label) {
  const tags = [...html.matchAll(/<\/?section\b[^>]*>/gi)];
  const first = tags.findIndex(tag => !tag[0].startsWith('</') && predicate(tag[0]));
  assert.ok(first >= 0, `Missing section: ${label}`);
  let depth = 0;
  for (let i = first; i < tags.length; i += 1) {
    depth += tags[i][0].startsWith('</') ? -1 : 1;
    if (depth === 0) {
      const start = tags[first].index, end = tags[i].index + tags[i][0].length;
      return { start, end, content: html.slice(start, end) };
    }
  }
  assert.fail(`Unclosed section: ${label}`);
}

const sections = {
  hero: section(tag => classList(tag).includes('hero'), 'original hero'),
  story: section(tag => attributes(tag).id === 'story', 'original 15-scene story'),
  afterword: section(tag => attributes(tag).id === 'references', 'original afterword'),
};
for (const [name, current] of Object.entries(sections)) {
  const saved = baseline.sections[name];
  assert.equal(hash(saved.content), saved.sha256, `Invalid saved baseline for ${name}.`);
  assert.equal(hash(current.content), saved.sha256,
    `Original ${name} section changed. Preserve its exact content and append after it.`);
}
assert.ok(sections.hero.end <= sections.story.start && sections.story.end <= sections.afterword.start,
  'The original title, story, and afterword must remain in their original order.');

const exploration = section(tag => attributes(tag).id === 'exploration-story', 'appended exploration');
assert.ok(exploration.start >= sections.afterword.end,
  'The new exploration must begin after the complete original afterword.');

const openingTags = [...html.matchAll(/<([A-Za-z][\w:-]*)\b[^>]*>/g)].map(match => match[0]);
assert.equal(openingTags.filter(tag => classList(tag).includes('hero')).length, 1,
  'Keep exactly one original hero/title scene.');
const seenIds = new Set();
for (const tag of openingTags) {
  const id = attributes(tag).id;
  if (!id) continue;
  assert.ok(!seenIds.has(id), `Duplicate HTML id: ${id}`);
  seenIds.add(id);
}
const moduleSources = openingTags
  .filter(tag => /^<script\b/i.test(tag))
  .map(attributes)
  .filter(({ type }) => type === 'module')
  .map(({ src = '' }) => src.replace(/^\.\//, ''));
assert.ok(moduleSources.includes('app.js'), 'The original app.js must still load.');
assert.ok(moduleSources.includes('exploration.js'), 'The appended exploration.js must load.');
assert.equal(moduleSources.filter(src => src === 'app.js').length, 1);
assert.equal(moduleSources.filter(src => src === 'exploration.js').length, 1);

console.log('Original-preservation checks passed: all four original assets are byte-identical; the hero, 15-scene story markup, and afterword are unchanged; the exploration follows them; there is one hero, unique IDs, and both independent modules.');
