import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const picture = require('../docs/shap/title-picture.js');
const assetPath = 'docs/shap/assets/prediction-investigators-motion.svg';
const asset = readFileSync(new URL(`../${assetPath}`, import.meta.url), 'utf8');

test('the no-JavaScript and print artwork matches the completed reduced-motion pose', () => {
  assert.equal(asset.trim(), picture.render(0, 1, true));
  const html = readFileSync(new URL('../docs/shap/index.html', import.meta.url), 'utf8');
  assert.match(html, /id="shap-title-fallback" src="assets\/prediction-investigators-motion\.svg"/);
  assert.match(html, /id="shap-title-picture"[^>]* hidden/);
  assert.match(html, /href="#shapley-formula">Start the explanation/);
});

test('the approved-artwork registry identifies the exact static fallback', () => {
  const registry = JSON.parse(readFileSync(new URL('../design/assets.json', import.meta.url), 'utf8'));
  const entry = registry.assets.find(item => item.path === assetPath);
  assert.ok(entry, 'Register the new title artwork');
  assert.equal(entry.sha256, createHash('sha256').update(asset).digest('hex'));
});
