import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const root = new URL('../docs/imbalanced-data/', import.meta.url);
const context = vm.createContext({window: {}});
for (const file of ['tree-model.js', 'tree-colors.js', 'tree-leaves.js', 'tree-rebalance.js', 'tree-entropy-chords.js']) {
  vm.runInContext(fs.readFileSync(new URL(file, root), 'utf8'), context);
}
const attributes = tag => Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => [match[1], match[2]]));
const close = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
const entropy = (positive, negative) => [positive, negative].filter(n => n > 0)
  .reduce((sum, n) => sum - n / (positive + negative) * Math.log(n / (positive + negative)), 0);

test('rendered entropy chords retain the correct child balances and weighted heights during thinning', () => {
  for (const retention of [1, .1, 1 / 99]) {
    const p = 100 / (100 + 9900 * retention);
    const counts = {A: [50, 0, 50, 9900 * retention], B: [90, 990 * retention, 10, 8910 * retention]};
    for (const narrow of [false, true]) {
      const thinning = attributes(context.window.TreeRebalance.figure(retention, narrow).match(/<svg\b[^>]*>/)[0]);
      close(Number(thinning['data-positive-count']), 100);
      close(Number(thinning['data-negative-count']), 9900 * retention);
      close(Number(thinning['data-parent-positive-share']), p);
      const svg = context.window.TreeEntropyChords.figure(p, narrow);
      const chords = [...svg.matchAll(/<path\b[^>]*class="entropy-chord"[^>]*>/g)];
      assert.equal(chords.length, 2);
      for (const match of chords) {
        const chord = attributes(match[0]), name = chord['data-split'];
        const [lp, ln, rp, rn] = counts[name], n = lp + ln + rp + rn;
        const childEntropy = (lp + ln) / n * entropy(lp, ln) + (rp + rn) / n * entropy(rp, rn);
        close(Number(chord['data-left-share']), lp / (lp + ln));
        close(Number(chord['data-right-share']), rp / (rp + rn));
        close(Number(chord['data-child-entropy']), childEntropy);
        close(Number(chord['data-gain']), entropy(lp + rp, ln + rn) - childEntropy);
        assert.equal(chord.stroke, context.window.TreeColors[name]);
        const parent = [...svg.matchAll(/<circle\b[^>]*class="entropy-chord-parent"[^>]*>/g)]
          .map(item => attributes(item[0])).find(item => item['data-split'] === name);
        // The plotted point at the parent must actually sit on the chord.
        const [x1, y1, x2, y2] = chord.d.match(/-?\d+(?:\.\d+)?/g).map(Number);
        const interpolatedY = y1 + (Number(parent.cx) - x1) / (x2 - x1) * (y2 - y1);
        close(Number(parent.cy), interpolatedY, .0002);
      }
    }
  }
});
