import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const require = createRequire(import.meta.url);
const picture = require('../docs/shap/title-picture.js');
const source = readFileSync(new URL('../docs/shap/title-picture.js', import.meta.url), 'utf8');
const close = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance, `${a} differs from ${b}`);

test('renderer exposes the same pure API in a browser and CommonJS', () => {
  const window = {};
  runInNewContext(source, { window });
  for (const method of ['stateAt', 'renderState', 'render']) assert.equal(typeof window.SHAPTitlePicture[method], 'function');
  assert.equal(window.SHAPTitlePicture.render(2000, .6), picture.render(2000, .6));
  assert.equal(picture.render(2000, .6), picture.renderState(picture.stateAt(2000, .6)));
  assert.equal(picture.viewBox, '0 0 1000 500');
});

test('walking repeats forever without a loop-boundary jump or foot drift', () => {
  for (const time of [0, 1170, 1800, 3000, 4300, 6000, 7500, 9300]) {
    assert.deepEqual(picture.stateAt(time, .4), picture.stateAt(time + picture.loopDuration * 17, .4));
  }
  const before = picture.stateAt(picture.loopDuration - .001, .4);
  const after = picture.stateAt(picture.loopDuration, .4);
  assert.deepEqual(before.figures, after.figures);
  for (const person of picture.stateAt(0, 0).figures) {
    const final = picture.stateAt(picture.loopDuration * 100, 0).figures.find(p => p.name === person.name);
    assert.deepEqual(final.feet, person.feet);
  }
});

test('each walk contains two alternating steps, with a stationary support foot', () => {
  for (const start of [1200, 5200]) {
    const firstQuarter = picture.stateAt(start + 300, .5).figures[1];
    const nextQuarter = picture.stateAt(start + 700, .5).figures[1];
    assert.ok(firstQuarter.feet[0].lift > 0);
    assert.equal(firstQuarter.feet[1].planted, true);
    assert.deepEqual(firstQuarter.feet[1], nextQuarter.feet[1]);
    const thirdQuarter = picture.stateAt(start + 1500, .5).figures[1];
    const fourthQuarter = picture.stateAt(start + 1900, .5).figures[1];
    assert.equal(thirdQuarter.feet[0].planted, true);
    assert.deepEqual(thirdQuarter.feet[0], fourthQuarter.feet[0]);
    assert.ok(thirdQuarter.feet[1].lift > 0);
    for (let offset = 0; offset <= 2400; offset += 25) {
      const walker = picture.stateAt(start + offset, 0).figures[1];
      assert.ok(walker.feet.some(foot => foot.planted), `no planted foot at ${start + offset}`);
    }
  }
  const back = picture.stateAt(0, 1).figures[1];
  const front = picture.stateAt(4300, 1).figures[1];
  close(front.x - back.x, 48);
  assert.ok(front.x - back.x > front.radius * .7, 'the walk must translate visibly, not just bob');
  assert.deepEqual(picture.stateAt(8000, 1).figures[1].feet, back.feet);
});

test('scroll controls only the hatch and reverses exactly while walking continues', () => {
  const closed = picture.stateAt(2300, 0);
  const middle = picture.stateAt(2300, .5);
  const opened = picture.stateAt(2300, 1);
  assert.deepEqual(closed.figures, opened.figures);
  assert.ok(closed.hatch.angle < middle.hatch.angle && middle.hatch.angle < opened.hatch.angle);
  close(closed.hatch.revealTop, closed.hatch.revealBottom);
  const closedTop = Math.min(...closed.hatch.outline.map(p => p.y));
  const closedBottom = Math.max(...closed.hatch.outline.map(p => p.y));
  assert.ok(closedTop < closed.hatch.hinge.y - 60, 'the closed hatch should read as a raised dome');
  assert.ok(closedBottom < closed.hatch.hinge.y + 30, 'the dome should have a shallow lower rim');
  assert.ok(opened.hatch.revealTop < 219, 'fully open reveal must include the hat');
  assert.deepEqual(picture.stateAt(2300, .5), middle);
  assert.deepEqual(picture.stateAt(2300, 0), closed);
  for (const progress of [0, .1, .5, .9, 1]) {
    assert.deepEqual(picture.stateAt(0, progress).hatch, picture.stateAt(5000, progress).hatch);
    assert.deepEqual(picture.stateAt(0, progress).hatch.hinge, { x: 866, y: 325 });
  }
});

test('round figures have limbs rooted in their single body, and always look right', () => {
  for (let time = 0; time < picture.loopDuration; time += 53) {
    const figures = picture.stateAt(time, .5).figures;
    assert.ok(figures[0].x + figures[0].radius < figures[1].x - figures[1].radius);
    for (const person of figures) {
      for (const root of [...person.legs.map(leg => leg.hip), ...person.arms.map(arm => arm.shoulder)]) {
        const distance = Math.hypot(root.x - person.x, root.y - person.y);
        assert.ok(distance <= person.radius && distance >= person.radius * .7, 'limb must emerge directly from round body');
      }
      for (const leg of person.legs) {
        close(Math.hypot(leg.hip.x - leg.knee.x, leg.hip.y - leg.knee.y), person.legLength);
        close(Math.hypot(leg.ankle.x - leg.knee.x, leg.ankle.y - leg.knee.y), person.legLength);
      }
      assert.ok(person.gaze.x > 0);
      if (person.magnifier) assert.ok(person.magnifier.x > person.x + person.radius);
    }
  }
});

test('planted inspection poses have only a slight knee bend', () => {
  for (const time of [0, 4300, 8000]) {
    for (const person of picture.stateAt(time, 1).figures) {
      assert.equal(person.walking, false);
      for (const leg of person.legs) {
        const dx = leg.ankle.x - leg.hip.x, dy = leg.ankle.y - leg.hip.y;
        const bend = Math.abs(dx * (leg.knee.y - leg.hip.y) - dy * (leg.knee.x - leg.hip.x)) / Math.hypot(dx, dy);
        assert.ok(bend >= 12 && bend <= 18, `resting knee bend ${bend} must remain gentle`);
      }
    }
  }
});

test('lid sweep, figure limbs, shoes and lens fit inside the fixed viewBox', () => {
  const inside = (p, padding = 0) => {
    assert.ok(p.x >= padding && p.x <= 1000 - padding, `horizontal overflow: ${JSON.stringify(p)}`);
    assert.ok(p.y >= padding && p.y <= 500 - padding, `vertical overflow: ${JSON.stringify(p)}`);
  };
  for (let progress = 0; progress <= 100; progress++) {
    for (const p of picture.stateAt(0, progress / 100).hatch.outline) inside(p, 5);
  }
  for (let time = 0; time < picture.loopDuration; time += 37) {
    for (const person of picture.stateAt(time, .8).figures) {
      inside({ x: person.x - person.radius, y: person.y - person.radius }, 5);
      inside({ x: person.x + person.radius, y: person.y + person.radius }, 5);
      for (const leg of person.legs) for (const p of Object.values(leg)) inside(p, 8);
      for (const arm of person.arms) for (const p of [arm.shoulder, arm.elbow, arm.hand]) inside(p, 25);
      for (const foot of person.feet) {
        inside({ x: foot.x - 32, y: foot.y - 15 }, 5);
        inside({ x: foot.x + 27, y: foot.y + 20 }, 5);
      }
      if (person.magnifier) {
        inside({ x: person.magnifier.x - 43, y: person.magnifier.y - 43 }, 5);
        inside({ x: person.magnifier.x + 43, y: person.magnifier.y + 43 }, 5);
      }
    }
  }
});

test('reduced motion is the same fully open, planted inspection pose at every time', () => {
  const still = picture.stateAt(0, 0, true);
  assert.deepEqual(still, picture.stateAt(987654, .4, true));
  assert.equal(still.openProgress, 1);
  assert.equal(still.reducedMotion, true);
  for (const person of still.figures) {
    assert.equal(person.walking, false);
    assert.equal(person.blink, false);
    assert.ok(person.feet.every(foot => foot.planted && foot.lift === 0));
  }
});

test('SVG keeps the lowercase math glyph behind an opaque lid and front hull', () => {
  const closed = picture.render(0, 0), open = picture.render(4300, 1);
  assert.match(closed, /viewBox="0 0 1000 500" aria-hidden="true" focusable="false"/);
  assert.match(closed, /clipPath id="shap-title-symbol-reveal"><rect[^>]*height="0"/);
  assert.match(open, /font-family="'Course Math','Cambria Math',serif"[^>]*>y<\/text>/);
  assert.doesNotMatch(open, />Y<\/text>|NaN|undefined|Infinity/);
  const symbolIndex = open.indexOf('data-prediction-symbol="y-hat"');
  const lidIndex = open.indexOf('data-hinged-lid="true"');
  const frontHullIndex = open.indexOf('M536 359Q549');
  assert.ok(symbolIndex < lidIndex && lidIndex < frontHullIndex);
  assert.match(open.slice(lidIndex), /<path[^>]*fill="#fff"/);
  assert.equal((open.match(/data-round-body="true"/g) || []).length, 2);
  assert.equal((open.match(/data-foot="/g) || []).length, 4);
});

test('inputs clamp predictably and never render invalid coordinates', () => {
  assert.deepEqual(picture.stateAt(-1, -2), picture.stateAt(0, 0));
  assert.equal(picture.stateAt(0, 3).openProgress, 1);
  assert.deepEqual(picture.stateAt(NaN, NaN), picture.stateAt(0, 0));
  for (const time of [undefined, NaN, Infinity, -100, 0, 123456789]) {
    assert.doesNotMatch(picture.render(time, .7), /NaN|undefined|Infinity/);
  }
});
