"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const directory = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(__dirname, "dist");
const close = (a, b, tolerance, message) => assert.ok(Math.abs(a - b) <= tolerance, `${message}: ${a} versus ${b}`);
const number = value => Number(value.replace(/,/g, "").replace(/−/g, "-"));
const attributes = source => Object.fromEntries([...source.matchAll(/([\w:-]+)="([^"]*)"/g)].map(m => [m[1], m[2]]));
const elements = (svg, tag) => [...svg.matchAll(new RegExp(`<${tag}\\b([^>]*)` + (tag === "text" ? ">([\\s\\S]*?)<\\/text>" : "\\/>"), "g"))].map(m => ({...attributes(m[1]), text: (m[2] || "").replace(/<[^>]*>/g, "")}));
const colors = {full: "#234e70", sampled: "#a54f32", corrected: "#007f73"};

// Recover the log-odds scale from actual rendered ticks, rather than copying
// the scene's layout formula. This lets line geometry validate its labels.
function scale(svg) {
  const texts = elements(svg, "text"), lines = elements(svg, "line");
  const ticks = lines.filter(l => l.y1 === l.y2).flatMap(l => {
    const tick = texts.find(t => t["text-anchor"] === "end" && Number(t.x) < Number(l.x1) && Math.abs(Number(t.y) - Number(l.y1) - 5) < 1e-8 && /^[−-]?\d+$/.test(t.text));
    return tick ? [{y: Number(l.y1), value: number(tick.text)}] : [];
  });
  assert.ok(ticks.length >= 2, "Two rendered log-odds ticks are needed");
  const a = ticks[0], b = ticks[ticks.length - 1];
  return y => a.value + (Number(y) - a.y) * (b.value - a.value) / (b.y - a.y);
}

function validate(svg) {
  assert.match(svg, /^<svg\b/);
  assert.ok(svg.length > 500, "A scene must contain the rendered figure");
  assert.doesNotMatch(svg, /NaN|undefined|Infinity/, "Geometry and labels must be finite");
  const geometry = svg.replace(/(<foreignObject\b[^>]*>)[\s\S]*?<\/foreignObject>/g, "$1</foreignObject>");
  for (const match of geometry.matchAll(/\b(?:x|y|x1|x2|y1|y2|cx|cy|r|width|height|opacity)="([^"]*)"/g)) {
    assert.ok(Number.isFinite(Number(match[1])), `Invalid SVG attribute: ${match[0]}`);
  }
}

function checkLogistic(scene) {
  const texts = elements(scene.svg, "text"), values = texts.map(t => t.text);
  if (![3, 5, 6].includes(scene.stage)) return;
  const positives = values.filter(t => /^[\d,.]+ positive$/.test(t)).map(t => number(t.split(" ")[0]));
  const negatives = values.filter(t => /^[\d,.]+ negative$/.test(t)).map(t => number(t.split(" ")[0]));
  assert.deepEqual(positives, [100, 5000], "Thinning preserves every positive");
  assert.equal(negatives.length, 2);
  const retention = negatives[0] / 9900;
  close(negatives[1] / 5000, retention, 0.00002, "Both groups use the same negative retention");
  assert.ok(retention >= 0.1 - 1e-8 && retention <= 1, "Retention remains between 10% and 100%");
  if (scene.stage === 3) {
    const shares = values.filter(t => /^Positive share/.test(t));
    positives.forEach((positive, i) => close(number(shares[i].match(/[\d.]+/)[0]), 100 * positive / (positive + negatives[i]), 0.051, "Displayed sampled probability follows its counts"));
    const retained = number(values.find(t => /^s =/.test(t)).match(/[\d.]+/)[0]);
    close(retained, retention * 100, 0.051, "Retention label follows the counts");
    return;
  }
  const ordinate = scale(scene.svg);
  const lines = elements(scene.svg, "line").filter(l => l["stroke-width"] === "3.5");
  const full = lines.find(l => l.stroke === colors.full), sampled = lines.find(l => l.stroke === colors.sampled);
  close(ordinate(full.y1), -Math.log(99), 1e-9, "Population log odds at x=0");
  close(ordinate(full.y2), 0, 1e-9, "Population log odds at x=1");
  positives.forEach((positive, i) => close(ordinate(sampled[i ? "y2" : "y1"]), Math.log(positive / negatives[i]), 0.0001, "Sampled line follows retained counts"));
  const logits = values.filter(t => /^Sampled log odds/.test(t)).map(t => number(t.split(" ").pop()));
  positives.forEach((positive, i) => close(logits[i], Math.log(positive / negatives[i]), 0.0051, "Log-odds label follows retained counts"));
  close(ordinate(sampled.y2) - ordinate(sampled.y1), Math.log(99), 1e-9, "Thinning preserves slope");
  if (scene.stage === 6) {
    const corrected = lines.find(l => l.stroke === colors.corrected);
    close(ordinate(corrected.y2) - ordinate(corrected.y1), Math.log(99), 1e-9, "Correction preserves slope");
    assert.ok(ordinate(corrected.y2) >= -1e-9 && ordinate(corrected.y2) <= Math.log(10) + 1e-9, "Correction stays between sampled and population intercepts");
  }
}

function checkCurved(scene) {
  const ordinate = scale(scene.svg), texts = elements(scene.svg, "text");
  const curves = elements(scene.svg, "path").filter(p => p["stroke-width"] === "2.5");
  const sampledTruth = curves.find(p => p.stroke === colors.sampled);
  if (sampledTruth) {
    const points = p => [...p.d.matchAll(/[ML]([\d.-]+),([\d.-]+)/g)].map(m => [Number(m[1]), Number(m[2])]);
    const population = points(curves.find(p => p.stroke === colors.full)), sampled = points(sampledTruth);
    const shift = ordinate(sampled[0][1]) - ordinate(population[0][1]);
    assert.ok(shift >= -0.002 && shift <= Math.log(10) + 0.002, "Sampled truth moves through the known positive shift");
    sampled.forEach(([x,y], i) => {
      close(x, population[i][0], 0.001, "The truth retains its predictor coordinates");
      close(ordinate(y) - ordinate(population[i][1]), shift, 0.002, "Sampling translates the true curve by a constant");
      assert.ok(y >= 43 && y <= 278, "The sampled truth stays inside the plot");
    });
    assert.match(scene.svg, /Sampled truth/);
  }
  const paths = elements(scene.svg, "path").filter(p => p["stroke-width"] === "3.7");
  for (const line of paths) {
    const coordinates = line.d.match(/^M([\d.-]+),([\d.-]+)\s*L([\d.-]+),([\d.-]+)$/);
    assert.ok(coordinates, "A fitted logit is a straight line");
    const intercept = ordinate(coordinates[2]), slope = (ordinate(coordinates[4]) - intercept) / 4;
    const label = { [colors.full]: "Full-data fit", [colors.sampled]: "Sampled fit", [colors.corrected]: "Corrected fit" }[line.stroke];
    const index = texts.findIndex(t => t.text === label);
    assert.ok(index >= 0, "Every visible fit has a coefficient label");
    close(number(texts[index + 1].text.replace("slope ", "")), slope, 0.001, "Slope label agrees with the currently rendered line");
    close(number(texts[index + 2].text.replace("intercept ", "")), intercept, 0.001, "Intercept label agrees with the currently rendered line");
  }
  return paths;
}

let renders = 0;
for (const width of [380, 900]) {
  const context = vm.createContext({window: {innerWidth: width}, innerWidth: width, document: {getElementById: () => null}});
  for (const file of ["model.js", "misspec-model.js", "logistic-math.js", "logistic-scenes.js", "misspec-scenes.js"]) vm.runInContext(fs.readFileSync(path.join(directory, file), "utf8"), context, {filename: file});
  const scenes = context.window.ImbalanceScenes;
  for (const [stage, key] of [[0,"linear"],[1,"normalized"],[2,"ratio"],[3,"sampledRatio"],[4,"expansionRhs"],[5,"sampleLine"],[6,"correctedLine"]]) {
    const scene = scenes.logistic(stage);
    assert.ok(scene.svg.includes(`data-formula="${stage === 4 && width > 650 ? "expansion" : key}"`), "Each opening scene uses its LaTeX formula");
    assert.match(scene.svg, /class="katex"/);
    assert.match(scene.svg, /<math /);
  }
  const midway = scenes.logistic(1.75).svg;
  scenes.logistic(6);
  assert.equal(scenes.logistic(1.75).svg, midway, "LaTeX cancellation reverses exactly");
  for (const [name, last, check] of [["logistic", 6, checkLogistic], ["misspecification", 1, checkCurved]]) {
    const positions = [...new Set([...Array.from({length: last * 100 + 1}, (_, i) => i / 100), ...Array.from({length: last}, (_, i) => i + 0.49999), ...Array.from({length: last}, (_, i) => i + 0.50001), -1, last + 1])];
    for (const position of positions) {
      const scene = scenes[name](position);
      assert.equal(scene.stage, Math.round(Math.max(0, Math.min(last, position))), "Scroll position selects a valid stage");
      validate(scene.svg); check(scene); renders++;
    }
  }
  const first = scenes.logistic(5.5), last = scenes.logistic(6), y = scale(last.svg);
  const corrected = elements(last.svg, "line").find(l => l.stroke === colors.corrected && l["stroke-width"] === "3.5");
  close(y(corrected.y2), 0, 1e-9, "Completed prior correction returns to the population intercept");
  assert.ok(first.svg.includes("990 negative"), "The correction starts from the 10% retained sample");
  const halfway = elements(scenes.logistic(2.75).svg, "text").filter(t => /^[\d,.]+ negative$/.test(t.text));
  [9900, 5000].forEach((original, i) => close(number(halfway[i].text.split(" ")[0]), original / Math.sqrt(10), 0.051, "Halfway thinning progresses between the original and retained pools"));
  const halfCorrection = scenes.logistic(5.75), halfY = scale(halfCorrection.svg);
  const halfLine = elements(halfCorrection.svg, "line").find(l => l.stroke === colors.corrected && l["stroke-width"] === "3.5");
  close(halfY(halfLine.y2), Math.log(10) / 2, 1e-9, "Halfway correction has removed half the log-odds shift");
  const final = scenes.misspecification(1), paths = checkCurved(final), finalY = scale(final.svg);
  const truths = elements(final.svg, "path").filter(p => p["stroke-width"] === "2.5");
  const truthStart = color => truths.find(p => p.stroke === color).d.match(/^M[\d.-]+,([\d.-]+)/)[1];
  close(finalY(truthStart(colors.sampled)) - finalY(truthStart(colors.full)), Math.log(10), 0.002, "Sampled truth receives the exact case-control shift");
  const sampled = paths.find(p => p.stroke === colors.sampled).d.match(/M[\d.-]+,([\d.-]+)\s*L/);
  const correctedPath = paths.find(p => p.stroke === colors.corrected).d.match(/M[\d.-]+,([\d.-]+)\s*L/);
  close(finalY(correctedPath[1]) - finalY(sampled[1]), Math.log(0.1), 0.001, "Curved-model correction applies the known intercept shift");
  for (const [position, label, coefficient, expected] of [
    [0.625, "Sampled fit", "slope", (2.230060566267418 + 1.932862169678228) / 2],
    [0.875, "Corrected fit", "intercept", -4.412987007339833 + Math.log(0.1) / 2]
  ]) {
    const texts = elements(scenes.misspecification(position).svg, "text"), index = texts.findIndex(t => t.text === label);
    close(number(texts[index + (coefficient === "slope" ? 1 : 2)].text.replace(`${coefficient} `, "")), expected, 0.00051, "Curved fit progresses through the halfway coefficients");
  }
}
console.log(`Scene checks passed: ${renders} fractional and endpoint renders across two viewport widths.`);
