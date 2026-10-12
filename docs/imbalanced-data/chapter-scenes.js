(function () {
  "use strict";

  const M = window.ImbalanceModel;
  const C = {blue: "#234e70", orange: "#a54f32", teal: "#007f73", ink: "#242424", muted: "#606060", grid: "#e4e7e9"};
  const clamp = value => Math.max(0, Math.min(1, value));
  const mix = (a, b, t) => a + (b - a) * clamp(t);
  const number = value => Math.round(value).toLocaleString("en-US");
  const safe = value => String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const text = (x, y, value, color = C.ink, size = 19, anchor = "start", extra = "") => `<text x="${x}" y="${y}" fill="${color}" font-size="${size}" text-anchor="${anchor}" ${extra}>${safe(value)}</text>`;
  const rect = (x, y, w, h, color, extra = "") => `<rect x="${x}" y="${y}" width="${Math.max(0, w)}" height="${h}" fill="${color}" rx="2" ${extra}/>`;
  const svg = (content, label) => `<svg class="scene-chart" viewBox="0 0 720 300" role="img" aria-label="${safe(label)}" xmlns="http://www.w3.org/2000/svg"><g font-family="LM, Georgia, serif">${content}</g></svg>`;

  function thinning(t) {
    const negativeCount = mix(990000, 10000, t);
    const visits = negativeCount + 10000;
    const width = 560, left = 76, positiveWidth = width * .01, negativeWidth = width * negativeCount / 1000000;
    const content = text(left, 30, "Keep all 10,000 positive cases", C.blue) +
      rect(left, 48, positiveWidth, 34, C.blue) +
      text(left, 115, `${number(negativeCount)} negative cases retained`, C.orange) +
      rect(left, 135, width * .99, 34, C.grid) +
      rect(left, 135, negativeWidth, 34, C.orange) +
      text(left, 197, "Pale = original negative pool", C.muted, 17) +
      text(left, 245, number(visits), C.ink, 37) +
      text(left, 276, "example visits per pass", C.muted, 18) +
      text(637, 245, `${(1000000 / visits).toFixed(1)}×`, C.teal, 37, "end") +
      text(637, 276, "fewer visits", C.muted, 18, "end");
    return svg(content, `All 10,000 positive cases stay. Negative cases decline from 990,000 to ${number(negativeCount)}. A pass now visits ${number(visits)} examples.`);
  }

  function batch(pi) {
    const expected = 128 * pi;
    let tiles = "";
    for (let i = 0; i < 128; i++) {
      const x = 56 + (i % 16) * 18, y = 58 + Math.floor(i / 16) * 18;
      tiles += rect(x, y, 15, 15, C.orange);
      const share = clamp(expected - i);
      if (share > 0) tiles += rect(x, y, 15 * share, 15, C.blue);
    }
    const zero = Math.pow(1 - pi, 128);
    const zeroText = zero < .00001 ? "< 0.001%" : `${(100 * zero).toFixed(1)}%`;
    const content = text(56, 30, "128 independent draws", C.ink, 21) + tiles +
      text(386, 79, expected.toFixed(expected < 10 ? 2 : 1), C.blue, 38) +
      text(386, 108, "expected positive cases", C.muted, 18) +
      text(386, 151, zeroText, C.orange, 31) +
      text(386, 181, "chance of no positives", C.muted, 18) +
      text(56, 222, `Positive influence in the unweighted loss: ${(pi * 100).toFixed(1)}%`, C.ink, 20) +
      rect(56, 239, 606, 20, C.orange) + rect(56, 239, 606 * pi, 20, C.blue) +
      text(56, 288, "Blue = positive · Orange = negative", C.muted, 17);
    return svg(content, `In a batch of 128 independent draws, ${expected.toFixed(2)} positives are expected. The chance of no positives is ${zeroText}. Positive cases have ${(100 * pi).toFixed(1)} percent influence in the unweighted training loss.`);
  }

  function efficiency(progress) {
    const p = Math.max(0, Math.min(2, Number(progress) || 0));
    if (p < 1) {
      const thin = clamp(p / .75);
      const retainedNegatives = mix(990000, 10000, thin);
      return {
        eyebrow: "02 / Training efficiency",
        title: "Stop revisiting so many common cases.",
        description: "Keep every positive and shrink the negative pool. A balanced pass visits 20,000 examples instead of one million. This saves visits; it also discards negative-case diversity.",
        equation: `10,000 positives + ${number(retainedNegatives)} negatives = ${number(retainedNegatives + 10000)} visits per pass`,
        svg: thinning(thin),
        label: "Negative cases shrink while positive cases remain fixed."
      };
    }
    const pi = mix(.01, .5, p - 1);
    const final = p > 1.8;
    return {
      eyebrow: "02 / Training efficiency",
      title: final ? "Choose which objective the faster batches serve." : "Give each batch more positive examples.",
      description: final
        ? "Unweighted balancing gives each class equal influence. Importance weights can target the original population loss while changing how examples are sampled. These are different training choices."
        : "As the batch mix changes, positives appear more often and empty batches become rarer. The same change also increases their influence in an unweighted training loss.",
      equation: final
        ? "Population risk = 0.01 · E[loss | positive] + 0.99 · E[loss | negative]"
        : `Positive share: ${(100 * pi).toFixed(1)}% · Expected positives: ${(128 * pi).toFixed(2)} of 128`,
      note: final ? "For 50/50 sampling, weights of 0.02 per positive and 1.98 per negative recover population risk in expectation." : "Tiles show expected composition, rather than one realized batch.",
      svg: batch(pi),
      label: "The expected batch composition and class influence move from the population mix to a balanced mix."
    };
  }

  function signals(t) {
    const a = M.stump("A", .01), b = M.stump("B", .01);
    const left = 250, width = 380;
    let content = text(56, 29, "The model may use only A or B", C.ink, 21) +
      text(56, 66, "Signal A", C.blue, 23) +
      text(56, 96, "Positives: 50%", C.muted, 18) + rect(left, 79, width * .5, 18, C.blue) +
      text(56, 125, "Negatives: 0.1%", C.muted, 18) + rect(left, 108, Math.max(1.5, width * .001), 18, C.blue) +
      text(56, 161, "Signal B", C.orange, 23) +
      text(56, 191, "Positives: 90%", C.muted, 18) + rect(left, 174, width * .9, 18, C.orange) +
      text(56, 220, "Negatives: 10%", C.muted, 18) + rect(left, 203, width * .1, 18, C.orange);
    const reveal = clamp(t * 1.5);
    const lossWidth = 250;
    content += text(56, 269, "Natural log loss", C.ink, 19) +
      rect(250, 242, lossWidth * a.logloss / .05 * reveal, 17, C.blue) +
      rect(250, 268, lossWidth * b.logloss / .05 * reveal, 17, C.orange) +
      text(645, 257, `A: ${a.logloss.toFixed(4)}`, C.blue, 18, "end") +
      text(645, 283, `B: ${b.logloss.toFixed(4)}`, C.orange, 18, "end");
    return svg(content, `A catches 50 percent of positives with a 0.1 percent false positive rate. B catches 90 percent with a 10 percent false positive rate. At 1 percent prevalence, A has lower optimized log loss: ${a.logloss.toFixed(4)} versus ${b.logloss.toFixed(4)}.`);
  }

  function lossChart(pi) {
    const fx = value => 74 + 560 * value / .5;
    const fy = value => 242 - 183 * value / .6;
    let content = text(74, 24, "Optimized log loss", C.muted, 18) + text(634, 24, `Positive weight: ${(100 * pi).toFixed(1)}%`, C.ink, 19, "end");
    [0, .2, .4, .6].forEach(y => {
      content += `<line x1="74" y1="${fy(y)}" x2="634" y2="${fy(y)}" stroke="${C.grid}"/>` + text(62, fy(y) + 6, y.toFixed(1), C.muted, 17, "end");
    });
    [0, .1, .25, .5].forEach(x => { content += text(fx(x), 271, `${100 * x}%`, C.muted, 17, "middle"); });
    content += text(354, 297, "Positive class’s influence in training", C.muted, 18, "middle");
    ["A", "B"].forEach(name => {
      const color = name === "A" ? C.blue : C.orange;
      const points = [];
      for (let i = 0; i <= 150; i++) {
        const q = .5 * i / 150;
        points.push(`${i ? "L" : "M"}${fx(q).toFixed(2)},${fy(M.stump(name, q).logloss).toFixed(2)}`);
      }
      content += `<path d="${points.join(" ")}" fill="none" stroke="${color}" stroke-width="3"/>`;
    });
    content += `<line x1="${fx(pi)}" y1="49" x2="${fx(pi)}" y2="242" stroke="${C.teal}" stroke-width="1.5" stroke-dasharray="4 4"/>`;
    const a = M.stump("A", pi), b = M.stump("B", pi), winner = M.bestStump(pi).name;
    [a, b].forEach(model => {
      content += `<circle cx="${fx(pi)}" cy="${fy(model.logloss)}" r="6" fill="${model.name === "A" ? C.blue : C.orange}" stroke="white" stroke-width="2"/>`;
    });
    content += text(99, 57, `A ${a.logloss.toFixed(4)}`, C.blue, 18) + text(260, 57, `B ${b.logloss.toFixed(4)}`, C.orange, 18) +
      text(634, 57, `${winner} wins`, winner === "A" ? C.blue : C.orange, 20, "end");
    return svg(content, `At ${(100 * pi).toFixed(1)} percent positive influence, A's optimized log loss is ${a.logloss.toFixed(4)} and B's is ${b.logloss.toFixed(4)}. ${winner} has the lower loss. The class-influence marker moves continuously across both loss curves.`);
  }

  function partialPath(points, fraction) {
    let total = 0;
    const lengths = [];
    for (let i = 1; i < points.length; i++) {
      const length = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
      lengths.push(length); total += length;
    }
    let remaining = total * clamp(fraction);
    let path = `M${points[0][0]},${points[0][1]}`;
    for (let i = 1; i < points.length && remaining > 0; i++) {
      const part = Math.min(1, remaining / lengths[i - 1]);
      path += ` L${mix(points[i - 1][0], points[i][0], part)},${mix(points[i - 1][1], points[i][1], part)}`;
      remaining -= lengths[i - 1];
    }
    return path;
  }

  function roc(t) {
    const fx = value => 82 + 465 * value, fy = value => 241 - 186 * value;
    let content = text(82, 27, "True positive rate", C.muted, 18);
    [0, .5, 1].forEach(y => {
      content += `<line x1="82" y1="${fy(y)}" x2="547" y2="${fy(y)}" stroke="${C.grid}"/>` + text(68, fy(y) + 6, `${100 * y}%`, C.muted, 17, "end");
    });
    [0, .5, 1].forEach(x => { content += text(fx(x), 269, `${100 * x}%`, C.muted, 17, "middle"); });
    content += text(315, 297, "False positive rate", C.muted, 18, "middle") +
      `<path d="M82,241 L547,55" stroke="${C.grid}" stroke-width="2" stroke-dasharray="5 5" fill="none"/>`;
    ["A", "B"].forEach(name => {
      const s = M.SIGNALS[name], color = name === "A" ? C.blue : C.orange;
      const points = [[fx(0), fy(0)], [fx(s.fpr), fy(s.tpr)], [fx(1), fy(1)]];
      content += `<path d="${partialPath(points, t)}" fill="none" stroke="${color}" stroke-width="3.5" stroke-linejoin="round"/>`;
      if (t > .42) content += `<circle cx="${fx(s.fpr)}" cy="${fy(s.tpr)}" r="5" fill="${color}"/>`;
    });
    content += text(578, 98, "AUC", C.muted, 18) + text(578, 137, "A  0.7495", C.blue, 21) + text(578, 176, "B  0.9000", C.orange, 21) +
      text(578, 221, "After", C.muted, 17) + text(578, 245, "correction", C.muted, 17);
    return svg(content, "ROC curves after population probability correction. Signal A has area 0.7495. Signal B has area 0.9000. Correcting probability levels preserves the ranking for either selected signal.");
  }

  function learning(progress) {
    const p = Math.max(0, Math.min(3, Number(progress) || 0));
    if (p < 1) {
      return {
        eyebrow: "03 / What the model learns",
        title: p < .5 ? "Unlimited compute still leaves a one-signal model." : "At the natural prevalence, A wins.",
        description: p < .5
          ? "This deliberately restricted model must choose one binary signal. A finds fewer positives with very few false alarms. B finds more positives but produces many more false alarms."
          : "At 1% prevalence, optimize each signal’s probabilities exactly. A achieves lower population log loss. Extra training cannot reverse which signal minimizes this objective.",
        equation: "At 1% positive influence: loss(A) = 0.0341 < loss(B) = 0.0388",
        svg: signals(p),
        label: "Two signals compete within a model restricted to one binary signal."
      };
    }
    if (p <= 2) {
      const pi = .01 * Math.pow(50, clamp(p - 1)), winner = M.bestStump(pi).name;
      return {
        eyebrow: "03 / What the model learns",
        title: winner === "A" ? "Change the influence of positive cases." : "The new objective selects B.",
        description: "Move toward equal class influence. The optimal one-signal model switches from A to B. This change survives exact optimization: balancing changes which compromises the loss rewards.",
        equation: `Positive influence ${(100 * pi).toFixed(1)}% → lowest optimized loss: signal ${winner}`,
        svg: lossChart(pi),
        label: "Moving positive-class influence changes which signal minimizes the exact optimized training loss."
      };
    }
    const reveal = clamp((p - 2) / .9);
    return {
      eyebrow: "03 / What the model learns",
      title: "Correct the probabilities. Keep the learned structure.",
      description: "Balancing selected B. Correcting its probabilities preserves B’s ranking. Its ROC area is higher, while its operating point admits more false alarms. Whether that helps depends on your goal.",
      equation: "AUC(A) = 0.7495 · AUC(B) = 0.9000 · Correction preserves each ranking",
      svg: roc(reveal),
      label: "Both ROC curves are drawn progressively; their ranking differences remain after correction."
    };
  }

  // Continuous, stateless scene API: each chapter maps scroll progress to copy and SVG.
  window.ImbalanceScenes = {efficiency, learning};
})();
