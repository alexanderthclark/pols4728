#!/usr/bin/env node
// Run with KaTeX 0.16.22 installed, or pass its package directory as the argument.
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const entry = require.resolve(process.argv[2] || 'katex');
const katex = require(entry);
const packageRoot = path.dirname(path.dirname(entry));
const root = path.resolve(__dirname, '..');
const output = path.join(root, fs.existsSync(path.join(root, 'dist')) ? 'dist' : 'docs/imbalanced-data');
if (katex.version !== '0.16.22') throw new Error('Use KaTeX 0.16.22.');

const positive = String.raw`\textcolor{#a54f32}{N_{+}}`;
const negative = String.raw`\textcolor{#234e70}{N_{-}}`;
const ratio = String.raw`\log\!\left(\frac{${positive}}{${negative}}\right)`;
const sampledRatio = String.raw`\log\!\left(\frac{${positive}}{s\,${negative}}\right)`;
const formulas = {
  linear: String.raw`\log\!\left(\frac{p}{1-p}\right)=\alpha+\beta x`,
  normalized: String.raw`\log\!\left(\frac{\dfrac{${positive}}{N}}{\dfrac{${negative}}{N}}\right)`,
  cancelled: String.raw`\log\!\left(\frac{\dfrac{${positive}}{\cancel{N}}}{\dfrac{${negative}}{\cancel{N}}}\right)`,
  ratio,
  total: String.raw`N=${positive}+${negative}`,
  sampledRatio,
  expansionStart: String.raw`${sampledRatio}\phantom{{}=${ratio}-\log s}`,
  expansion: String.raw`${sampledRatio}=${ratio}-\log s`,
  expansionRhs: String.raw`=${ratio}-\log s`,
  shift: String.raw`-\log(0.10)\approx 2.30`,
  sampleLine: String.raw`(\alpha-\log s)+\beta x`,
  correctedLine: String.raw`\alpha_{\mathrm{sampled}}+\log(0.10)=\alpha_{\mathrm{population}}`,
  coefficientKey: String.raw`\alpha=\text{intercept},\quad\beta=\text{slope}`,
  normalizedIdentity: String.raw`\begin{gathered}\log\!\left(\frac{\dfrac{N_{+}}{N}}{\dfrac{N_{-}}{N}}\right)\\=\log\!\left(\frac{N_{+}}{N_{-}}\right)\end{gathered}`,
  sampledIdentity: String.raw`\begin{gathered}\log\!\left(\frac{\dfrac{N_{+}}{N'}}{\dfrac{sN_{-}}{N'}}\right)\\=\log\!\left(\frac{N_{+}}{sN_{-}}\right)\\=\log\!\left(\frac{N_{+}}{N_{-}}\right)-\log s\end{gathered}`,
  entropy: String.raw`H(p)=-p\ln p-(1-p)\ln(1-p)`,
  informationGain: String.raw`\Delta=H(p)-qH(p_L)-(1-q)H(p_R)`,
  sampledIntercept: String.raw`\alpha_{\mathrm{sampled}}=-\log(99)-\log s`,
  populationIntercept: String.raw`\begin{gathered}\alpha_{\mathrm{population}}=\alpha_{\mathrm{sampled}}+\log s\\=-\log(99)\approx-4.60\end{gathered}`,
  naturalLoss: String.raw`\operatorname{loss}(A)=0.0315<\operatorname{loss}(B)=0.0388`,
  naturalLossNarrow: String.raw`\begin{gathered}\operatorname{loss}(A)=0.0315\\<\operatorname{loss}(B)=0.0388\end{gathered}`,
  aucComparison: String.raw`\operatorname{AUC}(A)=0.7500,\quad\operatorname{AUC}(B)=0.9000`,
  aucComparisonNarrow: String.raw`\begin{gathered}\operatorname{AUC}(A)=0.7500\\\operatorname{AUC}(B)=0.9000\end{gathered}`,
};
const inlineFormulas = {
  linearX: String.raw`\log\!\left(\frac{p(x)}{1-p(x)}\right)=\alpha+\beta x`,
  positiveCount: String.raw`N_{+}`,
  negativeCount: String.raw`N_{-}`,
  groupTotal: String.raw`N=N_{+}+N_{-}`,
  illustrativeLogit: String.raw`\operatorname{logit}p(x)=\log(99)(x-1)`,
  illustrativeTruth: String.raw`\log(99)(x-1)`,
  retentionRange: String.raw`0<s\leq1`,
  retentionValue: String.raw`s=0.10`,
  sampleTotal: String.raw`N'=N_{+}+sN_{-}`,
  sampleProbability: String.raw`q(x)`,
  sampledLogit: String.raw`\operatorname{logit}q(x)=(\alpha-\log s)+\beta x`,
  tenShift: String.raw`\log(10)\approx2.30`,
  tenCorrection: String.raw`\log(0.10)`,
  positiveRetention: String.raw`s_1`,
  negativeRetention: String.raw`s_0`,
  classCorrection: String.raw`\log\!\left(\frac{s_0}{s_1}\right)`,
  priorDifference: String.raw`\operatorname{logit}(\pi)-\operatorname{logit}(\rho)`,
  populationPrior: String.raw`\pi`,
  samplePrior: String.raw`\rho`,
  curvedTruth: String.raw`-6+0.3x+0.5x^2`,
  entropyFunction: String.raw`H(p)`,
  entropyBoundary: String.raw`H(0)=H(1)=0`,
  parentMixture: String.raw`p=qp_L+(1-q)p_R`,
  aHigherGain: String.raw`\Delta_A>\Delta_B`,
  bHigherGain: String.raw`\Delta_A<\Delta_B`,
  childWeight: String.raw`q`,
  leftShare: String.raw`p_L`,
  rightShare: String.raw`p_R`,
  gainSymbol: String.raw`\Delta`,
  relativeGain: String.raw`\Delta/H(p)`,
  entropyUnits: String.raw`\ln(2)`,
  rareLeafShare: String.raw`\frac{50}{50+9{,}900}\approx0.503\%`,
  balancedLeafShare: String.raw`\frac{50}{50+100}\approx33.333\%`,
  parentPrior: String.raw`p`,
  positiveRouting: String.raw`a`,
  negativeRouting: String.raw`b`,
  routingMass: String.raw`q=pa+(1-p)b`,
  routingLeft: String.raw`p_L=\frac{pa}{q}`,
  routingRight: String.raw`p_R=\frac{p(1-a)}{1-q}`,
};
const entries = [...Object.entries(formulas).map(([key, tex]) => [key, tex, true]),
  ...Object.entries(inlineFormulas).map(([key, tex]) => [key, tex, false])];
const rendered = Object.fromEntries(entries.map(([key, tex, displayMode]) => [key, {
  tex,
  html: katex.renderToString(tex, {
    displayMode, output: 'htmlAndMathml', trust: false, throwOnError: true,
  }),
}]));
fs.writeFileSync(path.join(output, 'logistic-math.js'),
  '// Generated by scripts/render-logistic-math.cjs; edit the TeX there.\n' +
  `window.LogisticMath = Object.freeze(${JSON.stringify(rendered, null, 2)});\n`);

// Keep the reading/print version typeset even without JavaScript. The closing
// marker makes regeneration safe despite KaTeX's nested span and MathML tags.
const index = path.join(output, 'index.html');
if (fs.existsSync(index)) {
  const html = fs.readFileSync(index, 'utf8').replace(
    /<(span|div)([^>]*\bdata-math="([^"]+)"[^>]*)>[\s\S]*?<!-- \/math -->/g,
    (_, tag, attributes, key) => {
      if (!rendered[key]) throw new Error(`Unknown formula: ${key}`);
      return `<${tag}${attributes}>${rendered[key].html}<!-- /math -->`;
    });
  fs.writeFileSync(index, html);
}

// Only the six font faces used by these formulas and their tall delimiters.
const fonts = ['Main-Regular', 'Math-Italic', 'Size1-Regular', 'Size2-Regular',
  'Size3-Regular', 'Size4-Regular'].map(name => `KaTeX_${name}.woff2`);
const vendor = path.join(output, 'vendor/katex');
fs.mkdirSync(path.join(vendor, 'fonts'), {recursive: true});
let css = fs.readFileSync(path.join(packageRoot, 'dist/katex.min.css'), 'utf8');
css = css.replace(/@font-face\{[^}]*\}/g, rule => {
  const source = rule.match(/url\(fonts\/([^)]*\.woff2)\) format\("woff2"\)/);
  return source && fonts.includes(source[1])
    ? rule.replace(/src:[^}]+/, `src:${source[0]}`) : '';
});
fs.writeFileSync(path.join(vendor, 'katex.min.css'), css);
for (const font of fonts) fs.copyFileSync(path.join(packageRoot, 'dist/fonts', font),
  path.join(vendor, 'fonts', font));
fs.copyFileSync(path.join(packageRoot, 'LICENSE'), path.join(vendor, 'LICENSE'));

// Browsers can retain a prior lesson revision's scripts and CSS between visits.
// Refresh their existing content-versioned URLs along with the rendered math.
if (fs.existsSync(index)) {
  let html = fs.readFileSync(index, 'utf8');
  for (const file of ['app.js', 'chapter-scenes.js', 'logistic-math.js',
    'logistic-scenes.js', 'misspec-scenes.js', 'tree-scenes.js', 'scrolly.js', 'scrolly.css']) {
    const source = path.join(output, file);
    if (!fs.existsSync(source)) continue;
    const version = createHash('sha256').update(fs.readFileSync(source)).digest('hex').slice(0, 12);
    const pattern = new RegExp('((?:src|href)="' + file.replaceAll('.', '\\.') + ')(?:\\?[^"]*)?(")', 'g');
    html = html.replace(pattern, (_, start, end) => `${start}?v=${version}${end}`);
  }
  fs.writeFileSync(index, html);
}
console.log(`Rendered ${Object.keys(rendered).length} formulas and refreshed the reading version; copied ${fonts.length} fonts.`);
