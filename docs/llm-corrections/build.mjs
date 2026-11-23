// Build the course page and a self-contained preview from the same source.
import {readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {scenes} from './scenes.js';
const dir=dirname(fileURLToPath(import.meta.url));
const read=p=>readFileSync(resolve(dir,p),'utf8');
const katex=createRequire(import.meta.url)('../transformers/vendor/katex/katex.min.js');
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const math=tex=>katex.renderToString(tex,{displayMode:true,throwOnError:true});
const data=read('example-results.json').replace(/<\/script/gi,'<\\/script');
const chapters=[['llm-workflow','Procedures','Three procedures'],['mean-corpus','Average','Correcting an average'],['regression-automated','Regression','Using a score as a feature']];
const reading=scenes.map((s,i)=>`<section class="reading-scene"><p class="eyebrow">${i+1} / ${escape(s.chapter)}</p><h3>${escape(s.title)}</h3><p>${escape(s.body)}</p><div class="reading-math">${math(s.equation)}</div><p class="reading-note">${escape(s.note)}</p></section>`).join('\n');
const body=`<a class="skip-link" href="#reading">Read without animation</a>
<header class="site-header"><a class="brand" href="../">POLS 4728</a><nav class="chapter-nav" aria-label="Lesson chapters">${chapters.map(([id,label,part])=>`<a href="#${id}" data-scene="${id}" data-part="${part}">${label}</a>`).join('')}<a href="#reading">Read</a></nav></header>
<main>
<p class="lesson-status">Loading the scroll-driven lesson. The complete reading version is below.</p>
<section id="story" aria-label="Scroll-driven lesson">
 <div id="stage" class="stage">
  <progress id="progress" max="1" value="0" aria-label="Lesson progress"></progress>
  <div class="stage-meta"><span id="chapter">Correcting LLM measurements</span><span id="scene-count">01 / ${scenes.length}</span></div>
  <div class="scene-composition">
   <div id="copy" class="scene-copy"><p id="eyebrow" class="eyebrow"></p><h1 id="scene-title" tabindex="-1">Correcting LLM measurements</h1><p id="scene-body"></p></div>
   <div id="visual" aria-label="Statistical illustration"></div>
  </div>
  <div class="math-band"><div id="equation"></div><p id="scene-note"></p></div>
  <div class="stage-footer"><span>Fixed predictions · random expert audit</span><span>Scroll forward or back ↓</span></div>
 </div>
</section>
<section class="afterword" id="reading">
 <p class="eyebrow">Sources and worked example</p>
 <h2>Predictions, paired errors, corrected estimates</h2>
 <p>The expert audit estimates discrepancies in the quantities the analysis uses. In the synthetic 40-post example, correcting the mean alone leaves the slope unchanged. Correcting the three regression moments produces the following estimates.</p>
 <table><caption class="eyebrow">40 posts · 16 expert labels</caption><thead><tr><th scope="col">Method</th><th scope="col">Slope</th><th scope="col">Population SE</th></tr></thead><tbody><tr><th scope="row">All expert labels: benchmark</th><td>2.000</td><td>—</td></tr><tr><th scope="row">Automated labels</th><td>3.111</td><td>—</td></tr><tr><th scope="row">Expert audit only</th><td>1.706</td><td>1.158</td></tr><tr><th scope="row">DSL corrected moments</th><td>2.061</td><td>0.802</td></tr></tbody></table>
 <p class="source">The standard errors are first-order population approximations for independent posts and a simple random audit. Repeatedly auditing this same fixed corpus is a different sampling experiment. Its 5,000-audit RMSEs are 0.979 for expert-only regression and 0.194 for DSL. Precision gains depend on the predictions and design.</p>
 <p class="source">Based on Naoki Egami, Musashi Hinck, Brandon M. Stewart, and Hanying Wei, <a href="https://naokiegami.com/paper/dsl_ss.pdf">Using Large Language Model Annotations for the Social Sciences: A General Framework of Using Predicted Variables in Downstream Analyses</a>. The four-step procedures paraphrase sections 3.2 and 4.1. The mean correction specializes equation (5); the feature-moment corrections specialize equation (8). The general doubly robust theory and optional learned prediction models are deferred.</p>
 <p class="source">The support-share illustration and 40-post regression are separate synthetic teaching examples, not empirical results from the paper. The regression reproduces <code>dsl_biased_regressor.ipynb</code>; the code below regenerates its point estimates and linearized standard errors.</p>
 <div class="file-links"><a href="https://github.com/alexanderthclark/pols4728/blob/codex/llm-corrections-storyboard/docs/llm-corrections/verify_example.py"><code>verify_example.py</code></a><a href="https://github.com/alexanderthclark/pols4728/blob/codex/llm-corrections-storyboard/docs/llm-corrections/example-results.json"><code>example-results.json</code></a><a href="https://github.com/alexanderthclark/pols4728/tree/codex/llm-corrections-storyboard/docs/llm-corrections">Lesson source</a></div>
 <details open><summary>Read the full lesson without animation</summary><article>${reading}</article></details>
</section></main>
<noscript><p>JavaScript is disabled. The complete reading version above includes every explanation and equation.</p></noscript>
<script type="application/json" id="example-data">${data}</script>`;
function page(styles,scripts){return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Correct LLM annotation errors in averages and downstream regressions using a random expert audit. An interactive POLS4728 lesson following Egami and coauthors."><title>Correcting LLM measurements · POLS 4728</title>${styles}</head><body>${body}${scripts}</body></html>\n`;}
writeFileSync(resolve(dir,'index.html'),page('<link rel="stylesheet" href="../transformers/vendor/katex/katex.min.css"><link rel="stylesheet" href="style.css">','<script src="../transformers/vendor/katex/katex.min.js"></script><script type="module" src="app.js"></script>'));

let kcss=read('../transformers/vendor/katex/katex.min.css');
kcss=kcss.replace(/url\(([^)]+\.woff2)\)/g,(_,fontPath)=>{
 const bytes=readFileSync(resolve(dir,'../transformers/vendor/katex',fontPath));
 return `url(data:font/woff2;base64,${bytes.toString('base64')})`;
});
const css=read('style.css').replace(/url\('([^']+\.otf)'\)/g,(_,p)=>`url(data:font/otf;base64,${readFileSync(resolve(dir,p)).toString('base64')})`);
const moduleFiles=['scenes.js','state.js','model.js','visuals.js','app.js'];
const bundled=moduleFiles.map(p=>{
 const original=read(p);
 const names=[...original.matchAll(/^export (?:const|let|function|class) (\w+)/gm)].map(m=>m[1]);
 const code=original.replace(/^import .*?;\s*$/gm,'').replace(/^export /gm,'');
 return names.length?`const {${names.join(',')}} = (()=>{${code}\nreturn {${names.join(',')}};})();`:`(()=>{${code}\n})();`;
}).join('\n');
const vendor=read('../transformers/vendor/katex/katex.min.js').replace(/<\/script/gi,'<\\/script');
const script=`<script>${vendor}</script><script>(()=>{${bundled}\n})();</script>`;
writeFileSync(resolve(dir,'preview.html'),page(`<style>${kcss}\n${css}</style>`,script).replace('href="../"','href="https://alexanderthclark.github.io/pols4728/"'));
console.log(`Built ${scenes.length} scenes; index.html and self-contained preview.html.`);
