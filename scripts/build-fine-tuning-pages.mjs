import {writeFile,mkdir,cp,readFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const repo=fileURLToPath(new URL('../',import.meta.url));
const esc=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const configs=[{folder:'peloton-fine-tuning',other:'style-distance',otherName:'Learning style distance',site:'peloton'},{folder:'style-distance',other:'peloton-fine-tuning',otherName:'Fine-tuning an encoder',site:'style-distance'}];
function html(lesson,config,atRoot=false){
 const prefix=atRoot?'./':'../';
 const own=atRoot?'./'+config.folder+'/':'./';
 const options=lesson.scenes.map((s,i)=>`<option value="${i}">${i+1}. ${esc(s.title)}</option>`).join('');
 const transcript=lesson.scenes.map((s,i)=>`<section class="reading-step"><h3><a href="#${esc(s.id)}" data-jump="${i}">${i+1}. ${esc(s.title)}</a></h3><div>${s.reading||s.body}</div>${s.equation?`<div class="equation">${s.equation}</div>`:''}${s.note?`<div class="note">${s.note}</div>`:''}</section>`).join('\n');
 return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="${esc(lesson.subtitle)}"><meta name="theme-color" content="#ffffff"><title>${esc(lesson.title)} · POLS 4728</title><link rel="stylesheet" href="${prefix}fine-tuning-shared/story.css"></head>
<body><a class="skip" href="#reading">Skip to the reading version</a>
<header class="site-header"><a class="brand" href="https://alexanderthclark.github.io/pols4728/">POLS 4728 · ML &amp; AI</a><nav class="header-links" aria-label="Lesson navigation"><a href="#reading">Read the lesson</a><a class="course-home" href="${prefix}${config.other}/">${esc(config.otherName)}</a></nav></header>
<main><section class="cover" id="top"><img class="course-logo" src="${prefix}assets/ML_and_AI_logo.png" alt="ML and AI for the Social Sciences" width="350" height="150"><p class="eyebrow">A scrollytelling lesson</p><h1>${esc(lesson.title)}</h1><p class="subtitle">${esc(lesson.subtitle)}</p><div class="intro">${lesson.intro||''}</div><a class="begin" href="#story">Begin the story <span aria-hidden="true">↓</span></a><p class="hint">Scroll to build the explanation. Scroll back to retrace it.</p></section>
<section class="story" id="story" data-story aria-label="Animated lesson"><div class="story-pin"><div class="stage-bar"><span class="chapter-name">${esc(lesson.title)}</span><div class="controls"><label class="sr-only" for="scene-select" style="position:absolute;left:-9999px">Choose a scene</label><select id="scene-select" aria-label="Choose a scene">${options}</select><button id="motion-toggle" aria-label="Reduce motion" aria-pressed="false" title="Show completed diagram states while scrolling">Steps</button><button id="previous-scene" aria-label="Previous scene">← <span>Back</span></button><button id="next-scene" aria-label="Next scene"><span>Next</span> →</button></div></div><div class="stage-visual"><svg id="lesson-visual" viewBox="0 0 1000 560" role="img" aria-label="Animated teaching diagram"></svg></div><div class="stage-copy"><p class="scene-count" id="scene-count"></p><h2 id="scene-heading" tabindex="-1"></h2><div class="stage-body" id="scene-body"></div><div class="equation" id="scene-equation"></div></div><div class="progress-track" role="presentation"><div class="progress-fill"></div></div></div><div class="scroll-space" aria-hidden="true"></div></section>
<noscript><p class="no-js">The full reading version follows. Enable JavaScript for the scroll-driven diagrams.</p></noscript>
<section class="reading" id="reading"><h2>Carry the argument with you</h2><ul>${lesson.takeaways.map(s=>`<li>${s}</li>`).join('')}</ul><p class="data-note">${lesson.dataNote||''}</p><details class="transcript"><summary>Read the complete lesson</summary><div>${transcript}</div></details><details><summary>Sources and teaching notes</summary><div><ol class="references">${lesson.sources.map(s=>`<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.label)}</a>${s.note?`<span class="reference-note">${s.note}</span>`:''}</li>`).join('')}</ol></div></details><a class="next-lesson" href="${prefix}${config.other}/">Continue: ${esc(config.otherName)} →</a></section></main><footer class="footer">Machine Learning and AI for the Social Sciences · Alexander Clark · POLS 4728</footer>
<script type="module">import {mountStory} from '${prefix}fine-tuning-shared/shell.mjs';import {lesson} from '${own}lesson.mjs';mountStory(lesson);</script></body></html>`;
}
for(const config of configs){
 const {lesson}=await import(pathToFileURL(path.join(repo,'docs',config.folder,'lesson.mjs')).href);
 await writeFile(path.join(repo,'docs',config.folder,'index.html'),html(lesson,config));
}
console.log('Regenerated both fine-tuning lessons.');
