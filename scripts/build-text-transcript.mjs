import fs from 'node:fs';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {chapters} from '../docs/text/lesson.mjs';
import {prose,latex} from '../docs/text/math.mjs';
const require=createRequire(import.meta.url);
globalThis.katex=require('../docs/word2vec/vendor/katex/katex.min.js');
const target=fileURLToPath(new URL('../docs/text/index.html',import.meta.url));
const extras={
  analogy:String.raw`v_{\mathrm{king}}-v_{\mathrm{man}}\approx v_{\mathrm{queen}}-v_{\mathrm{woman}}`,
  'analogy-arithmetic':String.raw`v_{\mathrm{king}}-v_{\mathrm{man}}+v_{\mathrm{woman}}\approx v_{\mathrm{queen}}`,
  capitals:String.raw`v_{\mathrm{Paris}}-v_{\mathrm{France}}\approx v_{\mathrm{Berlin}}-v_{\mathrm{Germany}}`,
};
const quote='<blockquote>“… wasn’t just good—he was a phantom”</blockquote><p class="small">ChatGPT output quoted by <a href="https://www.nytimes.com/2025/12/03/magazine/chatbot-writing-style.html">Sam Kriss in The New York Times Magazine</a>, December 3, 2025; excerpt about the story’s hero, Kael.</p>';
const transcript=`<section class="transcript" id="transcript"><h2>Read the complete explanation</h2><p>Each chapter includes the fuller explanation behind its scroll scenes.</p>\n${chapters.map(c=>`<details><summary>${c.title}</summary><div>${c.stages.map((s,i)=>`<h3>${s.title}</h3>${c.id==='ai-writing'&&i===0?quote:''}<p>${prose(s.readingText??s.text,s.links)}</p><p class="small">${prose(s.readingNote??s.note)}</p>${extras[s.visual]?`<p class="math">${latex(extras[s.visual],true)}</p><p class="small">The analogy diagrams use illustrative coordinates.</p>`:''}`).join('\n')}</div></details>`).join('\n')}</section>`;
const before=fs.readFileSync(target,'utf8');
const after=before.replace(/<!-- TEXT_TRANSCRIPT_START -->[\s\S]*?<!-- TEXT_TRANSCRIPT_END -->/,`<!-- TEXT_TRANSCRIPT_START -->\n${transcript}\n    <!-- TEXT_TRANSCRIPT_END -->`);
if(process.argv.includes('--check')){
  if(after!==before){console.error('The text reading version is out of date. Run node scripts/build-text-transcript.mjs.');process.exitCode=1;}
  else console.log('The text reading version matches the lesson source.');
}else{fs.writeFileSync(target,after);console.log('Updated the static text reading version.');}
