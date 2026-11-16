import {scenes} from './scenes.js';
import {renderVisual} from './visuals.js';
import {sceneAt, narrativeOpacity} from './state.js';
import {estimateFromAudit} from './model.js';

const source = JSON.parse(document.getElementById('example-data').textContent);
const computed = estimateFromAudit(source.rows);
// Keep the full-information benchmark only for the explicitly labelled reveal.
source.moments = computed.moments;
Object.assign(source.fits,computed.fits);
source.population_se = computed.population_se;
const byId = id => document.getElementById(id);
const story = byId('story');
const stage = byId('stage');
const visual = byId('visual');
const copy = byId('copy');
const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
let lastIndex = -1, lastEquation = '', frame = 0;

function renderMath(tex) {
  if (tex === lastEquation) return;
  window.katex.render(tex, byId('equation'), {displayMode: true, throwOnError: true});
  lastEquation = tex;
}

function update() {
  frame = 0;
  const mobile = window.innerWidth <= 760;
  const top = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-height'));
  const extent = story.offsetHeight - stage.offsetHeight;
  const state = sceneAt(top - story.getBoundingClientRect().top, extent, scenes.length,
    motion.matches || new URLSearchParams(location.search).get('motion') === 'reduce');
  const scene = scenes[state.index];
  const reduced = motion.matches || new URLSearchParams(location.search).get('motion') === 'reduce';
  if (state.index !== lastIndex) {
    byId('chapter').textContent = scene.chapter;
    byId('scene-count').textContent = `${String(state.index+1).padStart(2,'0')} / ${scenes.length}`;
    byId('eyebrow').textContent = scene.eyebrow;
    byId('scene-title').textContent = scene.title;
    byId('scene-body').textContent = scene.body;
    byId('scene-note').textContent = scene.note;
    document.querySelectorAll('.chapter-nav a').forEach(a => {
      const active = a.dataset.part === scene.chapter;
      if(active) a.setAttribute('aria-current','step'); else a.removeAttribute('aria-current');
    });
    lastIndex = state.index;
  }
  renderMath(mobile && scene.equationMobile ? scene.equationMobile : scene.equation);
  // Only narrative copy fades. Shared equations and the pinned axes remain steady.
  copy.style.opacity = narrativeOpacity(state.index===scenes.length-1?Math.min(.85,state.fraction):state.fraction, state.index === 0, reduced);
  copy.style.transform = reduced ? 'none' : `translateY(${(1-Math.min(1,state.fraction/.10))*10}px)`;
  byId('progress').value = state.overall;
  byId('progress').setAttribute('aria-valuetext',`Scene ${state.index+1} of ${scenes.length}`);
  stage.dataset.scene = scene.id;
  stage.dataset.progress = state.progress.toFixed(4);
  renderVisual(visual, scene.visual, state.progress, source,
    {mobile, reduced, workflow:scene.workflow});
}

function schedule() { if(!frame) frame = requestAnimationFrame(update); }
function goTo(id, focus = false) {
  const index = scenes.findIndex(s => s.id === id);
  if(index < 0) return;
  const top = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-height'));
  const absolute = story.getBoundingClientRect().top + window.scrollY;
  const extent = story.offsetHeight-stage.offsetHeight;
  window.scrollTo({top:absolute-top+extent*(index+0.18)/scenes.length,behavior:'instant'});
  if(focus) byId('scene-title').focus({preventScroll:true});
  schedule();
}

document.querySelectorAll('a[data-scene]').forEach(a => a.addEventListener('click', event => {
  event.preventDefault();
  history.replaceState(null,'',`#${a.dataset.scene}`);
  goTo(a.dataset.scene, true);
}));
window.addEventListener('scroll',schedule,{passive:true});
window.addEventListener('resize',schedule);
window.addEventListener('hashchange',()=>goTo(location.hash.slice(1)));
motion.addEventListener('change',schedule);
new ResizeObserver(schedule).observe(visual);
story.style.setProperty('--scene-count',scenes.length);
document.documentElement.classList.add('lesson-ready');
update();
if(location.hash) requestAnimationFrame(()=>goTo(location.hash.slice(1)));
