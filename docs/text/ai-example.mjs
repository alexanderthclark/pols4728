const clamp = value => Math.max(0, Math.min(1, value));

/** A short, attributed excerpt: the article explicitly identifies chatbot output. */
function reveals(stage, progress, {reducedMotion=false}={}) {
  const p = reducedMotion ? stage : progress;
  const contrast = reducedMotion ? Number(stage>=1) : clamp((p - 0.35) / 0.55);
  const dash = reducedMotion ? Number(stage>=1) : clamp((p - 0.85) / 0.45);
  const imagery = reducedMotion ? Number(stage>=1) : clamp((p - 1.3) / 0.45);
  const count = reducedMotion ? Number(stage>=2) : clamp((p - 1.65) / 0.35);
  return {contrast,dash,imagery,count};
}

export function aiExample(stage, progress, options={}) {
  const {contrast,dash,imagery,count}=reveals(stage,progress,options);
  const markStyle = (amount, color) => `--reveal:${(amount*100).toFixed(2)}%;--cue-color:${color}`;
  return `<div class="ai-example" data-contrast="${contrast.toFixed(3)}" data-dash="${dash.toFixed(3)}" data-imagery="${imagery.toFixed(3)}">
    <p class="ai-example-label">A chatbot’s science-fiction story</p>
    <blockquote class="ai-sentence" cite="https://www.nytimes.com/2025/12/03/magazine/chatbot-writing-style.html">“<span class="excerpt-ellipsis" aria-label="Excerpt begins">… </span><mark class="ai-mark" style="${markStyle(contrast,'#234e70')}">wasn’t just</mark> good<mark class="ai-mark" style="${markStyle(dash,'#a54f32')}">—</mark>he was a <mark class="ai-mark" style="${markStyle(imagery,'#007f73')}">phantom</mark>”</blockquote>
    <div class="ai-annotations" aria-label="Possible features to investigate">
      <span style="opacity:${contrast.toFixed(3)}"><span class="cue-rule cue-contrast"></span>Contrast phrase</span>
      <span style="opacity:${dash.toFixed(3)}"><span class="cue-rule cue-dash"></span>Em dash</span>
      <span style="opacity:${imagery.toFixed(3)}"><span class="cue-rule cue-imagery"></span>Dramatic imagery</span>
    </div>
    <p class="ai-attribution">ChatGPT output quoted by <a href="https://www.nytimes.com/2025/12/03/magazine/chatbot-writing-style.html">Sam Kriss, <em>The New York Times Magazine</em></a><br>December 3, 2025 · excerpt about the story’s hero, Kael</p>
    <div class="ai-feature-counts" style="opacity:${count.toFixed(3)}" aria-hidden="${count<0.9}"><span>phantom <span class="feature-number">1</span></span><span>em dash <span class="feature-number">1</span></span><span>contrast phrase <span class="feature-number">1</span></span></div>
  </div>`;
}

/** Update reveals in place so scrolling never detaches the citation link. */
export function updateAiExample(root,stage,progress,options={}) {
  if(!root.querySelector('.ai-example'))root.innerHTML=aiExample(stage,progress,options);
  const {contrast,dash,imagery,count}=reveals(stage,progress,options);
  const example=root.querySelector('.ai-example');
  const marks=example.querySelectorAll('.ai-mark');
  const labels=example.querySelectorAll('.ai-annotations>span');
  [contrast,dash,imagery].forEach((amount,index)=>{
    marks[index].style.setProperty('--reveal',`${(amount*100).toFixed(2)}%`);
    labels[index].style.opacity=amount.toFixed(3);
  });
  Object.assign(example.dataset,{contrast:contrast.toFixed(3),dash:dash.toFixed(3),imagery:imagery.toFixed(3)});
  const counts=example.querySelector('.ai-feature-counts');
  counts.style.opacity=count.toFixed(3);
  counts.setAttribute('aria-hidden',String(count<0.9));
}
