export const COLORS = {ink:'#242424',muted:'#606060',blue:'#234E70',orange:'#A54F32',green:'#007F73',red:'#A43B3B',line:'#c8cecf',paper:'#ffffff'};
export const esc = value => String(value).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;');
export const m = tex => `<span class="math" data-tex="${esc(tex)}"></span>`;
export const equation = tex => `<div class="equation" data-tex="${esc(tex)}" data-display="true"></div>`;
export const svg = (content, viewBox='0 0 600 360', label='Diagram') => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" role="img" aria-label="${esc(label)}" class="diagram">${content}</svg>`;
export const mathLabel = (x,y,tex,width=100,height=40) => `<foreignObject x="${x-width/2}" y="${y-height/2}" width="${width}" height="${height}" overflow="visible"><div xmlns="http://www.w3.org/1999/xhtml" class="svg-math">${m(tex)}</div></foreignObject>`;
export const format = (value,digits=2) => Number(value).toFixed(digits);
export function renderMath(root=document){
  root.querySelectorAll('[data-tex]:not([data-rendered])').forEach(el=>{
    if(!globalThis.katex){el.textContent=el.dataset.tex;return;}
    globalThis.katex.render(el.dataset.tex,el,{displayMode:el.dataset.display==='true',throwOnError:true,strict:'error',output:'htmlAndMathml'});
    el.dataset.rendered='true';
  });
}
