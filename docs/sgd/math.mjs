// One renderer for figure notation and math embedded in the lesson's prose.
// Delimiters are explicit: ordinary words and numeric ticks remain text.
const cache=new Map();
export const escapeHTML=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
export function renderMath(tex){
 if(!cache.has(tex))cache.set(tex,globalThis.katex.renderToString(tex,{displayMode:false,output:'htmlAndMathml',throwOnError:true,strict:'error'}));
 return cache.get(tex);
}
export function plainText(source=''){
 return String(source).replace(/\\\(([\s\S]*?)\\\)/g,(_,tex)=>{
  const math=renderMath(tex).match(/<math[\s\S]*?<\/math>/)?.[0]||tex;
  return math.replace(/<annotation[\s\S]*?<\/annotation>/g,'').replace(/<[^>]*>/g,'').replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&#x([0-9a-f]+);/gi,(_,code)=>String.fromCodePoint(parseInt(code,16)));
 });
}
export function renderText(source=''){
 const value=String(source),pattern=/\\\(([\s\S]*?)\\\)/g;
 let html='',last=0;
 for(const match of value.matchAll(pattern)){
  html+=escapeHTML(value.slice(last,match.index))+renderMath(match[1]);
  last=match.index+match[0].length;
 }
 return html+escapeHTML(value.slice(last));
}

function fontSize(cls){
 const width=globalThis.innerWidth||0,height=globalThis.innerHeight||0;
 if(width>=1600&&height>800)return cls==='small'?20:cls==='large'?40:22;
 if(width>=1000)return cls==='small'?16:cls==='large'?36:18;
 if(width&&width<=700)return cls==='large'?21:11;
 return cls==='small'?11:cls==='large'?28:13;
}
function label(x,y,html,cls,anchor,color,options,tex){
 const size=options.size||fontSize(cls),width=options.width||1;
 const left=options.width?(anchor==='middle'?x-width/2:anchor==='end'?x-width:x):x;
 const align=anchor==='middle'?'center':anchor==='end'?'right':'left';
 const height=options.height||size*2.7;
 const natural=options.width?'':`width:max-content;transform:translateX(${anchor==='middle'?'-50%':anchor==='end'?'-100%':'0'});`;
 return `<foreignObject x="${left}" y="${y-size*.98}" width="${width}" height="${height}" class="svg-math-label${tex?' svg-math-only':''}"${tex?` data-tex="${escapeHTML(tex)}"`:''}><div xmlns="http://www.w3.org/1999/xhtml" style="font:${size}px/1.2 Arial,sans-serif;color:${color||(cls==='small'?'#606060':cls==='large'?'#234e70':'#242424')};text-align:${align};white-space:nowrap;${natural}">${html}</div></foreignObject>`;
}
export function mathText(x,y,tex,cls='',anchor='start',color='',options={}){
 return label(x,y,renderMath(tex),cls,anchor,color,options,tex);
}
export function richText(x,y,source,cls='',anchor='start',color='',options={}){
 return label(x,y,renderText(source),cls,anchor,color,options);
}
