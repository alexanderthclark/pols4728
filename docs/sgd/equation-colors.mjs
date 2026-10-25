import {C} from './sgd-detail.mjs';

// Use the figure palette in KaTeX itself, including the static reading route.
// Color denotes the object drawn in this scene; general counts stay neutral.
const paint=(role,tex)=>String.raw`\textcolor{${C[role]}}{${tex}}`;
function terms(tex,entries){
 const roles=new Map(entries),escape=value=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
 const pattern=new RegExp([...roles.keys()].sort((a,b)=>b.length-a.length).map(escape).join('|'),'g');
 return tex.replace(pattern,term=>paint(roles.get(term),term));
}

export function colorEquation(kind,tex){
 if(!tex)return tex;
 switch(kind){
  case 'step-target':return paint('muted',terms(tex,[[String.raw`y_i`,'rust']]));
  case 'pair':{
   const parts=tex.split(String.raw`\qquad`);
   return parts.length===2?paint('blue',parts[0])+String.raw`\qquad`+paint('rust',parts[1]):terms(tex,[[String.raw`h_1(x)`,'blue'],[String.raw`h_2(x)`,'rust']]);
  }
  case 'input-first':return paint('blue',tex);
  case 'input-second':return paint('rust',tex);
  case 'flip':{
   const [first,...second]=tex.split(String.raw`\qquad`);
   return paint('blue',first)+String.raw`\qquad`+paint('rust',second.join(String.raw`\qquad`));
  }
  case 'add':case 'steepen':case 'parameter-step':case 'initialize-wide':case 'batches-wide':return paint('teal',tex);
  case 'sgd-opening':return paint('blue',terms(tex,[[String.raw`f(x_i)`,'muted']]));
  case 'partition':return terms(tex,[[String.raw`B_1`,'rust']]);
  case 'batch':return paint('rust',tex);
  case 'point-loss':return terms(tex,[[String.raw`\ell_i(\theta,\phi)`,'rust'],[String.raw`\hat f(x_i;\theta,\phi)`,'blue'],[String.raw`\hat f(x_i)`,'blue'],[String.raw`y_i`,'rust'],[String.raw`B_1`,'rust']]);
  case 'batch-loss':return terms(tex,[[String.raw`S_{B_1}(\theta,\phi)`,'rust'],[String.raw`\ell_i(\theta,\phi)`,'rust'],[String.raw`L_{B_1}(\theta,\phi)`,'blue']]);
  case 'batch-gradient':return paint('blue',terms(tex,[[String.raw`\phi_0`,'teal']]));
  case 'batch-gradient-comparison':return terms(tex,[[String.raw`g_B`,'rust'],[String.raw`L_B`,'rust'],[String.raw`g_{\mathcal D}`,'blue'],[String.raw`L_{\mathcal D}`,'blue']]);
  case 'initialize':case 'update':case 'next-batch':case 'parameter-probe':return paint('blue',tex);
  case 'train-two':return terms(tex,[['m=2','blue']]);
  case 'train-wide':return terms(tex,[['m=12','teal']]);
  case 'compare':return terms(tex,[['m=2','blue'],['m=12','teal']]);
  default:return tex;
 }
}

export const colorSceneEquations=scene=>({...scene,tex:colorEquation(scene.kind,scene.tex),...(scene.compact?{compact:colorEquation(scene.kind,scene.compact)}:{})});
