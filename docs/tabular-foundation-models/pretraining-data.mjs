// Original illustrative SCMs, not samples from a production model's pretraining.
// Independent exogenous draws U are Uniform(-1, 1); Y is a generated binary target.
export const scmFamilies = [
  {id:'chain',name:'Chain',features:['X₁','X₂'],formula:'X₂ ← .8X₁ + noise',targetFormula:'Y ← threshold(X₂ + noise)'},
  {id:'latent',name:'Shared cause',features:['X₁','X₂','X₃'],formula:'H drives features and Y',targetFormula:'H is hidden from the predictor'},
  {id:'nonlinear',name:'Nonlinear rule',features:['X₁','X₂','X₃'],formula:'Curved effects + noise',targetFormula:'X₃ is an independent distractor'},
  {id:'interaction',name:'Interaction',features:['X₁','X₂'],formula:'Y favors X₁ = X₂',targetFormula:'Outcome noise can reverse Y'}
];
function random(seed){let s=Number(seed)>>>0;return ()=>{s=(1664525*s+1013904223)>>>0;return 2*(s/4294967296)-1;};}
export function evaluateScm(familyId,noise){
  if(!scmFamilies.some(f=>f.id===familyId))throw new RangeError('Unknown SCM family');
  if(!Array.isArray(noise)||noise.length<5||noise.some(x=>!Number.isFinite(x)))throw new TypeError('Five finite exogenous values are required');
  const [u1,u2,u3,u4,u5]=noise;
  let features,score,latent;
  if(familyId==='chain'){features=[u1,.8*u1+.3*u2];score=1.2*features[1]+.4*u3;}
  else if(familyId==='latent'){latent=u1;features=[latent+.3*u2,-.7*latent+.4*u3,.5*latent+.4*u4];score=latent+.4*u5;}
  else if(familyId==='nonlinear'){features=[u1,u2,u4];score=Math.sin(3*u1)+u2*u2+.35*u3-.3;}
  else{features=[Number(u1>0),Number(u2>0)];score=(features[0]===features[1]?1:-1)+1.5*u3;}
  return {features,target:Number(score>0),score,...(latent===undefined?{}:{latent})};
}
export function makeScmTask(familyId,{seed=4728,contextRows=4}={}){
  const family=scmFamilies.find(f=>f.id===familyId);if(!family)throw new RangeError('Unknown SCM family');
  if(!Number.isInteger(contextRows)||contextRows<1||contextRows>1000)throw new RangeError('Context row count must be 1–1000');
  const next=random(seed),rows=Array.from({length:contextRows+1},()=>{const noise=Array.from({length:5},()=>next());return {...evaluateScm(familyId,noise),noise};});
  const query=rows.at(-1);
  return {familyId,featureNames:[...family.features],context:rows.slice(0,-1),query:{features:[...query.features]},scoringTarget:query.target,queryRecord:query};
}
// Fixed illustrative tasks show both class codes; no prediction scores are selected.
const illustrationSeeds=[4728,4830,4930,5032];
export const scmTasks=scmFamilies.map((family,index)=>makeScmTask(family.id,{seed:illustrationSeeds[index]}));
