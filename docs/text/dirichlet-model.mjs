export const simplexVertices = [[80, 320], [340, 75], [600, 320]];
export const exampleMixture = [.5, .2, .3];
export const topicNames = ['Movement', 'Breath', 'Reflection'];
export const topicColors = ['#234e70', '#007f73', '#a54f32'];

export function simplexPoint(theta) {
  if (theta.length !== 3 || theta.some(x => !Number.isFinite(x) || x < 0) || Math.abs(theta.reduce((a,b) => a+b,0)-1) > 1e-9) throw new RangeError('A simplex point needs three nonnegative probabilities summing to one.');
  return [0,1].map(axis => theta.reduce((sum,p,k) => sum+p*simplexVertices[k][axis],0));
}

// Lanczos log-gamma keeps densities finite over the displayed concentration range.
function logGamma(z) {
  const coefficients = [676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];
  if (z < .5) return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*z))-logGamma(1-z);
  z-=1;let x=.99999999999980993;
  coefficients.forEach((c,i)=>{x+=c/(z+i+1);});
  const t=z+coefficients.length-.5;
  return .5*Math.log(2*Math.PI)+(z+.5)*Math.log(t)-t+Math.log(x);
}

export function dirichletLogDensity(theta, alpha) {
  if (theta.length !== alpha.length || alpha.some(a=>!Number.isFinite(a)||a<=0) || theta.some(p=>!Number.isFinite(p)||p<=0) || Math.abs(theta.reduce((a,b)=>a+b,0)-1)>1e-9) throw new RangeError('Use positive concentrations and an interior simplex point.');
  const total=alpha.reduce((a,b)=>a+b,0);
  return logGamma(total)-alpha.reduce((sum,a)=>sum+logGamma(a),0)+theta.reduce((sum,p,k)=>sum+(alpha[k]-1)*Math.log(p),0);
}

export function dirichletMean(alpha) {
  if (!alpha.length || alpha.some(a=>!Number.isFinite(a)||a<=0)) throw new RangeError('Concentrations must be positive.');
  const total=alpha.reduce((a,b)=>a+b,0);
  return alpha.map(a=>a/total);
}

export function categoricalProbability(theta, category) {
  simplexPoint(theta);
  if (!Number.isInteger(category)||category<0||category>=theta.length) throw new RangeError('Unknown category.');
  return theta[category];
}

// Equal-area triangles. Density is evaluated at each interior centroid;
// no vertex evaluation or artificial truncation of singular boundary values.
export function simplexMesh(n=20) {
  if (!Number.isInteger(n)||n<1) throw new RangeError('Mesh resolution must be a positive integer.');
  const point=(i,j)=>[i/n,j/n,(n-i-j)/n];
  const triangle=vertices=>({vertices,centroid:[0,1,2].map(k=>vertices.reduce((sum,v)=>sum+v[k],0)/3)});
  const cells=[];
  for(let i=0;i<n;i++)for(let j=0;j<n-i;j++){
    cells.push(triangle([point(i,j),point(i+1,j),point(i,j+1)]));
    if(i+j<n-1)cells.push(triangle([point(i+1,j),point(i+1,j+1),point(i,j+1)]));
  }
  return cells;
}

export function concentrationAt(q) {
  q=Math.max(0,Math.min(1,q));
  // First hold the sparse prior while introducing its fixed mean, then pass
  // continuously through the uniform prior and end at a concentrated prior.
  if(q<=.2)return .3;
  if(q<=.5)return .3*Math.exp(Math.log(1/.3)*(q-.2)/.3);
  if(q<=.7)return 1;
  return Math.exp(Math.log(8)*(q-.7)/.3);
}
