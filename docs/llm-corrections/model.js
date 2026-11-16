const average = xs => xs.reduce((a,b)=>a+b,0)/xs.length;
const sampleVariance = xs => {
  const m = average(xs);
  return xs.reduce((sum,x)=>sum+(x-m)**2,0)/(xs.length-1);
};

function ols(rows, key) {
  const mean_x=average(rows.map(r=>r[key])), mean_y=average(rows.map(r=>r.y));
  const variance_x=average(rows.map(r=>r[key]**2))-mean_x**2;
  const slope=(average(rows.map(r=>r[key]*r.y))-mean_x*mean_y)/variance_x;
  return {slope,intercept:mean_y-slope*mean_x,mean_x,variance_x};
}

/** Feasible estimator: x is accessed only after filtering to audited rows. */
export function estimateFromAudit(rows) {
  const audit=rows.filter(r=>r.audited), N=rows.length, n=audit.length;
  if(n<2) throw new Error('At least two expert-coded observations are required.');
  const momentFunctions=[['X',(x,y)=>x],['X2',(x,y)=>x*x],['XY',(x,y)=>x*y]];
  const moments=momentFunctions.map(([moment,g])=>{
    const automated=average(rows.map(r=>g(r.q,r.y)));
    const correction=average(audit.map(r=>g(r.x,r.y)-g(r.q,r.y)));
    return {moment,automated,correction,corrected:automated+correction};
  });
  const [mx,mx2,mxy]=moments.map(m=>m.corrected), my=average(rows.map(r=>r.y));
  const variance_x=mx2-mx*mx;
  if(variance_x<=0) throw new Error('Corrected variance must be positive for this illustration.');
  const slope=(mxy-mx*my)/variance_x, intercept=my-slope*mx;
  const dsl={slope,intercept,mean_x:mx,variance_x}, expert_only=ols(audit,'x');
  const hx=audit.map(r=>(r.x-mx)*(r.y-intercept-slope*r.x)/variance_x);
  const hq=audit.map(r=>(r.q-mx)*(r.y-intercept-slope*r.q)/variance_x);
  const auditVariance=(1/n-1/N)*sampleVariance(hx.map((h,i)=>h-hq[i]));
  const e=expert_only;
  const he=audit.map(r=>(r.x-e.mean_x)*(r.y-e.intercept-e.slope*r.x)/e.variance_x);
  return {moments,mean_y:my,
    fits:{automated:ols(rows,'q'),expert_only,dsl},
    population_se:{dsl:Math.sqrt(sampleVariance(hx)/N+auditVariance),expert_only:Math.sqrt(sampleVariance(he)/n)},
    audit_only_se:{dsl:Math.sqrt(Math.max(0,auditVariance))}};
}
