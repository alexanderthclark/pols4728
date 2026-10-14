import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
import assert from 'node:assert/strict';
const M = require('../docs/imbalanced-data/tree-model.js');
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-12, `${a} != ${b}`);
// Independent calculation from integer class counts in each child.
function countsGain(lp,ln,rp,rn) {
  const impurity=(a,b)=>[a,b].filter(c=>c>0).reduce((s,c)=>s-c/(a+b)*Math.log(c/(a+b)),0);
  const n=lp+ln+rp+rn;
  return impurity(lp+rp,ln+rn)-(lp+ln)/n*impurity(lp,ln)-(rp+rn)/n*impurity(rp,rn);
}
close(M.evaluate('A',.01).gain,countsGain(50,0,50,9900));
close(M.evaluate('B',.01).gain,countsGain(90,990,10,8910));
close(M.evaluate('A',.5).gain,countsGain(50,0,50,100));
close(M.evaluate('B',.5).gain,countsGain(90,10,10,90));
assert.equal(M.choose(.01),'A'); assert.equal(M.choose(.5),'B');
assert.equal(M.choose(.01,.02),'A'); assert.equal(M.choose(.01,.03),null);
for(let c=0;c<=.03;c+=.00001) assert.notEqual(M.choose(.01,c),'B');
close(M.entropy(0),0); close(M.entropy(1),0); close(M.entropy(.5),Math.log(2));
// Class-uniform thinning changes the prior while leaving routing rates fixed.
for(let i=1;i<=99;i++) {
  const p=i/100;
  for(const name of ['A','B']) {
    const r=M.evaluate(name,p);
    close(r.q*r.pLeft+(1-r.q)*r.pRight,p);
    // Information gain also equals the average KL divergence of each child's
    // class distribution from the parent, including the pure-child limit.
    const kl=u=>[u,1-u].reduce((s,v,j)=>v===0?s:s+v*Math.log(v/(j?1-p:p)),0);
    close(r.gain,r.q*kl(r.pLeft)+(1-r.q)*kl(r.pRight));
  }
}
console.log('Entropy gains match independent child-count and KL calculations; ranking reversal and cutoff behavior verified.');
