import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const M = require('../docs/imbalanced-data/tree-model.js');
const close = (a,b) => assert.ok(Math.abs(a-b)<1e-12, `${a} != ${b}`);
// Independent calculation from integer class counts in each child.
function countsGain(lp,ln,rp,rn) {
  const impurity=(a,b)=>1-(a/(a+b))**2-(b/(a+b))**2;
  const n=lp+ln+rp+rn;
  return impurity(lp+rp,ln+rn)-(lp+ln)/n*impurity(lp,ln)-(rp+rn)/n*impurity(rp,rn);
}
close(M.evaluate('A',.01).gain,countsGain(10,0,90,9900));
close(M.evaluate('B',.01).gain,countsGain(90,990,10,8910));
close(M.evaluate('A',.5).gain,countsGain(10,0,90,100));
close(M.evaluate('B',.5).gain,countsGain(90,10,10,90));
assert.equal(M.choose(.01),'A'); assert.equal(M.choose(.5),'B');
assert.equal(M.choose(.01,.0015),'A'); assert.equal(M.choose(.01,.0022),null);
for(let c=0;c<=.0025;c+=.000001) assert.notEqual(M.choose(.01,c),'B');
// Class-uniform thinning changes the prior while leaving routing rates fixed.
for(let i=1;i<=99;i++) {
  const p=i/100;
  for(const name of ['A','B']) {
    const r=M.evaluate(name,p);
    close(r.q*r.pLeft+(1-r.q)*r.pRight,p);
    close(r.gain,2*r.q*(1-r.q)*(r.pLeft-r.pRight)**2);
  }
}
console.log('Tree gains match independent child-count calculations; ranking reversal and cutoff behavior verified.');
