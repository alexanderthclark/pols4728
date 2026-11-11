import test from 'node:test';
import assert from 'node:assert/strict';
import {people,mu,aipw,estimates,mean,plr,theta,m,ell,residualSlope,partialResiduals,position} from '../docs/causal-ml/model.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('constructed ATE and selection difference have their stated values',()=>{near(mean(people.map(o=>mu(1,o.x)-mu(0,o.x))),4);near(estimates().raw,13);near(estimates().ipw,4);});
test('AIPW exact four-case cancellation, including both misspecified',()=>{for(const [a,b,t] of [[true,true,4],[true,false,4],[false,true,4],[false,false,13]])near(mean(aipw(a,b).map(o=>o.phi)),t);near(estimates(false,true).outcome,2);});
test('each full residual follows the structural equation and recovers theta',()=>{for(const o of plr){near(o.d-m(o.x),o.v);near(o.y-ell(o.x),theta*o.v+o.u);}near(residualSlope(),2);});
test('partialling-out perturbation has the analytical second-order behavior',()=>{for(const [a,b] of [[.2,0],[0,.2],[.1,.1],[-.4,.3]])near(residualSlope(a,b),(1+2*a*b)/(.5+2*b*b));const h=1e-5;near((residualSlope(h,h)-residualSlope(-h,-h))/(2*h),0);});
test('scroll geometry returns to identical earlier coordinates',()=>{const before=partialResiduals(.4,.7);partialResiduals(.8,1);assert.deepEqual(before,partialResiduals(.4,.7));assert.equal(position(1,15).index,14);assert.equal(position(0,15).index,0);});
