(function(root,factory){
  'use strict';
  const math=typeof module==='object'&&module.exports?require('./math.js'):root.NNMath;
  const api=factory(math);
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.NNStoryState=api;
})(typeof window!=='undefined'?window:globalThis,function(M){
  'use strict';
  const scenes=['affine','relu','hinge','hill-rise','hill-fall','hill-flat','shallow','width-one','width-two','width-three','width-many','sine-coarse','sine-zoom','sine-joint','sine-wide','universal'];
  const widthUnits=[{t:-1.5,a:.8},{t:-.9,a:-1.2},{t:-.3,a:1.1},{t:.3,a:-1},{t:.9,a:.9},{t:1.5,a:-.6}];
  const initial=[0,.9,1.95,3.05,4.2,5.35,M.TAU];
  const coarse=M.modelFromKnots(initial);
  const firstError=M.maxError(initial);
  const jointX=firstError.x;
  const caches=[{knots:initial,model:coarse,error:firstError.error}];
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const f=(x,n=3)=>Number(x.toFixed(n)).toString();
  function approximation(width){
    const desired=clamp(Math.round(width),6,32);
    while(caches.length<=desired-6){
      const old=caches[caches.length-1],worst=M.maxError(old.knots);
      const knots=M.refineAt(old.knots,worst.x),model=M.modelFromKnots(knots),error=M.maxError(knots).error;
      caches.push({knots,model,error});
    }
    return caches[desired-6];
  }
  function blend(oldModel,newModel,alpha){
    const p=clamp(alpha,0,1);
    const units=newModel.neurons.map(u=>{const old=oldModel.neurons.find(v=>Math.abs(v.t-u.t)<1e-10);return{t:u.t,a:(old?old.a:0)*(1-p)+u.a*p};});
    const model=M.createShallow(oldModel.c*(1-p)+newModel.c*p,units);
    model.neurons[0].baseline=true;model.knots=newModel.knots.slice();
    return model;
  }
  function frame(scene,progress){
    if(!scenes.includes(scene))throw new RangeError('Unknown story scene: '+scene);
    const p=clamp(Number(progress)||0,0,1),frame={scene,progress:p,kind:'shallow',x:.4,components:false,showNetwork:true,caption:'',equation:''};
    const hill=()=>M.createShallow(0,[{t:-1,a:1},{t:0,a:-2},{t:1,a:1}]);
    switch(scene){
      case'affine':frame.kind='affine';frame.x=-1.5+3*p;frame.caption='A weighted input and a bias give a straight line. This example is ŷ = x.';frame.equation='ŷ = wx + b';break;
      case'relu':frame.kind='relu';frame.x=-1.5+3*p;frame.showAffine=true;frame.caption='ReLU replaces every negative input with zero. The two straight pieces meet at a joint.';frame.equation='h(x) = max(0, x)';break;
      case'hinge':frame.model=M.createShallow(0,[{t:-.7+1.4*p,a:1.2}]);frame.components=true;frame.x=.9;frame.caption='Scrolling moves t and the joint together. The output weight 1.2 sets the change in slope.';frame.equation='ŷ = 1.2 max(0, x − t)';break;
      case'hill-rise':frame.model=M.createShallow(0,[{t:-1,a:1}]);frame.components=true;frame.caption='The first neuron adds slope +1 at x = −1. Its contribution continues to the right.';frame.equation='ŷ = ReLU(x + 1)';break;
      case'hill-fall':frame.model=M.createShallow(0,[{t:-1,a:1},{t:0,a:-2*p}]);frame.components=true;frame.caption='A second neuron gradually changes the slope at x = 0. At weight −2, the slope goes from +1 to −1.';frame.equation='ŷ = ReLU(x + 1) − '+f(2*p,2)+' ReLU(x)';break;
      case'hill-flat':frame.model=M.createShallow(0,[{t:-1,a:1},{t:0,a:-2},{t:1,a:p}]);frame.components=true;frame.caption='The third neuron adds slope +1 at x = 1. At its final weight, the total slope returns to zero.';frame.equation='ŷ = ReLU(x + 1) − 2 ReLU(x) + '+f(p,2)+' ReLU(x − 1)';break;
      case'shallow':frame.model=hill();frame.x=-1.7+3.4*p;frame.caption='Three hidden neurons form one continuous piecewise linear function. Trace the input through all three contributions.';frame.equation='ŷ = c + Σⱼ aⱼ ReLU(wⱼx + bⱼ)';break;
      case'width-one':case'width-two':case'width-three':case'width-many':{
        const fixed={'width-one':1,'width-two':2,'width-three':3};
        const width=scene==='width-many'?3+Math.min(3,Math.floor(p*4)):fixed[scene];
        frame.model=M.createShallow(0,widthUnits.slice(0,width));frame.width=width;frame.x=.4;
        frame.caption='Width '+width+', one hidden layer: '+width+' '+(width===1?'joint':'joints')+' and '+(width+1)+' linear pieces in this example. Width gives an upper bound; coincident joints or zero output weights can produce fewer.';
        frame.equation='width m → at most m distinct joints and m + 1 linear pieces';
        break;
      }
      case'sine-coarse':case'sine-zoom':case'sine-joint':case'sine-wide':case'universal':{
        frame.kind='sine';frame.x=jointX;frame.highlightX=jointX;frame.zoom=scene!=='sine-coarse';
        let current=approximation(6);
        if(scene==='sine-joint'){
          const next=approximation(7);frame.model=blend(coarse,next.model,p);frame.oldModel=coarse;frame.width=7;
          frame.caption='One new neuron is inserted at x* = '+f(jointX)+'. Existing weights update as its weight grows. Error at x*: '+Math.abs(M.evaluate(frame.model,jointX)-Math.sin(jointX)).toFixed(5)+'.';
          frame.equation='As the weights update, ŷ(x*) → sin(x*).';
        }else{
          const width=scene==='sine-wide'?7+Math.min(19,Math.floor(p*20)):scene==='universal'?26:6;
          current=approximation(width);frame.model=current.model;frame.width=width;frame.maxError=current.error;
          frame.caption=width+' hidden neurons, '+(width-1)+' interior joints. Maximum error on 2,000 evaluation points ≈ '+current.error.toFixed(4)+'.';
          frame.equation=scene==='universal'?'For every ε > 0, a sufficiently wide shallow network can make maxₓ |f(x) − ŷ(x)| < ε.':'Solid blue: ReLU network. Dashed rust: sin(x).';
        }
        break;
      }
    }
    return frame;
  }
  return Object.freeze({scenes:Object.freeze(scenes),initialKnots:Object.freeze(initial),jointX,approximation,blend,frame});
});
