(function(root,factory){
  'use strict';
  const math=typeof module==='object'&&module.exports?require('./math.js'):root.NNMath;
  const data=typeof module==='object'&&module.exports?require('./classification-data.js'):root.NNClassificationData;
  const widthData=typeof module==='object'&&module.exports?require('./width-data.js'):root.NNWidthData;
  const api=factory(math,data,widthData);
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.NNStoryState=api;
})(typeof window!=='undefined'?window:globalThis,function(M,D,W){
  'use strict';
  const scenes=['affine','relu','hinge','hill-rise','hill-fall','hill-flat','shallow','width-one','width-two','width-three','width-many','sine-coarse','sine-zoom','sine-joint','sine-wide','universal','conflict','sigmoid-one','sigmoid-two','class-input','class-trace','class-move','class-head','class-compare','generalize-zoom','generalize-noise','generalize-width','generalize-test','generalize-errors','generalize-descent'];
  const widthStops=Object.freeze(Array.from({length:8},(_,i)=>(i+.55)/8));
  function stops(scene){
    return scene==='sigmoid-one'?[.12,.45,.9]:scene==='generalize-width'||scene==='generalize-errors'?widthStops:scene==='generalize-test'?[.3,.9]:scene==='generalize-noise'?[.12,.55,.95]:scene==='generalize-descent'?[.2,.55,.95]:[.12];
  }
  function reducedProgress(scene,p){
    if(scene==='sigmoid-one')return p<.2?.12:p<.6?.45:.9;
    if(scene==='generalize-width'||scene==='generalize-errors')return widthStops[Math.min(7,Math.floor(p*8))];
    if(scene==='generalize-test')return p<.5?.3:.9;
    if(scene==='generalize-noise')return p<.25?.12:p<.7?.55:.95;
    if(scene==='generalize-descent')return p<.35?.2:p<.7?.55:.95;
    if(scene==='affine')return p<.65?0:1;
    return scene==='width-many'||scene==='sine-wide'?Math.round(p*10)/10:1;
  }
  const widthUnits=[{t:-1.5,a:.8},{t:-.9,a:-1.2},{t:-.3,a:1.1},{t:.3,a:-1},{t:.9,a:.9},{t:1.5,a:-.6}];
  const initial=[0,.9,1.95,3.05,4.2,5.35,M.TAU];
  const coarse=M.modelFromKnots(initial);
  const firstError=M.maxError(initial);
  const jointX=firstError.x;
  const caches=[{knots:initial,model:coarse,error:firstError.error}];
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const ease=x=>{const t=clamp(x,0,1);return t*t*(3-2*t);};
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
    if(scene.startsWith('generalize-')){
      frame.kind='width-experiment';frame.hideInspector=true;frame.widthExperiment=true;
      frame.modelIndex=0;frame.previousModelIndex=0;frame.modelMix=1;frame.noiseMix=1;frame.testOpacity=0;frame.boundaryOpacity=1;frame.errorReveal=p;
      frame.equation=String.raw`\widehat p(\mathbf x)=\operatorname{sigmoid}\!\left(\phi_0+\sum_{j=1}^{m}\phi_j\operatorname{ReLU}(\theta_{j0}+\boldsymbol\theta_j^{\mathsf T}\mathbf x)\right)`;
      frame.equationCompact=String.raw`\widehat p=\operatorname{sigmoid}\!\left(\phi_0+\sum_{j=1}^{m}\phi_jh_j\right)`;
      if(scene==='generalize-zoom'){
        frame.kind='classification';frame.inputZoom=ease((p-.1)/.75);frame.morph=1;
        frame.equation=String.raw`(x_1,x_2)\longmapsto\widehat p(\mathbf x)`;frame.equationCompact=frame.equation;
        frame.caption='Return to the original coordinates. What happens to this boundary when we add hidden neurons?';
      }else if(scene==='generalize-noise'){
        frame.noiseMix=ease((p-.18)/.48);frame.boundaryOpacity=ease((p-.74)/.15);
        frame.equation=String.raw`x_2=-\sin^2(x_1)+0.9y+\varepsilon,\qquad\varepsilon\sim\mathcal N(0,0.18^2)`;
        frame.equationCompact=String.raw`\begin{gathered}x_2=-\sin^2(x_1)+0.9y+\varepsilon\\\varepsilon\sim\mathcal N(0,0.18^2)\end{gathered}`;
        frame.caption=p<.7?'Add variation around the same two class curves. Each point keeps its class; this noisy sample will stay fixed.':'Refit the two-neuron network. This experiment trains all layers with cross-entropy and gradient updates, with no explicit penalty.';
      }else if(scene==='generalize-width'||scene==='generalize-errors'){
        const position=Math.min(7.999,p*8),index=Math.floor(position);
        frame.modelIndex=index;frame.previousModelIndex=Math.max(0,index-1);frame.modelMix=index===0?1:ease((position-index)/.32);
        frame.errorReveal=(index+1)/8;frame.compareBaseline=index===7;
        frame.caption=scene==='generalize-width'?'Add neurons to the same hidden layer and refit. The observations stay fixed. The wider boundary is rounder here; it is still piecewise linear.':'Measured classification errors on 40,000 fresh observations. All eight widths classify every training point correctly; test error does not fall at every step.';
      }else if(scene==='generalize-test'){
        frame.testOpacity=ease((p-.06)/.18);frame.modelIndex=p<.5?0:7;frame.previousModelIndex=0;frame.modelMix=p<.5?1:ease((p-.5)/.2);frame.showErrors=true;
        const model=W.models[frame.modelIndex];
        frame.caption='New observations from the same process: '+model.testMistakes.toLocaleString('en-US')+' mistakes out of 40,000 with '+model.width+' neurons. Hollow points show a fixed subset; rings mark mistakes.';
      }else{
        frame.errorReveal=clamp(p/.92,0,1);
        frame.equation=String.raw`\text{More capacity can improve prediction beyond interpolation.}`;frame.equationCompact=String.raw`\text{Beyond interpolation}`;
        frame.caption='Double descent: test error can fall, rise near interpolation, and fall again. This schematic introduces the broader pattern; our example measured improvement after perfect training classification.';
        frame.source='https://doi.org/10.1073/pnas.1903070116';frame.sourceLabel='Belkin et al. (2019), double descent';
      }
      frame.width=W.models[frame.modelIndex].width;
      return frame;
    }
    if(scene==='conflict'||scene.startsWith('sigmoid-')){
      frame.kind='sigmoid';frame.hideInspector=true;frame.sigmoidBias=-3;frame.sigmoidSlope=30;
      frame.sigmoidThresholds=[.2,.4,.6,.8];
      frame.sigmoidWeights=scene==='sigmoid-two'?[6,-6*ease((p-.12)/.7)]:[6];
      frame.width=frame.sigmoidWeights.length;
      frame.source='https://gking.harvard.edu/files/improv.pdf';
      frame.sourceLabel='Beck, King & Zeng (2000), Equation 3';
      if(scene==='conflict'){
        frame.equationOnly=true;
        frame.equation=String.raw`\begin{gathered}\begin{aligned}Y_i&\sim\operatorname{Bernoulli}(\pi_i),\\[1.2em]\pi_i&=\operatorname{sigmoid}\!\Bigl[\gamma_0+\gamma_1\operatorname{sigmoid}(X_i\beta_{(1)})\\&\qquad+\gamma_2\operatorname{sigmoid}(X_i\beta_{(2)})+\cdots+\gamma_M\operatorname{sigmoid}(X_i\beta_{(M)})\Bigr].\end{aligned}\\[2em]\operatorname{sigmoid}(z)=\frac{1}{1+e^{-z}}\end{gathered}`;
        frame.equationCompact=String.raw`\begin{gathered}Y_i\sim\operatorname{Bernoulli}(\pi_i),\\[1.2em]\pi_i=\operatorname{sigmoid}\!\Bigl[\gamma_0+\sum_{j=1}^{M}\gamma_j\operatorname{sigmoid}(X_i\beta_{(j)})\Bigr].\\[2em]\operatorname{sigmoid}(z)=\frac{1}{1+e^{-z}}\end{gathered}`;
        frame.intro='Beck, King, and Zeng train a neural network with a sigmoid activation function to predict international conflict.';
        frame.source+='#page=5';
      }else if(scene==='sigmoid-one'){
        frame.equationOnly=true;frame.equationComparison=true;
        frame.comparisonOpen=ease((p-.2)/.15);
        frame.comparisonMix=ease((p-.6)/.18);
        frame.comparisonPlotOpacity=frame.comparisonOpen;
        frame.comparisonPlots=frame.comparisonOpen>0;
        frame.comparisonPhase=p<.2?'equations':p<.69?'logistic':'neural';
        frame.logisticEquation=String.raw`\begin{gathered}Y_i\sim\operatorname{Bernoulli}(\pi_i)\\[1.2em]\pi_i=\operatorname{sigmoid}(X_i\beta)\end{gathered}`;
        frame.neuralEquation=String.raw`\begin{gathered}Y_i\sim\operatorname{Bernoulli}(\pi_i)\\[1.2em]\pi_i=\operatorname{sigmoid}\!\Bigl[\gamma_0+\sum_{j=1}^{M}\gamma_j\operatorname{sigmoid}(X_i\beta_{(j)})\Bigr]\end{gathered}`;
        frame.logisticEquationCompact=frame.logisticEquation;
        frame.neuralEquationCompact=frame.neuralEquation;
        frame.equation=frame.neuralEquation;
        frame.equationCompact=frame.neuralEquationCompact;
        frame.caption=frame.comparisonPhase==='equations'?'The network combines sigmoid features before the final sigmoid. Scroll to compare the shapes these models can express.':frame.comparisonPhase==='logistic'?'Logistic regression: a monotone probability curve and a straight classification boundary. The weights are illustrative; the observations stay fixed.':'A sigmoid network can rise and fall, and enclose a group with a curved boundary. The weights are illustrative; the observations stay fixed.';
        frame.source+='#page=5';
      }else{
        frame.equation=String.raw`h_j(x)=\sigma\!\left(30(x-t_j)\right),\quad\widehat p(x)=\sigma\!\left(-3+\sum_{j=1}^{`+frame.width+String.raw`}\phi_jh_j(x)\right)`;
        frame.equationCompact=String.raw`\begin{gathered}h_j(x)=\sigma\!\left(30(x-t_j)\right)\\\widehat p(x)=\sigma\!\left(-3+\sum_{j=1}^{`+frame.width+String.raw`}\phi_jh_j(x)\right)\end{gathered}`;
        frame.caption='With one hidden neuron, this probability rises. Add a second neuron with a negative output weight, and it can rise and then fall.';
      }
      return frame;
    }
    if(scene.startsWith('class-')){
      frame.kind='classification';frame.hideInspector=true;
      frame.morph=scene==='class-move'?ease((p-.12)/.72):scene==='class-head'||scene==='class-compare'?1:0;
      frame.boundaryOpacity=scene==='class-head'?ease((p-.15)/.65):scene==='class-compare'?1:0;
      frame.traceIndex=Math.floor(D.inputs.length/8);
      frame.equation=String.raw`h_j(\mathbf x)=\operatorname{ReLU}(\theta_{j0}+\theta_{j1}x_1+\theta_{j2}x_2),\quad j=1,2`;
      frame.equationCompact=String.raw`\begin{gathered}h_j=\operatorname{ReLU}(\theta_{j0}+\theta_{j1}x_1+\theta_{j2}x_2)\\j=1,2\end{gathered}`;
      if(scene==='class-input'){
        const coefficient=value=>Math.abs(value).toFixed(3);
        const term=(value,variable)=>(value<0?'-':'+')+coefficient(value)+variable;
        const hidden=D.hiddenWeights.map((weights,j)=>'h_'+(j+1)+String.raw`(\mathbf x)&\approx\operatorname{ReLU}\!\left(`+D.hiddenBias[j].toFixed(3)+term(weights[0],'x_1')+term(weights[1],'x_2')+String.raw`\right)`);
        const head=String.raw`\widehat p&\approx\operatorname{sigmoid}\!\left(`+D.outputBias.toFixed(3)+term(D.outputWeights[0],'h_1')+term(D.outputWeights[1],'h_2')+String.raw`\right)`;
        frame.equation=String.raw`\begin{aligned}`+hidden.join(String.raw`\\[.25em]`)+String.raw`\\[.25em]`+head+String.raw`\end{aligned}`;
        frame.equationCompact=frame.equation;
        frame.caption='A fitted binary example: two inputs, two ReLU hidden neurons, and one sigmoid output. Parameters are rounded to three decimals; plots use full precision.';
      }
      if(scene==='class-trace'){
        frame.equation=String.raw`(x_1,x_2)\longmapsto\mathbf h(\mathbf x)=(h_1(\mathbf x),h_2(\mathbf x))`;
        frame.equationCompact=String.raw`(x_1,x_2)\longmapsto(h_1,h_2)`;
        frame.caption='Follow the marked observation through the fitted hidden neurons. Their two activations give it two new coordinates.';
      }
      if(scene==='class-move'){
        frame.equation=String.raw`\mathbf x\longmapsto\mathbf h(\mathbf x)`;
        frame.caption='The same observations move to their actual hidden activations. This transition changes the coordinate display; the fitted weights stay fixed. The boundary is still hidden.';
      }
      if(scene==='class-head'){
        frame.equation=String.raw`\widehat p=\sigma(\phi_0+\phi_1h_1+\phi_2h_2),\quad\widehat p=\tfrac12\iff\phi_0+\phi_1h_1+\phi_2h_2=0`;
        frame.equationCompact=String.raw`\begin{gathered}\widehat p=\sigma(\phi_0+\phi_1h_1+\phi_2h_2)\\\widehat p=\tfrac12\iff\phi_0+\phi_1h_1+\phi_2h_2=0\end{gathered}`;
        frame.caption='Now reveal the fitted head: one straight line separates the learned features. At the 0.5 cutoff it correctly classifies all '+D.inputs.length+' training observations.';
      }
      if(scene==='class-compare'){
        frame.equation=String.raw`\mathbf x\ \xrightarrow{\text{learned representation}}\ \mathbf h(\mathbf x)\ \xrightarrow{\text{logistic head}}\ \widehat p`;
        frame.equationCompact=String.raw`\mathbf x\longrightarrow\mathbf h(\mathbf x)\longrightarrow\widehat p`;
        frame.caption='One fitted classifier, two coordinate systems: a nonlinear boundary in the inputs and a straight boundary in the representation. Perfect training classification does not guarantee perfect predictions on new observations.';
      }
      return frame;
    }
    const hill=()=>M.createShallow(0,[{t:-1,a:1},{t:0,a:-2},{t:1,a:1}]);
    switch(scene){
      case'affine':
        frame.kind='affine';frame.x=-1.5+3*p;frame.weight=1;frame.bias=.5;frame.showBias=p<.65;
        frame.caption=frame.showBias?'The faint constant-1 input supplies the intercept.':'The intercept is still 0.5; only its connection is hidden.';
        frame.equation='\\hat{y}=wx+b=1\\times x+0.5';break;
      case'relu':frame.kind='relu';frame.x=-1.5+3*p;frame.showAffine=true;frame.caption='Negative inputs become zero; positive inputs pass through.';frame.equation='h(x)=\\max(0,x)';break;
      case'hinge':{
        const smooth=u=>{const q=clamp(u,0,1);return q*q*(3-2*q);};
        const t=p<.4?-.8+1.4*smooth(p/.4):.6;
        const a=p<.4?1:p<.7?1+.6*smooth((p-.4)/.3):1.6-2.8*smooth((p-.7)/.3);
        frame.hingeA=a;frame.hingeT=t;frame.hingePhase=p<.4?'threshold':p<.7?'steepen':'flip';
        frame.hingeExamples=[
          {a:1,t:-.8,label:'Original: a = 1, t = −0.8'},
          {a:1,t:.6,label:'Shifted: a = 1, t = 0.6'}
        ];
        frame.model=M.createShallow(0,[{t,a}]);frame.components=true;frame.x=.9;
        frame.caption=frame.hingePhase==='threshold'?'Shift the joint; keep the output weight fixed.':frame.hingePhase==='steepen'?'Increase the output weight to steepen the active segment.':'Lower the output weight through zero to turn the segment downward.';
        frame.equation='\\hat{y}='+f(a,2)+'\\,\\operatorname{ReLU}(x'+(t<0?'+':'-')+f(Math.abs(t),2)+')';
        break;
      }
      case'hill-rise':frame.model=M.createShallow(0,[{t:-1,a:1}]);frame.components=true;frame.caption='The first neuron adds slope +1 at −1.';frame.equation='\\hat{y}=\\operatorname{ReLU}(x+1)';break;
      case'hill-fall':frame.model=M.createShallow(0,[{t:-1,a:1},{t:0,a:-2*p}]);frame.components=true;frame.caption='At output weight −2, the slope changes from +1 to −1.';frame.equation='\\hat{y}=\\operatorname{ReLU}(x+1)-'+f(2*p,2)+'\\,\\operatorname{ReLU}(x)';break;
      case'hill-flat':
        frame.model=M.createShallow(0,[{t:-1,a:1},{t:0,a:-2},{t:1,a:p}]);frame.components=true;
        frame.caption='At output weight +1, the third neuron flattens the right-hand segment.';
        frame.equation='\\hat{y}=\\operatorname{ReLU}(x+1)-2\\,\\operatorname{ReLU}(x)+'+f(p,2)+'\\,\\operatorname{ReLU}(x-1)';
        frame.equationCompact='\\begin{aligned}\\hat{y}={}&\\operatorname{ReLU}(x+1)\\\\&-2\\,\\operatorname{ReLU}(x)+'+f(p,2)+'\\,\\operatorname{ReLU}(x-1)\\end{aligned}';break;
      case'shallow':frame.model=hill();frame.x=-1.7+3.4*p;frame.caption='The output sums the weighted activations and adds a bias.';frame.equation='\\hat{y}=c+\\sum_{j=1}^{m}a_j\\,\\operatorname{ReLU}(w_jx+b_j)';break;
      case'width-one':case'width-two':case'width-three':case'width-many':{
        const fixed={'width-one':1,'width-two':2,'width-three':3};
        const width=scene==='width-many'?3+Math.min(3,Math.floor(p*4)):fixed[scene];
        frame.model=M.createShallow(0,widthUnits.slice(0,width));frame.width=width;frame.x=.4;
        frame.caption='Here: '+width+' '+(width===1?'joint':'joints')+', '+(width+1)+' pieces. Width gives an upper bound; coincident joints, inactive units, zero weights, or cancellation can yield fewer.';
        frame.equation='\\text{width }m:\\quad\\text{joints}\\le m,\\quad\\text{pieces}\\le m+1';
        frame.equationCompact='\\begin{gathered}\\text{width }m:\\quad\\text{joints}\\le m\\\\\\text{pieces}\\le m+1\\end{gathered}';
        break;
      }
      case'sine-coarse':case'sine-zoom':case'sine-joint':case'sine-wide':case'universal':{
        frame.kind='sine';frame.x=jointX;frame.highlightX=jointX;frame.zoom=scene!=='sine-coarse';
        let current=approximation(6);
        if(scene==='sine-joint'){
          const next=approximation(7);frame.model=blend(coarse,next.model,p);frame.oldModel=coarse;frame.width=7;
          frame.caption='One new neuron; neighboring weights adjust. Error at the marked input: '+Math.abs(M.evaluate(frame.model,jointX)-Math.sin(jointX)).toFixed(4)+'.';
          frame.equation='\\hat{y}(x^*)\\longrightarrow\\sin(x^*)';
        }else{
          const width=scene==='sine-wide'?7+Math.min(19,Math.floor(p*20)):scene==='universal'?26:6;
          current=approximation(width);frame.model=current.model;frame.width=width;frame.maxError=current.error;
          frame.caption=width+' neurons · '+(width-1)+' interior joints · max error ≈ '+current.error.toFixed(4)+' (2,000 sampled inputs).';
          frame.equation=scene==='universal'?'\\forall\\,\\varepsilon>0,\\ \\exists\\,\\hat{y}:\\quad\\max_{x\\in[a,b]}\\lvert f(x)-\\hat{y}(x)\\rvert<\\varepsilon':'\\hat{y}(x)\\approx\\sin x,\\quad 0\\le x\\le 2\\pi';
          if(scene==='universal'){
            frame.equationCompact='\\begin{gathered}\\forall\\,\\varepsilon>0,\\ \\exists\\,\\hat{y}:\\\\\\max_{x\\in[a,b]}\\lvert f(x)-\\hat{y}(x)\\rvert<\\varepsilon\\end{gathered}';
            frame.caption='For a continuous function on a closed, bounded interval, suitable weights exist. This does not guarantee training success or justify extrapolation.';
          }
        }
        break;
      }
    }
    return frame;
  }
  return Object.freeze({scenes:Object.freeze(scenes),initialKnots:Object.freeze(initial),jointX,approximation,blend,frame,stops,reducedProgress});
});
