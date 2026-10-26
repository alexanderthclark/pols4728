import {renderMath,mathText} from './math.mjs';

// Color identifies network width; each fit reports its error numerically.
export const initializationColor=width=>({2:'#a54f32',12:'#234e70'})[width];

const number=value=>value.toFixed(4);
export function initializationDomains(data){
 const finalValues=data.runs.flatMap(run=>run.predictions);
 const final=[Math.min(-.25,Math.floor(Math.min(...finalValues)*4)/4),Math.max(1.25,Math.ceil(Math.max(...finalValues)*4)/4)];
 const allValues=[...finalValues,...data.runs.flatMap(run=>run.initialPredictions)];
 // One expanded frame contains every starting function with a little headroom.
 const expanded=[Math.floor(Math.min(final[0],...allValues)-.15),Math.ceil(Math.max(final[1],...allValues)+.15)];
 return {final,expanded};
}

function fitPlot(run,data,yDomain,view){
 const color=initializationColor(run.width);
 const X=x=>30+(x+1)*128,Y=y=>134-(y-yDomain[0])/(yDomain[1]-yDomain[0])*120;
 const path=values=>data.x.map((x,i)=>`${i?'L':'M'}${X(x).toFixed(3)},${Y(values[i]).toFixed(3)}`).join('');
 const target=`M${X(-1)},${Y(0)}H${X(0)}V${Y(1)}H${X(1)}`;
 const id=`initialization-${run.width}-${run.run}-${view}`;
 const expanded=view==='expanded';
 const ticks=expanded?[yDomain[0],0,1,yDomain[1]]:[0,1];
 return `<svg class="initialization-view initialization-view--${view}" viewBox="0 0 300 165" role="img" aria-labelledby="${id}-title" aria-label="Width ${run.width}, start ${run.run}, ${expanded?'initial and final functions':'final function'}, training mean squared error ${number(run.trainMSE)}" data-view="${view}" data-x-domain="-1,1" data-y-domain="${yDomain.join(',')}">
 <title id="${id}-title">Width ${run.width}, start ${run.run}. Final training mean squared error ${number(run.trainMSE)}. ${expanded?'Faint dotted initial function and solid final function on shared expanded axes.':'Solid final function.'} Dashed gray target step.</title>
 ${ticks.map(y=>`<path d="M30,${Y(y)}H286" stroke="#ebebeb" fill="none"/><text x="21" y="${Y(y)+4}" text-anchor="end">${y<0?'−'+Math.abs(y):y}</text>`).join('')}
 <path class="initialization-target" d="${target}" fill="none" stroke="#777" stroke-width="1.5" stroke-dasharray="4 4"/>
${expanded?`<path class="initialization-start" d="${path(run.initialPredictions)}" fill="none" stroke="${color}" stroke-width="1.5" opacity=".65" stroke-dasharray="1 5" stroke-linecap="round" stroke-linejoin="round"/>`:''}
 <path class="initialization-curve" d="${path(run.predictions)}" fill="none" stroke="${color}" stroke-width="3" stroke-linejoin="round"/>
 ${[-1,0,1].map(x=>`<text x="${X(x)}" y="151" text-anchor="middle">${x<0?'−1':x}</text>`).join('')}
 ${mathText(158,164,'x','','middle','#606060',{size:12})}
 </svg>`;
}

function fitFigure(run,data,domains){
 const color=initializationColor(run.width);
 return `<figure class="initialization-fit" data-width="${run.width}" data-run="${run.run}" data-mse="${run.trainMSE}" data-color="${color}">
 <figcaption><span>Start ${run.run}</span><span>${renderMath(String.raw`\mathrm{MSE}=${number(run.trainMSE)}`)}</span></figcaption>
 ${fitPlot(run,data,domains.final,'final')}${fitPlot(run,data,domains.expanded,'expanded')}</figure>`;
}

export function renderInitializations(data){
 const {config}=data;
 const domains=initializationDomains(data);
 const strokeKey=kind=>`<span class="initialization-stroke-key ${kind}" aria-hidden="true"><i></i><i></i></span>`;
 return `<section class="initializations" id="initializations" aria-labelledby="initializations-title">
 <p class="eyebrow">Across random initializations</p>
 <h2 id="initializations-title">Different starts. Different fits.</h2>
 <p class="initializations-context">Plain SGD · ${config.epochs} epochs · same data and batch order</p>
 <div class="initializations-controls"><div class="initializations-keys"><span class="initialization-key">${strokeKey('initialization-final-key')}Final</span><span class="initialization-key initialization-start-legend">${strokeKey('initialization-dotted-key')}Initialization</span><span class="initialization-key initialization-target-legend"><span class="initialization-target-key"></span>${renderMath('f(x)')} target</span></div>
 <label class="initialization-toggle" for="show-initialization"><input type="checkbox" id="show-initialization" aria-controls="initializations-grid" aria-describedby="initialization-axis-note">Show initialization</label></div>
 <p id="initialization-axis-note" class="sr-only">Showing initialization expands every plot to the same vertical range. Turning it off restores the shared final-fit axes. The MSE labels always describe the final fits.</p>
 <div class="initializations-grid" id="initializations-grid">${config.widths.map(width=>`<section class="initialization-width" aria-labelledby="initialization-width-${width}"><h3 id="initialization-width-${width}" style="color:${initializationColor(width)}">Width ${width}</h3><div class="initialization-plots">${data.runs.filter(run=>run.width===width).map(run=>fitFigure(run,data,domains)).join('')}</div></section>`).join('')}</div>
 <p class="initializations-note">All ${config.runsPerWidth} starts shown for each width.</p>
 </section>`;
}

export function renderInitializationReading(data){
 const config=data.config;
 return `<section><h3>Different starts. Different fits.</h3><p>The closing grid shows ${config.runsPerWidth} random initializations for each of two shallow ReLU networks: width 2 and width 12. Each curve is the final fitted function ${renderMath(String.raw`\hat f(x)`)}; the dashed gray step is the same target ${renderMath('f(x)')} used throughout the lesson. Every panel uses the same axes. All starts are included, in their original order.</p>
 <p>These are separate fits generated with scikit-learn’s MLPRegressor using plain SGD, no momentum, and no regularization. They use the same ${config.n.toLocaleString('en-US')} observations, ${config.epochs} epochs, batches of ${config.batchSize}, and learning-rate schedule as the animation. The initial weights use scikit-learn’s default initialization. A separate random stream supplies the same batch order to every fit, so varying the initialization does not also vary the training data or their order. These curves are not replays of the earlier animated runs.</p>
 <p>Rust curves show width 2; blue curves show width 12. Training ${renderMath(String.raw`\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}\bigl(\hat f(x_i)-y_i\bigr)^2`)} is printed above every curve. This is the full mean squared error, twice the half-MSE loss used for the updates.</p>
 <p>“Show initialization” adds each network’s function before its first training update as a faint dotted curve in the same color. Every plot expands to the same vertical range so all starting functions remain visible. Turning it off restores the final-fit axes. The solid curves and MSE labels continue to describe the final fits. The starting functions are recovered using the saved random seeds and the recorded software version; their parameters and predictions are stored with the final fits.</p>
 <p>A network’s ability to represent a good approximation does not guarantee that training will find it from every starting point. Width 2 can fit the step well, but some starts finish far from it. In these runs the wider network is more consistent, although the best width-two fit has the lowest error. This comparison holds the number of updates fixed, not computation time, and measures training fit rather than performance on unseen data. Every run stops at the same epoch budget; this does not establish convergence.</p></section>`;
}
