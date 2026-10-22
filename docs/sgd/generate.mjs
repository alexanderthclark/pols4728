// Run from any working directory: node docs/sgd/generate.mjs
import {writeFileSync} from 'node:fs';
import {train,data,gradient,loss} from './model.mjs';
const runs=[],traces={};
for(const width of [2,12])for(let seed=1;seed<=8;seed++){
  const run=train({width,seed,record:seed===1});
  if(seed===1)traces[width]=run.trace;
  runs.push({...run,trace:undefined});
}
const first=traces[2][0],batch=first.indices.map(i=>data[i]);
const output={config:{n:data.length,seeds:[1,2,3,4,5,6,7,8],widths:[2,12],epochs:300,steps:9600,batchesPerEpoch:32,batchSize:64,learningRate:'0.025 / (1 + step / 6000)',initialization:'w ~ U(-1,1), b ~ U(-0.5,0.5), a ~ U(-1,1)/sqrt(width), c = 0.5',loss:'Mean squared error',sampling:'Shuffle without replacement each epoch; matched batch order across widths'},data,runs,traces,firstUpdate:{before:first.model,after:first.after,gradient:gradient(first.model,batch),indices:first.indices,batchBefore:loss(first.model,batch),batchAfter:loss(first.after,batch)}};
writeFileSync(new URL('./experiment.json',import.meta.url),JSON.stringify(output));
for(const w of [2,12])console.log(w,runs.filter(r=>r.width===w).map(r=>({seed:r.seed,train:+r.loss.toFixed(5),grid:+r.gridMSE.toFixed(5)})));
