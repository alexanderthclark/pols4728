import {dirichletVisual} from './dirichlet-visual.mjs?v=20261108-mechanism-1';
import {ldaMechanismVisual} from './lda-mechanism.mjs?v=20261108-mechanism-1';

export {pelotonExample} from './lda-mechanism.mjs?v=20261108-mechanism-1';

// The simplex leads into the same categorical-mixture mechanism used by LDA.
export function topicModelsVisual(stage,p=stage,motion=true){
  if(!Number.isInteger(stage)||stage<0||stage>9)throw new RangeError(`Unknown topic-model scene: ${stage}`);
  return stage<3?dirichletVisual(stage,p,motion):ldaMechanismVisual(stage-3,p-3,motion);
}
