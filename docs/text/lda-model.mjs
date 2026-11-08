// All probabilities in this teaching diagram are constructed, not fitted estimates.
export const pelotonExample = {
  instructor: 'Chelsea Jackson Roberts',
  classTitle: 'Focused Flow: Seated Baby Cradle',
  date: 'May 13, 2021',
  excerpts: ["So I love Seated Baby Cradle, because it's an opportunity for me to practice holding my body in a way that acknowledges every part of me as being precious."],
  excerptLines: ["“So I love Seated Baby Cradle, because it's", 'an opportunity for me to practice holding', 'my body in a way that acknowledges every', 'part of me as being precious.”'],
};
export const topicLabels = ['Movement', 'Breath', 'Reflection'];
export const topicColors = ['#234e70', '#007f73', '#a54f32'];
export const wordLabels = ['cradle', 'inhale', 'precious', 'all other words'];
export const theta = [.5, .2, .3];
export const beta = [
  [.55, .10, .02, .33],
  [.05, .60, .01, .34],
  [.05, .05, .20, .70],
];
export const contentBeta = [
  [...beta[0]], [...beta[1]], [.05, .05, .35, .55],
];
pelotonExample.topics = topicLabels.map((label, k) => ({label, share: theta[k], color: topicColors[k]}));
export function wordContributions(wordIndex, mixture=theta, topics=beta) {
  return mixture.map((share,k)=>share*topics[k][wordIndex]);
}
export function wordProbability(wordIndex, mixture=theta, topics=beta) {
  return wordContributions(wordIndex,mixture,topics).reduce((a,b)=>a+b,0);
}
export function conditionalTopics(wordIndex, mixture=theta, topics=beta) {
  const contributions=wordContributions(wordIndex,mixture,topics), total=contributions.reduce((a,b)=>a+b,0);
  if(total<=0)throw new RangeError('The observed word must have positive probability.');
  return contributions.map(v=>v/total);
}
export function baselineSoftmax(logRatios) {
  const logits=[...logRatios,0], maximum=Math.max(...logits), exponentials=logits.map(v=>Math.exp(v-maximum));
  const sum=exponentials.reduce((a,b)=>a+b,0);
  return exponentials.map(v=>v/sum);
}
export const clamp = n => Math.max(0,Math.min(1,n));
export const interpolate = (from,to,q) => from+(to-from)*clamp(q);
export function interpolatedTopics(q) {
  return beta.map((row,k)=>row.map((v,j)=>interpolate(v,contentBeta[k][j],q)));
}
