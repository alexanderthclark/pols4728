export const illustration = [[105,95],[105,104],[111,115],[107,106],[94,90]];

export function fitRule(games) {
  if (!games.length) throw new RangeError('A training sample needs at least one game.');
  const differences = games.map(([knicks,spurs]) => knicks-spurs);
  const wins = differences.filter(x => x > 0);
  const losses = differences.filter(x => x <= 0);
  if (!wins.length) return { cutoff: null, constant: 0, differences };
  if (!losses.length) return { cutoff: null, constant: 1, differences };
  return { cutoff: (Math.max(...losses)+Math.min(...wins))/2, constant: null, differences };
}
export function predict(rule,x) {
  return rule.constant ?? Number(x >= rule.cutoff);
}
export function summarize(rules,x=0.5) {
  if (!rules.length) throw new RangeError('An ensemble needs at least one rule.');
  const truth = Number(x > 0);
  const predictions = rules.map(rule=>predict(rule,x));
  const wins = predictions.reduce((a,b)=>a+b,0);
  const mean = wins/rules.length;
  const bias = mean-truth;
  const variance = predictions.reduce((a,p)=>a+(p-mean)**2,0)/rules.length;
  const mse = predictions.reduce((a,p)=>a+(p-truth)**2,0)/rules.length;
  return {truth,predictions,wins,losses:rules.length-wins,mean,bias,biasSquared:bias**2,variance,mse,noise:0};
}
export function randomGenerator(seed=1) {
  let state=seed >>> 0 || 1;
  return () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0)/4294967296; };
}
export function simulate(count=1000,n=5,seed=1) {
  const random=randomGenerator(seed);
  return Array.from({length:count},()=> {
    const knicks=Array.from({length:n},()=>Math.floor(random()*101)+0.5);
    const spurs=Array.from({length:n},()=>Math.floor(random()*101));
    const games=knicks.map((k,i)=>[k,spurs[i]]);
    return {games,...fitRule(games)};
  });
}
