// Exact Shapley–Shubik power for a weighted majority game.
// A player's numerator counts the orders in which that player crosses the quota.
// Floating-point shares are only a display convenience; all counting uses BigInt.
function ratio(numerator, denominator) {
  if (numerator === 0n) return 0;
  const n = numerator.toString();
  const d = denominator.toString();
  const nLead = n.slice(0, 17);
  const dLead = d.slice(0, 17);
  return Number(nLead) / Number(dLead) * 10 ** ((n.length - nLead.length) - (d.length - dLead.length));
}

export function exactWeightedVotingPower(weights, quota) {
  if (!Array.isArray(weights) || weights.length === 0 || weights.some(weight => !Number.isSafeInteger(weight) || weight < 0)) {
    throw new Error('Use a nonempty array of nonnegative integer voting weights.');
  }
  if (!Number.isSafeInteger(quota) || quota < 1) throw new Error('Use a positive integer quota.');
  const n = weights.length;
  const totalVotes = weights.reduce((sum, weight) => sum + weight, 0);
  if (!Number.isSafeInteger(totalVotes)) throw new Error('The sum of voting weights must be a safe integer.');
  const factorial = [1n];
  for (let k = 1; k <= n; k++) factorial[k] = factorial[k - 1] * BigInt(k);
  const totalOrders = factorial[n];
  const result = numerators => {
    const values = numerators.map(numerator => ({ numerator, denominator: totalOrders, share: ratio(numerator, totalOrders) }));
    return { quota, totalVotes, totalOrders, values, shares: values.map(value => value.share) };
  };
  if (quota > totalVotes) return result(weights.map(() => 0n));

  // coalitions[k][s] is the coefficient of y^k x^s in ∏(1 + y x^weight).
  // Only losing totals s < quota can precede a pivotal player.
  const coalitions = Array.from({ length: n + 1 }, () => Array(quota).fill(0n));
  coalitions[0][0] = 1n;
  weights.forEach((weight, index) => {
    for (let k = index + 1; k >= 1; k--) {
      for (let sum = weight; sum < quota; sum++) coalitions[k][sum] += coalitions[k - 1][sum - weight];
    }
  });

  // Divide out one factor (1 + y x^weight). Equal-weight players share a result.
  const byWeight = new Map();
  for (const weight of new Set(weights)) {
    let previous = Array(quota).fill(0n);
    let numerator = 0n;
    for (let k = 0; k < n; k++) {
      const withoutPlayer = Array(quota);
      let pivotalCoalitions = 0n;
      for (let sum = 0; sum < quota; sum++) {
        withoutPlayer[sum] = coalitions[k][sum] - (k > 0 && sum >= weight ? previous[sum - weight] : 0n);
        if (sum >= quota - weight) pivotalCoalitions += withoutPlayer[sum];
      }
      // Each such predecessor coalition occurs in k! (n-k-1)! orders.
      numerator += pivotalCoalitions * factorial[k] * factorial[n - k - 1];
      previous = withoutPlayer;
    }
    byWeight.set(weight, numerator);
  }
  return result(weights.map(weight => byWeight.get(weight)));
}

export const ELECTORAL_QUOTA = 270;
export const ELECTORAL_TOTAL = 538;
export const ELECTORAL_SOURCE = 'https://www.archives.gov/electoral-college/allocation';

// 2020-census allocation, used in the 2024 and 2028 presidential elections.
// This teaching game treats every state and DC as one winner-take-all bloc,
// including Maine and Nebraska; it describes structural, not forecast, power.
export const ELECTORAL_STATES = Object.freeze([
  ['AL', 'Alabama', 9], ['AK', 'Alaska', 3], ['AZ', 'Arizona', 11],
  ['AR', 'Arkansas', 6], ['CA', 'California', 54], ['CO', 'Colorado', 10],
  ['CT', 'Connecticut', 7], ['DE', 'Delaware', 3], ['DC', 'District of Columbia', 3],
  ['FL', 'Florida', 30], ['GA', 'Georgia', 16], ['HI', 'Hawaii', 4],
  ['ID', 'Idaho', 4], ['IL', 'Illinois', 19], ['IN', 'Indiana', 11],
  ['IA', 'Iowa', 6], ['KS', 'Kansas', 6], ['KY', 'Kentucky', 8],
  ['LA', 'Louisiana', 8], ['ME', 'Maine', 4], ['MD', 'Maryland', 10],
  ['MA', 'Massachusetts', 11], ['MI', 'Michigan', 15], ['MN', 'Minnesota', 10],
  ['MS', 'Mississippi', 6], ['MO', 'Missouri', 10], ['MT', 'Montana', 4],
  ['NE', 'Nebraska', 5], ['NV', 'Nevada', 6], ['NH', 'New Hampshire', 4],
  ['NJ', 'New Jersey', 14], ['NM', 'New Mexico', 5], ['NY', 'New York', 28],
  ['NC', 'North Carolina', 16], ['ND', 'North Dakota', 3], ['OH', 'Ohio', 17],
  ['OK', 'Oklahoma', 7], ['OR', 'Oregon', 8], ['PA', 'Pennsylvania', 19],
  ['RI', 'Rhode Island', 4], ['SC', 'South Carolina', 9], ['SD', 'South Dakota', 3],
  ['TN', 'Tennessee', 11], ['TX', 'Texas', 40], ['UT', 'Utah', 6],
  ['VT', 'Vermont', 3], ['VA', 'Virginia', 13], ['WA', 'Washington', 12],
  ['WV', 'West Virginia', 4], ['WI', 'Wisconsin', 10], ['WY', 'Wyoming', 3],
].map(([abbreviation, name, votes]) => Object.freeze({ abbreviation, name, votes }))
  .sort((a, b) => b.votes - a.votes || a.name.localeCompare(b.name)));

// Precomputed with exactWeightedVotingPower; tests reproduce every integer.
// Store one numerator per distinct weight so the story loads without doing DP.
const ELECTORAL_TOTAL_ORDERS = 1551118753287382280224243016469303211063259720016986112000000000000n;
const PIVOTAL_ORDERS = Object.freeze({
  3: '8379575581199454411772520249931101109689577180254699520000000000',
  4: '11192400820095117076027491920901159736461883949847674880000000000',
  5: '14015211534202232092452478299389428593708873919475220480000000000',
  6: '16848112109549624574280007591659903865255621989446451200000000000',
  7: '19691209023153332473012944617033592735594228313353093120000000000',
  8: '22544610983946814025766726133362680022356761374999183360000000000',
  9: '25408429016814937747246570914878874092645112526594375680000000000',
  10: '28282776518250001047714532785874268129651201006622474240000000000',
  11: '31167769261777146229032400256854786859445375596117360640000000000',
  12: '34063525654465113391566039772157922537213834219497717760000000000',
  13: '36970166815076820654928008152963671932700076455262945280000000000',
  14: '39887816286295583573694135989247218706331272401293148160000000000',
  15: '42816600955376437693452757680107958880901337927018086400000000000',
  16: '45756649850955747369886851333304431566907274399391416320000000000',
  17: '48708096043112026269570317934644667914757863423166382080000000000',
  19: '54645726254216515744957994140975358064494934221106708480000000000',
  28: '81967611670926160115108120688460752144997256604376104960000000000',
  30: '88180803606555866016368144325754851705476466690553282560000000000',
  40: '120100422117080468125868527134714653256085478205114286080000000000',
  54: '167577958723249493436290712077424366610393535940887838720000000000',
});

export const electoralResults = Object.freeze(ELECTORAL_STATES.map(state => {
  const numerator = PIVOTAL_ORDERS[state.votes];
  const share = ratio(BigInt(numerator), ELECTORAL_TOTAL_ORDERS);
  const voteShare = state.votes / ELECTORAL_TOTAL;
  return Object.freeze({ ...state, share, voteShare, relativePower: share / voteShare,
    numerator, denominator: ELECTORAL_TOTAL_ORDERS.toString() });
}));
