// A game supplies its players and coalition value. Story and drawing are separate.
export function defineGame({ id, name, players, value }) {
  const n = players.length;
  if (n < 1 || n > 8 || new Set(players).size !== n) throw new Error('Use 1–8 distinct players.');
  const nodes = Array.from({ length: 2 ** n }, (_, mask) => {
    const members = players.filter((_, i) => mask & (1 << i));
    const score = value(members, mask);
    if (!Number.isFinite(score)) throw new Error('Every coalition must have a finite value.');
    return { mask, members, size: members.length, value: score, label: members.length ? `{${members.join(', ')}}` : '∅' };
  });
  const factorial = k => k < 2 ? 1 : k * factorial(k - 1);
  const edges = nodes.flatMap(node => players.flatMap((player, i) => {
    if (node.mask & (1 << i)) return [];
    const to = node.mask | (1 << i);
    const count = factorial(node.size) * factorial(n - node.size - 1);
    return [{ id: `${node.mask}-${to}`, from: node.mask, to, player, playerIndex: i, delta: nodes[to].value - node.value, count, weight: count / factorial(n) }];
  }));
  const permute = items => items.length ? items.flatMap((item, i) => permute(items.filter((_, j) => j !== i)).map(rest => [item, ...rest])) : [[]];
  const orders = permute(players.map((_, i) => i)).map(order => {
    let mask = 0;
    const path = order.map(playerIndex => {
      const next = mask | (1 << playerIndex);
      const edge = edges.find(e => e.from === mask && e.to === next);
      mask = next;
      return edge;
    });
    return { order, label: order.map(i => players[i]).join(' → '), path };
  });
  const shares = players.map((_, i) => edges.filter(e => e.playerIndex === i).reduce((sum, e) => sum + e.weight * e.delta, 0));
  return { id, name, players, nodes, edges, orders, shares, totalOrders: factorial(n), gain: nodes.at(-1).value - nodes[0].value };
}
export const majority = defineGame({ id: 'majority-three', name: 'Majority voting', players: ['A', 'B', 'C'], value: members => Number(members.length >= 2) });
export function matchingOrders(game, edgeId) {
  if (!game.edges.some(edge => edge.id === edgeId)) throw new Error('Unknown joining edge.');
  return game.orders.filter(order => order.path.some(edge => edge.id === edgeId));
}
export function fraction(value) {
  if (Number.isInteger(value)) return String(value);
  for (let d = 2; d <= 720; d++) {
    const n = Math.round(value * d);
    if (Math.abs(n / d - value) < 1e-9) return `${n}/${d}`;
  }
  return value.toFixed(3);
}
