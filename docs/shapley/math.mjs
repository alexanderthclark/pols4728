// Native mathematical notation shared by the coalition story and its views.
export const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
export const variable = (name, upright = false) => `<mi${upright ? ' mathvariant="normal"' : ''}>${escape(name)}</mi>`;
export const operator = value => `<mo>${escape(value)}</mo>`;
export const row = (...parts) => `<mrow>${parts.join('')}</mrow>`;
export const quotient = (numerator, denominator) => `<mfrac>${row(numerator)}${row(denominator)}</mfrac>`;
export function number(value) {
  const raw = String(value);
  const rational = raw.match(/^(-?\d+)\/(\d+)$/);
  if (rational) return row(rational[1].startsWith('-') ? operator('−') : '', quotient(`<mn>${escape(rational[1].replace(/^-/, ''))}</mn>`, `<mn>${escape(rational[2])}</mn>`));
  return raw.startsWith('-') ? row(operator('−'), `<mn>${escape(raw.slice(1))}</mn>`) : `<mn>${escape(raw)}</mn>`;
}
export const fence = (content, open = '(', close = ')') => row(`<mo stretchy="false" lspace="0" rspace="0">${escape(open)}</mo>`, content, `<mo stretchy="false" lspace="0" rspace="0">${escape(close)}</mo>`);
export const cardinality = content => fence(content, '|', '|');
export const factorial = content => row(content, '<mo form="postfix" lspace="0" rspace="0">!</mo>');
export const coalition = members => members.length ? fence(row(...members.flatMap((player, i) => [i ? operator(',') : '', variable(player, true)])), '{', '}') : operator('∅');
export const order = players => players.length ? row(...players.flatMap((player, i) => [i ? operator('→') : '', variable(player, true)])) : operator('∅');
export const valueApplication = node => row(variable('v'), '<mo lspace="0" rspace="0">⁡</mo>', fence(coalition(node.members)));
export function math(content, { className = 'math-inline', ariaLabel = '', hidden = false } = {}) {
  return `<math xmlns="http://www.w3.org/1998/Math/MathML"${className ? ` class="${escape(className)}"` : ''}${ariaLabel ? ` aria-label="${escape(ariaLabel)}"` : ''}${hidden ? ' aria-hidden="true"' : ''}>${row(content)}</math>`;
}
