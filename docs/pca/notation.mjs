// Native MathML keeps changing scores and directions in the same typography
// as the lesson's static equations, without an asynchronous typesetting pass.
const mathNamespace = 'http://www.w3.org/1998/Math/MathML';
const escapeMathText = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

export const mVar = value => `<mi>${escapeMathText(value)}</mi>`;
export const mNum = value => `<mn>${escapeMathText(value)}</mn>`;
export const mOp = value => `<mo>${escapeMathText(value)}</mo>`;
export const mRow = body => `<mrow>${body}</mrow>`;
export const mSub = (base, index) => `<msub>${base}${mRow(index)}</msub>`;
export const mSup = (base, power) => `<msup>${base}${power}</msup>`;
export const mTranspose = body => mSup(mRow(body), mOp('⊤'));
export const mNorm = body => mRow(`${mOp('‖')}${body}${mOp('‖')}`);
export const mTuple = values => mRow(`<mo stretchy="false">(</mo>${values.map(mNum).join(mOp(','))}<mo stretchy="false">)</mo>`);
export const mFunction = (name, body) => mRow(`<mi mathvariant="normal">${escapeMathText(name)}</mi><mo stretchy="false" lspace="0em" rspace="0em">(</mo>${body}<mo stretchy="false" lspace="0em" rspace="0em">)</mo>`);

export function inlineMath(body) {
  const node = document.createElementNS(mathNamespace, 'math');
  node.innerHTML = body;
  return node;
}

// Strings become text nodes; only explicitly built mathematical fragments
// become MathML. Prose and accessible diagram descriptions stay ordinary text.
export function setNotation(target, parts) {
  target.replaceChildren(...(Array.isArray(parts) ? parts : [parts]));
}

// SVG descriptions and live announcements need a linear version that retains
// fraction bars, radicals, and scripts, which textContent alone would omit.
export function notationText(target) {
  const subscripts = { '0':'₀', '1':'₁', '2':'₂', '3':'₃', '4':'₄', '5':'₅', '6':'₆', '7':'₇', '8':'₈', '9':'₉', i:'ᵢ', j:'ⱼ' };
  const read = node => {
    if (node.nodeType === 3) return node.textContent;
    const children = [...node.childNodes];
    const parts = children.map(read);
    if (node.localName === 'mfrac') return `(${parts[0]})/(${parts[1]})`;
    if (node.localName === 'msqrt') return `√(${parts.join('')})`;
    if (node.localName === 'msub') return parts[0] + [...parts[1]].map(char => subscripts[char] ?? `_${char}`).join('');
    if (node.localName === 'msup') return parts[0] + (parts[1] === '⊤' ? 'ᵀ' : parts[1] === '2' ? '²' : `^(${parts[1]})`);
    return parts.join('');
  };
  return read(target);
}
