(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.SvgMath = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const escape = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  const subscripts = Object.fromEntries([...('₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ')].map((character, index) => [character, [...('0123456789+−=()aehijklmnoprstuvx')][index]]));
  const superscripts = Object.fromEntries([...('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵀ')].map((character, index) => [character, [...('0123456789+−=()T')][index]]));
  const variable = character => /[xyzuvFfhrLRSabcdi]|[α-ωΑ-Ω]|ℓ/.test(character);
  const wordCharacter = character => Boolean(character) && /[A-Za-z]/.test(character);
  const identifierCharacter = character => Boolean(character) && /[\p{L}\p{N}_]/u.test(character);

  // The source strings stay useful as accessible text and numerical readouts.
  // Only their visible scripts and isolated mathematical variables are styled.
  function tokens(value, italic = true) {
    const source = String(value), result = [];
    const append = (kind, text) => {
      if (result.length && result[result.length - 1].kind === kind) result[result.length - 1].text += text;
      else result.push({ kind, text });
    };
    for (let index = 0; index < source.length;) {
      const scientific = /^(\d+(?:\.\d+)?)e([+−-]?\d+)/.exec(source.slice(index));
      if (scientific) {
        append('text', `${scientific[1]} × 10`);
        append('sup', scientific[2].replace('-', '−'));
        index += scientific[0].length;
        continue;
      }
      const exponent = /^10\^([+−-]?\d+)/.exec(source.slice(index));
      if (exponent) {
        append('text', '10');
        append('sup', exponent[1].replace('-', '−'));
        index += exponent[0].length;
        continue;
      }
      const character = source[index];
      if (Object.hasOwn(subscripts, character)) append('sub', subscripts[character]);
      else if (Object.hasOwn(superscripts, character)) append('sup', superscripts[character]);
      else if (italic && variable(character) && !wordCharacter(source[index - 1]) && !wordCharacter(source[index + 1]) && !(character === 'a' && /^\s+[A-Za-z]{2}/.test(source.slice(index + 1)))) append('variable', character);
      else append('text', character);
      index++;
    }
    return result;
  }

  function svg(value, italic = true) {
    return tokens(value, italic).map(({ kind, text }) => {
      if (kind === 'variable') return `<tspan font-family="'Course Math','Cambria Math',serif" font-style="italic">${escape(text)}</tspan>`;
      if (kind === 'sub' || kind === 'sup') return `<tspan baseline-shift="${kind === 'sup' ? 'super' : 'sub'}" font-family="'Course Math','Cambria Math',serif" font-size="75%" font-style="${kind === 'sub' && /^[a-z]+$/.test(text) ? 'italic' : 'normal'}">${escape(text)}</tspan>`;
      return escape(text);
    }).join('');
  }

  function html(value, italic = true) {
    return tokens(value, italic).map(({ kind, text }) => kind === 'text' ? escape(text) : `<${kind === 'variable' ? 'i' : kind}>${escape(text)}</${kind === 'variable' ? 'i' : kind}>`).join('');
  }

  function setText(element, document, value, { svg: inSvg = false, italic = true } = {}) {
    element.setAttribute('aria-label', String(value));
    const children = tokens(value, italic).map(({ kind, text }) => {
      const child = inSvg ? document.createElementNS('http://www.w3.org/2000/svg', 'tspan') : document.createElement(kind === 'variable' ? 'i' : kind === 'text' ? 'span' : kind);
      if (inSvg && kind !== 'text') child.setAttribute('font-family', "'Course Math','Cambria Math',serif");
      if (inSvg && kind === 'variable') child.setAttribute('font-style', 'italic');
      if (inSvg && (kind === 'sub' || kind === 'sup')) {
        child.setAttribute('baseline-shift', kind === 'sup' ? 'super' : 'sub');
        child.setAttribute('font-size', '75%');
        child.setAttribute('font-style', kind === 'sub' && /^[a-z]+$/.test(text) ? 'italic' : 'normal');
      }
      child.textContent = text;
      return child;
    });
    element.replaceChildren(...children);
  }

  // Explicit builders keep prose out of the math renderer. The source values
  // remain unchanged; only fractions, scripts, accents, and spacing are laid out.
  const notation = {
    variable: value => `<mi>${escape(value)}</mi>`,
    operator: value => `<mo${/^[()[\]{}]$/.test(value) ? ' stretchy="false"' : ''}>${escape(value)}</mo>`,
    row: body => `<mrow>${body}</mrow>`,
    subscript: (base, index) => `<msub>${base}<mrow>${index}</mrow></msub>`,
    power: (base, exponent) => `<msup>${base}<mrow>${exponent}</mrow></msup>`,
    fraction: (numerator, denominator) => `<mfrac><mrow>${numerator}</mrow><mrow>${denominator}</mrow></mfrac>`,
    bar: base => `<mover accent="true">${base}<mo>¯</mo></mover>`,
    sum: (index, body) => `<munder><mo>∑</mo><mrow>${index}</mrow></munder>${body}`,
  };
  notation.number = value => {
    const source = String(value), scientific = /^(.+?)(?: × 10\^|e)([+−-]?\d+)$/.exec(source), fraction = /^([+−-]?\d+)\/(\d+)$/.exec(source);
    if (scientific) return `${notation.number(scientific[1])}${notation.operator('×')}${notation.power('<mn>10</mn>', `<mn>${escape(scientific[2].replace('-', '−'))}</mn>`)}`;
    if (fraction) return notation.fraction(`<mn>${escape(fraction[1])}</mn>`, `<mn>${escape(fraction[2])}</mn>`);
    return `<mn>${escape(source)}</mn>`;
  };
  notation.apply = (name, body, upright = false) => notation.row(`${upright ? `<mi mathvariant="normal">${escape(name)}</mi>` : name}${notation.operator('(')}${body}${notation.operator(')')}`);
  notation.norm = body => notation.row(`${notation.operator('‖')}${body}${notation.operator('‖')}`);
  notation.tuple = values => notation.row(`${notation.operator('(')}${values.map(notation.number).join(notation.operator(','))}${notation.operator(')')}`);
  notation.math = (body, label = '', display = false) => `<math xmlns="http://www.w3.org/1998/Math/MathML"${display ? ' display="block"' : ''}${label ? ` aria-label="${escape(label)}"` : ''}><mrow>${body}</mrow></math>`;
  notation.part = (body, label = '') => ({ math: body, label });
  notation.replaceParts = (source, replacements) => {
    const parts=[],choices=replacements.filter(([text])=>text).sort((a,b)=>b[0].length-a[0].length);
    let start=0,index=0;
    while(index<source.length){
      const found=choices.find(([text])=>source.startsWith(text,index) && (!/^\p{L}$/u.test(text) || (!identifierCharacter(source[index-1]) && !identifierCharacter(source[index+text.length]))));
      if(!found){index++;continue;}
      if(index>start)parts.push(source.slice(start,index));
      parts.push(notation.part(found[1],found[0]));
      index+=found[0].length;start=index;
    }
    if(start<source.length)parts.push(source.slice(start));
    return parts;
  };
  notation.parts = parts => (Array.isArray(parts) ? parts : [parts]).map(part => typeof part === 'string' ? escape(part) : notation.math(part.math, part.label)).join('');
  notation.setParts = (element, parts, label = '') => {
    if (label) element.setAttribute('aria-label', label); else element.removeAttribute?.('aria-label');
    element.innerHTML = notation.parts(parts);
  };

  return { svg, html, setText, notation };
});
