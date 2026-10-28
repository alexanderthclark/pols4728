const cache = new Map();

export const escapeText = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');

/** The lesson and its static reading version share the same mathematical typography. */
export function latex(tex, display = false) {
  const key = `${display}:${tex}`;
  if (!cache.has(key)) cache.set(key, globalThis.katex.renderToString(tex, {
    throwOnError: true, displayMode: display, output: 'htmlAndMathml',
  }));
  return cache.get(key);
}

export function prose(text = '', links = []) {
  let result = text.split(/(\\\([\s\S]*?\\\))/g).map(part =>
    part.startsWith('\\(') ? latex(part.slice(2, -2)) : escapeText(part)
  ).join('');
  for (const link of links) result = result.replace(escapeText(link.label), `<a href="${escapeText(link.url)}">${escapeText(link.label)}</a>`);
  return result;
}

export function mathLabel(x, y, width, height, tex, size = 26) {
  return `<foreignObject x="${x}" y="${y}" width="${width}" height="${height}"><div xmlns="http://www.w3.org/1999/xhtml" class="svg-math" style="font-size:${size}px">${latex(tex)}</div></foreignObject>`;
}
