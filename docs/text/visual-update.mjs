const key = node => node.nodeType === 1 ? node.getAttribute('data-key') : null;
const matches = (left, right) => left?.nodeType === right.nodeType
  && (left.nodeType !== 1 || (left.localName === right.localName && left.namespaceURI === right.namespaceURI))
  && key(left) === key(right);

/** Update a diagram in place so words, rows, and cells survive scroll boundaries. */
export function morphNode(current, next) {
  if (current.nodeType !== 1) {
    if (current.nodeValue !== next.nodeValue) current.nodeValue = next.nodeValue;
    return;
  }
  for (const attribute of [...current.attributes]) {
    if (!next.hasAttribute(attribute.name)) current.removeAttribute(attribute.name);
  }
  for (const attribute of next.attributes) {
    if (current.getAttribute(attribute.name) !== attribute.value) current.setAttribute(attribute.name, attribute.value);
  }
  let position = 0;
  for (const child of next.childNodes) {
    let existing = current.childNodes[position];
    if (!matches(existing, child)) {
      const keyed = key(child) && [...current.childNodes].slice(position + 1).find(candidate => matches(candidate, child));
      if (keyed) {
        current.insertBefore(keyed, existing ?? null);
        existing = keyed;
      } else {
        current.insertBefore(child.cloneNode(true), existing ?? null);
        position++;
        continue;
      }
    }
    morphNode(existing, child);
    position++;
  }
  while (current.childNodes.length > position) current.lastChild.remove();
}

export function updateVisual(container, markup) {
  const template = container.ownerDocument.createElement('template');
  template.innerHTML = markup;
  const next = template.content.firstElementChild;
  const current = container.firstElementChild;
  const family = next?.getAttribute('data-visual-family');
  if (family && current?.getAttribute('data-visual-family') === family) morphNode(current, next);
  else container.replaceChildren(template.content);
}
