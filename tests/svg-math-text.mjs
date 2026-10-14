// Inspect label content independently of the positioned SVG script markup.
// Existing numerical and geometry checks continue to read the outer elements.
export function svgMathText(markup) {
  const sub = Object.fromEntries([...'0123456789+−=()aehijklmnoprstuvx'].map((character, index) => [character, [...'₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ'][index]]));
  const sup = Object.fromEntries([...'0123456789+−=()T'].map((character, index) => [character, [...'⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵀ'][index]]));
  return markup.replace(/<tspan\b([^>]*)>([^<]*)<\/tspan>/g, (_, attributes, text) => {
    const alphabet = /baseline-shift="sub"/.test(attributes) ? sub : /baseline-shift="super"/.test(attributes) ? sup : null;
    return alphabet ? [...text].map(character => alphabet[character] ?? character).join('') : text;
  });
}
