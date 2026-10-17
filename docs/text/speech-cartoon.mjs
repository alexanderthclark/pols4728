import {speechFrame} from './speech-motion.mjs';

const escapeText = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');

/** The image supplies the character art; words and changing counts stay real text. */
export function speechCartoon(progress, {reducedMotion=false, id='speech', showCounts=true}={}) {
  const frame = speechFrame(progress, {reducedMotion});
  const prefix = id.replace(/[^a-z0-9-]/gi, '');
  const tilt = -frame.gesture * 0.9;
  const lean = frame.catcherLean * 0.7;
  const lowerLip = 258 + frame.mouthOpen * 13;
  const tokens = frame.tokens.map(token => `
    <text data-speech-word="${token.index}" x="0" y="0"
      transform="translate(${token.x.toFixed(3)} ${token.y.toFixed(3)}) rotate(${token.rotation.toFixed(2)}) scale(${token.scale.toFixed(3)})"
      opacity="${token.opacity.toFixed(3)}" text-anchor="middle" dominant-baseline="middle"
      fill="#234e70" font-size="42">${escapeText(token.word)}</text>`).join('');
  const counts = Object.entries(frame.caughtCounts).filter(([,count])=>count>0);
  return `<div class="speech-cartoon" data-speech-progress="${progress.toFixed(4)}" data-caught="${frame.caughtTotal}">
    <svg class="speech-art" viewBox="0 0 1000 480" role="img"
      aria-label="A Lincoln-like round, torso-less figure speaks as separate words float upward into a sack held by another figure. ${frame.caughtTotal} of 10 words have been caught.">
      <defs>
        <clipPath id="${prefix}-speaker"><rect width="600" height="480"/></clipPath>
        <clipPath id="${prefix}-catcher"><rect x="650" width="350" height="480"/></clipPath>
        <clipPath id="${prefix}-words"><path d="M0 0H1000V285H680V480H0Z"/></clipPath>
      </defs>
      <g transform="rotate(${tilt.toFixed(3)} 195 420)">
        <image href="assets/speech-bag.png" width="1000" height="460.747" clip-path="url(#${prefix}-speaker)"/>
        <ellipse cx="210" cy="257" rx="25" ry="21" fill="white"/>
        <path class="speaking-mouth" d="M193 252 Q210 263 229 248 Q227 ${lowerLip.toFixed(2)} 211 ${lowerLip.toFixed(2)} Q200 ${lowerLip.toFixed(2)} 193 252Z" fill="#111"/>
      </g>
      <g transform="rotate(${lean.toFixed(3)} 900 420)">
        <image href="assets/speech-bag.png" width="1000" height="460.747" clip-path="url(#${prefix}-catcher)"/>
      </g>
      <g class="floating-words" clip-path="url(#${prefix}-words)">${tokens}</g>
    </svg>
    ${showCounts?`<div class="speech-tally"><span class="caught-total">${frame.caughtTotal} / 10 words in the bag</span><div class="caught-counts" aria-label="Word counts in the bag">${counts.length?counts.map(([word,count])=>`<span>${word} <span class="word-count">${count}</span></span>`).join(''):'<span>Each repeated word counts again.</span>'}</div></div>`:''}
  </div>`;
}
