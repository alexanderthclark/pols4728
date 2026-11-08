import {escapeText as esc} from './math.mjs?v=20261108-topics-3';

// The quotations are transcript excerpts. These two topics and their shares
// are a teaching illustration, not an estimate from the Peloton corpus.
export const pelotonExample = {
  instructor: 'Chelsea Jackson Roberts',
  classTitle: 'Focused Flow: Seated Baby Cradle',
  date: 'May 13, 2021',
  excerpts: ["So I love Seated Baby Cradle, because it's an opportunity for me to practice holding my body in a way that acknowledges every part of me as being precious."],
  excerptLines: ["“So I love Seated Baby Cradle, because it's", "an opportunity for me to practice holding", "my body in a way that acknowledges every", "part of me as being precious.”"],
  topics: [
    {label: 'Movement', share: .7, words: ['seated', 'cradle', 'body'], color: '#234e70'},
    {label: 'Reflection', share: .3, words: ['precious', 'love', 'acknowledges'], color: '#a54f32'},
  ],
};

const clamp = value => Math.max(0, Math.min(1, value));
const fixed = value => Number(value.toFixed(4));
const text = (key, x, y, label, size=30, extra='') => `<text data-key="${key}" x="${x}" y="${y}" font-size="${size}" ${extra}>${esc(label)}</text>`;

export function topicModelsVisual(stage, p=stage, motion=true) {
  if (![0, 1, 2].includes(stage)) throw new RangeError(`Unknown topic-model scene: ${stage}`);
  const q = motion ? clamp((p-stage+.5)*2) : 1;
  const mix = stage === 0 ? 0 : stage === 1 ? q : 1;
  const metadata = stage === 2 ? q : 0;
  const grow = clamp((mix-.15)/.7);
  const topicReveal = clamp((mix-.35)/.5);
  const metadataReveal = clamp(metadata*3);
  const arrow = clamp((metadata-.3)/.7);
  const lift = fixed(115*(1-mix));
  const barWidth = 624;
  const description = mix === 0
    ? 'Chelsea Jackson Roberts reflects on Seated Baby Cradle in a Focused Flow class, combining movement and personal reflection.'
    : metadata === 0
      ? 'An illustrative class mixture is 70 percent Movement and 30 percent Reflection. The topics are distributions over words; selected words appear below. These are not fitted results.'
      : 'The same illustrative mixture and topic words remain visible. Instructor and class-type metadata enter a model of topic prevalence. No numerical metadata effect is estimated in this diagram.';
  return `<div class="topic-model-flow" data-key="topic-model-flow" data-visual-family="topic-models" data-mixture-progress="${fixed(mix)}" data-metadata-progress="${fixed(metadata)}">
    <svg class="topic-model-workspace" viewBox="0 0 680 610" role="img" aria-label="${esc(`${pelotonExample.instructor}: “${pelotonExample.excerpts[0]}” ${description}`)}">
      <g data-key="class-excerpt" transform="translate(0 ${lift})">
        ${text('class-author', 28, 32, pelotonExample.instructor, 31)}
        ${text('class-citation', 28, 64, `${pelotonExample.classTitle} · ${pelotonExample.date}`, 25, 'fill="#606060"')}
        ${pelotonExample.excerptLines.map((quote,i) => text(`excerpt-${i}`, 28, 100+i*34, quote, 30, 'fill="#242424"')).join('')}
      </g>
      <g data-key="metadata" opacity="${fixed(metadataReveal)}" aria-hidden="${metadataReveal===0}">
        <g data-key="instructor-metadata" transform="translate(${fixed(-55*(1-metadata))} 0)">
          <line x1="28" y1="220" x2="307" y2="220" stroke="#767676" stroke-width="1.5"/>
          ${text('instructor-label', 28, 248, 'Instructor', 24, 'fill="#606060"')}
          ${text('instructor-value', 28, 281, pelotonExample.instructor, 25)}
        </g>
        <g data-key="type-metadata" transform="translate(${fixed(55*(1-metadata))} 0)">
          <line x1="365" y1="220" x2="652" y2="220" stroke="#767676" stroke-width="1.5"/>
          ${text('class-type-label', 365, 248, 'Class type', 24, 'fill="#606060"')}
          ${text('class-type-value', 365, 281, 'Yoga', 30)}
        </g>
        <path data-key="metadata-arrow" d="M168 291 V301 H340 V330 M510 291 V301 H340" fill="none" stroke="#767676" stroke-width="2" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${fixed(1-arrow)}"/>
        <path data-key="metadata-arrowhead" d="M334 322 L340 330 L346 322" fill="none" stroke="#767676" stroke-width="2" opacity="${fixed(clamp((arrow-.9)*10))}"/>
      </g>
      <g data-key="mixture" opacity="${fixed(clamp(mix*3))}" aria-hidden="${mix===0}">
        ${text('mixture-label', 28, 326, 'Topic mixture', 30)}
        <rect data-key="mixture-outline" x="28" y="344" width="624" height="46" fill="#fff" stroke="#767676" stroke-width="1.5"/>
        ${pelotonExample.topics.map((topic,i) => {
          const previous = pelotonExample.topics.slice(0,i).reduce((sum,t)=>sum+t.share,0);
          const width = fixed(barWidth*topic.share*grow), x = fixed(28+barWidth*previous*grow);
          return `<g data-key="mixture-topic-${i}" data-share="${topic.share}"><rect data-key="mixture-segment-${i}" x="${x}" y="344" width="${width}" height="46" fill="${topic.color}"/>${text(`mixture-percent-${i}`, fixed(28+barWidth*(previous+topic.share/2)), 377, `${Math.round(topic.share*100)}%`, 31, `text-anchor="middle" fill="#fff" opacity="${fixed(clamp((grow-.8)*5))}"`)}</g>`;
        }).join('')}
      </g>
      <g data-key="topic-words" opacity="${fixed(topicReveal)}" aria-hidden="${topicReveal===0}" transform="translate(0 ${fixed(30*(1-topicReveal))})">
        ${pelotonExample.topics.map((topic,i) => `<g data-key="topic-words-${i}" fill="${topic.color}">
          ${text(`topic-label-${i}`, i===0?28:365, 438, topic.label, 33)}
          ${topic.words.map((word,j)=>text(`topic-word-${i}-${j}`, i===0?28:365, 478+j*34, word, 30)).join('')}
        </g>`).join('')}
      </g>
      ${text('illustration-note', 340, 596, 'Illustrative topics and proportions', 25, `text-anchor="middle" fill="#606060" opacity="${fixed(clamp(mix*3))}" aria-hidden="${mix===0}"`)}
    </svg>
  </div>`;
}
