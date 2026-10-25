import {renderMath} from './math.mjs';

const math=tex=>renderMath(tex);
const number=n=>`<span class="sgd-loop-number" aria-hidden="true">${n}</span>`;
const arrow='<span class="sgd-flow-arrow" aria-hidden="true">→</span>';
const observations=()=>`<svg class="sgd-batch-dots" viewBox="0 0 102 80" aria-hidden="true">${Array.from({length:20},(_,i)=>`<circle cx="${11+(i%5)*20}" cy="${10+Math.floor(i/5)*20}" r="4.5" fill="${i%5<2&&i<15?'#a54f32':'#d6d6d6'}"/>`).join('')}<rect x="2" y="1" width="38" height="58" rx="9" fill="none" stroke="#a54f32" stroke-width="1.3"/></svg>`;

// Initialization is outside the return path. All five steps stay visible,
// including with reduced motion; the detailed lesson animates the actual run.
export function renderOverview(){
 return `<ol class="sgd-loop" role="list" aria-label="Five steps of stochastic gradient descent">
 <li class="sgd-loop-initialize">${number(1)}<div><h3>Set weights and biases</h3><p>Initialize once.</p></div><div class="sgd-initial-parameters">${math(String.raw`(\theta,\phi)`)}</div></li>
 <li class="sgd-loop-loss">${number(2)}<h3>Calculate batch loss</h3><div class="sgd-loop-visual sgd-batch-to-loss" role="img" aria-label="Select a small batch from the data to calculate one loss value.">${observations()}${arrow}<span>${math(String.raw`L_B`)}</span></div><p>One batch → one loss value.</p></li>
 <li class="sgd-loop-gradient">${number(3)}<h3>Calculate the gradient</h3><div class="sgd-loop-visual">${math(String.raw`g_B=\nabla_{(\theta,\phi)}L_B`)}</div><p>One vector.<br> One component per parameter.</p></li>
 <li class="sgd-loop-update">${number(4)}<h3>Update the parameters</h3><div class="sgd-loop-visual">${math(String.raw`(\theta,\phi)\leftarrow(\theta,\phi)-\alpha g_B`)}</div><p>Move opposite the gradient.<br> Learning rate ${math(String.raw`\alpha`)} sets the step size.</p></li>
 <li class="sgd-loop-repeat">${number(5)}<div><h3>Repeat with a new batch</h3><p>Keep the updated parameters.</p></div></li>
 </ol><div class="sgd-overview-controls"><p><span>Other settings</span>Batch size · maximum epochs · regularization strength</p><p><span>Early stopping</span>Stop when validation improvement stalls.</p></div>`;
}
