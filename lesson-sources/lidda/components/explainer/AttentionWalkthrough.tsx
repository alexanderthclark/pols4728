"use client";

import { useMemo, useState } from "react";
import { TOUCHES } from "../../lib/attribution-model";
import { DETAIL_VALUES, traceAttention, traceWholeModel } from "../../lib/attention-detail";
import { MathText, PaperLink } from "./MathText";
import "./attention-walkthrough.css";

const STAGES = ["Encode", "Project", "Compare", "Normalize", "Mix values", "Predict", "Read credit", "Train"];
const f = (x: number) => x.toFixed(3);
const percent = (x: number) => `${(100 * x).toFixed(1)}%`;
const vector = (xs: readonly number[]) => `(${xs.map(x => x.toFixed(1)).join(",\\,")})`;

export function AttentionWalkthrough() {
  const [head, setHead] = useState(0);
  const [query, setQuery] = useState(0);
  const row = useMemo(() => traceAttention(head, query), [head, query]);
  const whole = useMemo(() => traceWholeModel(), []);
  return <section className="paper-section attention-detail" id="attention-detail">
    <span className="eyebrow">Inside the attention mechanism</span>
    <h2>Follow one weight,<br />from an event to a credit.</h2>
    <p className="section-lede">A channel does not have one permanent attention coefficient. The model learns shared parameters, then computes new weights for each journey. Walk through the arithmetic behind the same four-touch example.</p>
    <p className="aw-example-note">All numerical values here are fixed teaching parameters, used throughout this lesson. They were not fitted to data and are not LinkedIn estimates. Displayed numbers are rounded. <PaperLink section="S2.SS4">LiDDA §2.4</PaperLink></p>
    <nav className="aw-stage-nav" id="attention-steps" aria-label="Attention calculation steps">{STAGES.map((label, index) => <a key={label} href={`#attention-step-${index + 1}`}><span>{String(index + 1).padStart(2, "0")}</span>{label}</a>)}</nav>
    <p className="aw-example-note">Scroll through all eight steps in order, or use the links above to jump to a calculation. The head and query controls update the worked numbers throughout.</p>
    <div className="aw-controls"><label>Attention head<select aria-label="Attention head" value={head} onChange={e => setHead(Number(e.target.value))}><option value={0}>Head 1</option><option value={1}>Head 2</option></select></label><label>Query touch<select aria-label="Query touch" value={query} onChange={e => setQuery(Number(e.target.value))}>{TOUCHES.map((t, i) => <option key={t.id} value={i}>{t.label}</option>)}</select></label><p>Choose the query doing the reading in steps 2–5. All four keys remain in the path.</p></div>
    <div className="aw-scroll-sequence" id="attention-stage">
      <article className="aw-stage aw-scroll-step" id="attention-step-1" aria-labelledby="attention-step-title-1">
        <p className="aw-counter">Step 1 of 8 · Encode</p>
        <h3 id="attention-step-title-1">Describe the event before assigning it any credit.</h3>
        <p>A touch records a channel and an action, campaign information, and timing. Its channel identifies where this company bought or delivered the ad. It is not the identity of a competing advertiser.</p>
        <div className="aw-event"><span className="aw-event-color" style={{ backgroundColor: TOUCHES[query].color }}/><div><strong>{TOUCHES[query].label}</strong><span>{TOUCHES[query].action} · campaign information · {Math.abs(TOUCHES[query].day)} days before the prediction cutoff</span></div></div>
        <div className="aw-encoding-grid"><div><strong>What happened?</strong><span>Learned channel/action embedding</span></div><div><strong>Which message?</strong><span>Campaign-content embedding</span></div><div><strong>When?</strong><span>Position, elapsed days, and day of week</span></div></div>
        <MathText block>{"\\begin{aligned}x_j&=\\operatorname{concat}(E_{\\mathrm{MCID},j},E_{T,j})\\\\&\\quad+E_{D,j}+E_{\\mathrm{DOW},j}+\\mathrm{tAPE}_j\\end{aligned}"}</MathText>
        <p className="aw-small">Each <MathText>{"x_j"}</MathText> is a vector for touch <MathText>{"j"}</MathText>; stacking the vectors gives <MathText>{"X"}</MathText>. The paper uses timing relative to the final conversion state. For a prospective response model, an advertiser must define a prediction cutoff and comparable non-conversion windows; the future outcome cannot be a feature.</p>
      </article>
      <article className="aw-stage aw-scroll-step" id="attention-step-2" aria-labelledby="attention-step-title-2">
        <p className="aw-counter">Step 2 of 8 · Project</p>
        <h3 id="attention-step-title-2">Make queries, keys, and values from those vectors.</h3>
        <div className="aw-role-grid"><div><MathText>{"Q=XW_Q"}</MathText><strong>Query</strong><p>Used to score which other touch representations to read.</p></div><div><MathText>{"K=XW_K"}</MathText><strong>Key</strong><p>Compared with a query to calculate compatibility.</p></div><div><MathText>{"V=XW_V"}</MathText><strong>Value</strong><p>The information that the attention weights will mix.</p></div></div>
        <p>Each head has its own projection parameters. The matrices <MathText>{"W_Q,W_K,W_V"}</MathText> are learned across training journeys. The resulting queries, keys, and values depend on this journey’s inputs.</p>
        <div className="aw-selected"><span>Selected query: {TOUCHES[query].shortLabel}, head {head + 1}</span><MathText>{`q_r=${vector(row.q)},\\qquad d_k=2`}</MathText></div>
        <p className="aw-small">Here <MathText>{"r"}</MathText> indexes the query touch and <MathText>{"d_k"}</MathText> is the key/query dimension. To keep the arithmetic inspectable, the demo specifies these resulting vectors directly and uses scalar values. A real model usually carries vector values.</p>
      </article>
      <article className="aw-stage aw-scroll-step" id="attention-step-3" aria-labelledby="attention-step-title-3">
        <p className="aw-counter">Step 3 of 8 · Compare</p>
        <h3 id="attention-step-title-3">Compare this query with every key.</h3>
        <p>Multiply matching coordinates and add them. Then divide by the square root of the key dimension. The result is a compatibility score, before normalization into a mixing weight.</p>
        <MathText block>{"s_{rj}=\\frac{q_r^{\\top}k_j}{\\sqrt{d_k}}"}</MathText>
        <div className="aw-selected"><span>Query from {TOUCHES[query].shortLabel}</span><MathText>{`q_r=${vector(row.q)}`}</MathText></div>
        <div className="aw-number-grid">{TOUCHES.map((touch, j) => <div className="aw-number-card" key={touch.id} style={{ borderTopColor: touch.color }}><strong>{touch.shortLabel} key</strong><MathText>{`k_${j + 1}=${vector(row.keys[j])}`}</MathText><dl><div><dt><MathText>{"q_r^\\top k_j"}</MathText></dt><dd>{f(row.dots[j])}</dd></div><div className="aw-emphasis"><dt>Scaled score</dt><dd>{f(row.scores[j])}</dd></div></dl></div>)}</div>
        <p className="aw-small">The same key can receive different scores from different queries. Change the selected query or head to see the complete calculation change.</p>
      </article>
      <article className="aw-stage aw-scroll-step" id="attention-step-4" aria-labelledby="attention-step-title-4">
        <p className="aw-counter">Step 4 of 8 · Normalize</p>
        <h3 id="attention-step-title-4">Softmax turns the scores into one row of weights.</h3>
        <p>Exponentiate every score and divide by their shared total. An included, unmasked touch receives a positive weight. These weights tell the model how to combine the value information for this query.</p>
        <MathText block>{"A_{rj}=\\frac{\\exp(s_{rj})}{\\sum_k\\exp(s_{rk})},\\qquad \\sum_j A_{rj}=1"}</MathText>
        <div className="aw-number-grid">{TOUCHES.map((touch, j) => <div className="aw-number-card" key={touch.id} style={{ borderTopColor: touch.color }}><strong>{touch.shortLabel}</strong><dl><div><dt>Scaled score</dt><dd>{f(row.scores[j])}</dd></div><div><dt><MathText>{"\\exp(s_{rj})"}</MathText></dt><dd>{f(row.exponents[j])}</dd></div><div className="aw-emphasis"><dt>Row weight</dt><dd>{percent(row.weights[j])}</dd></div></dl></div>)}</div>
        <div className="aw-total"><span>Shared denominator</span><strong>{f(row.denominator)}</strong><span>Row total</span><strong>100%</strong></div>
        <p className="aw-small">These are the four cells in the selected query’s row. A column belongs to the touch receiving attention. The full matrix repeats this operation for every query.</p>
      </article>
      <article className="aw-stage aw-scroll-step" id="attention-step-5" aria-labelledby="attention-step-title-5">
        <p className="aw-counter">Step 5 of 8 · Mix values</p>
        <h3 id="attention-step-title-5">The weights mix values into a new representation.</h3>
        <p>Keep the softmax weights fixed and multiply each one by its corresponding value. The sum is the output for this query and head. The value can carry information with a different sign or scale from its attention weight.</p>
        <MathText block>{"z_r^{(h)}=\\sum_j A_{rj}^{(h)}v_j^{(h)}"}</MathText>
        <div className="aw-number-grid">{TOUCHES.map((touch, j) => <div className="aw-number-card" key={touch.id} style={{ borderTopColor: touch.color }}><strong>{touch.shortLabel}</strong><dl><div><dt>Weight</dt><dd>{percent(row.weights[j])}</dd></div><div><dt>Scalar value</dt><dd>{DETAIL_VALUES[j].toFixed(1)}</dd></div><div className="aw-emphasis"><dt>Weighted value</dt><dd>{f(row.contributions[j])}</dd></div></dl></div>)}</div>
        <div className="aw-total"><span>Sum of weighted values</span><MathText>{`z_${query + 1}^{(${head + 1})}=${f(row.output)}`}</MathText></div>
        <p className="aw-small">This sum is an internal representation. It is not a conversion probability. In the full model, additional operations convert attention outputs into a prediction.</p>
      </article>
      <article className="aw-stage aw-scroll-step" id="attention-step-6" aria-labelledby="attention-step-title-6">
        <p className="aw-counter">Step 6 of 8 · Predict</p>
        <h3 id="attention-step-title-6">Use all the attention outputs to predict conversion.</h3>
        <p>Repeat the calculation across the four query positions and two demo heads. Here are the scalar outputs; the selected row is only one of these eight calculations.</p>
        <div className="aw-output-grid"><div className="aw-output-heading"><span>Query</span><span>Head 1 output</span><span>Head 2 output</span></div>{TOUCHES.map((t, r) => <div key={t.id}><strong>{t.shortLabel}</strong><span>{f(whole.headOutputs[0][r])}</span><span>{f(whole.headOutputs[1][r])}</span></div>)}</div>
        <div className="aw-readout-grid"><div><span className="aw-label">What the paper specifies</span><strong>A flattened readout, with unresolved details</strong><p>LiDDA specifies flattening and separate member/company context. Its prose averages heads, but its equation concatenates them. It also uses “scores” and “output” loosely, leaving the exact flattened tensor ambiguous. Flattening preserves entries in one long feature vector; it does not itself predict conversion. <PaperLink section="S2.SS4">§2.4</PaperLink></p><p><a href="#attention-readout">Follow the shapes and an explicit readout</a>.</p></div><div><span className="aw-label">This demo’s simpler classifier</span><strong>Average, add a bias, apply a sigmoid</strong><MathText block>{"\\hat p=\\sigma\\!\\left(b+\\frac{1}{2n}\\sum_{h=1}^{2}\\sum_{r=1}^{n}z_r^{(h)}\\right)"}</MathText><p>With <MathText>{"n=4"}</MathText>, pooled output {f(whole.pooled)} and bias {whole.bias.toFixed(1)}, the prediction is <strong>{percent(whole.probability)}</strong>.</p></div></div>
        <p className="aw-small"><MathText>{"\\sigma(a)=1/(1+e^{-a})"}</MathText> maps a scalar to a probability. The pooled toy makes the arithmetic visible; it does not reproduce the paper’s classifier.</p>
      </article>
      <article className="aw-stage aw-scroll-step" id="attention-step-7" aria-labelledby="attention-step-title-7">
        <p className="aw-counter">Step 7 of 8 · Read credit</p>
        <h3 id="attention-step-title-7">Reading attention as credit is a separate step.</h3>
        <p>Our predictor uses the weighted values. Our credit rule instead reads the attention weights. LiDDA describes head aggregation, zero credit for padding, and normalization within a path; its exact reduction and masking order are incompletely specified. <PaperLink section="S2.SS4">§2.4</PaperLink></p>
        <div className="aw-definition"><strong>Check the dimensions first.</strong><p>Two <MathText>{"4\\times4"}</MathText> attention matrices still give a <MathText>{"4\\times4"}</MathText> matrix when added. Normalization does not remove the query dimension. We explicitly choose to sum down each source column over valid query positions.</p></div>
        <MathText block>{"\\hat\\alpha_j=\\frac{\\alpha_j}{\\sum_k\\alpha_k}"}</MathText>
        <div className="aw-credit-list">{TOUCHES.map((t, j) => <div key={t.id}><strong>{t.shortLabel}</strong><div><span style={{ width: `${Number((100 * whole.credit[j]).toFixed(3))}%`, backgroundColor: t.color }}/></div><span>{percent(whole.credit[j])}</span></div>)}</div>
        <div className="aw-definition"><strong>Our explicit illustration convention</strong><MathText block>{"\\alpha_j=\\sum_{h=1}^{2}\\sum_{r=1}^{n}A_{rj}^{(h)}"}</MathText><p>We total incoming attention across query positions and heads, then normalize. The paper does not fully specify how the usual query-by-key matrix is reduced over query positions; this convention fills that gap for the example.</p></div>
        <p className="aw-small">A row’s softmax normalization and the final path-credit normalization are different operations. Neither is LayerNorm, whose use is not specified in the paper. <a href="#attention-readout">Inspect the incoming totals and the denominator</a>.</p>
      </article>
      <article className="aw-stage aw-scroll-step" id="attention-step-8" aria-labelledby="attention-step-title-8">
        <p className="aw-counter">Step 8 of 8 · Train</p>
        <h3 id="attention-step-title-8">How do the parameters get learned?</h3>
        <p>For each training journey, predict the outcome, compare it with the observed label, and update the model’s parameters. A standard binary prediction loss illustrates the feedback:</p>
        <MathText block>{"\\ell(y,\\hat p)=-y\\log\\hat p-(1-y)\\log(1-\\hat p)"}</MathText>
        <div className="aw-training-loop"><span>Journeys + outcomes</span><b>→</b><span>Prediction</span><b>→</b><span>Loss</span><b>→</b><span>Parameter update</span></div>
        <p>Gradients pass through the classifier, value mixing, and softmax into the projections and embeddings. The model learns which representations help predict purchases across customers. No training label directly supplies the desired attention weight for each touch.</p>
        <div className="aw-readout-grid"><div><strong>Learned across journeys</strong><p>Trainable embeddings, projection matrices, and classifier parameters. The paper’s campaign vectors come from an upstream model; whether they are frozen or fine-tuned is unspecified.</p></div><div><strong>Computed for this journey</strong><p>Queries, keys, values, attention weights, and conversion probability.</p></div></div>
        <p className="aw-small">This walkthrough runs a forward calculation with fixed parameters. It does not simulate model training.</p>
      </article>
    </div>
    <nav className="aw-step-footer" aria-label="Continue the attention lesson"><a href="#attention-steps">Back to the eight steps</a><a href="#attention-readout">Continue to the readout</a></nav>
  </section>;
}
