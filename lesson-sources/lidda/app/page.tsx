"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { GuideRoadmap, PathRepresentation, PaperReference } from "../components/explainer/FullPaperGuide";
import { TrainingExamples } from "../components/explainer/TrainingExamples";
import { AttentionWalkthrough } from "../components/explainer/AttentionWalkthrough";
import { ArchitectureCheckpoint, AttentionReadoutBridge } from "../components/explainer/AttentionReadoutBridge";
import { TOUCHES } from "../lib/attribution-model";
import { traceWholeModel } from "../lib/attention-detail";
import { MathText as Eq } from "../components/explainer/MathText";
import "../components/explainer/lesson-readability.css";
import "../components/explainer/course-presentation.css";

const PAPER = "https://arxiv.org/html/2505.09861v3";
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
function Ref() { return <a className="ref" href={`${PAPER}#S2.SS4`} target="_blank" rel="noreferrer">paper ↗</a>; }
function Bars({ values }: { values: number[] }) {
  return <div className="bars">
    {TOUCHES.map((touch, i) => <div className="bar-row" key={touch.id}>
      <span className="bar-label"><i style={{ backgroundColor: touch.color }} />{touch.shortLabel}</span>
      <div className="bar-track"><div className="bar-fill" style={{ width: `${values[i] * 100}%`, backgroundColor: touch.color }} /></div>
      <span className="bar-number">{pct(values[i])}</span>
    </div>)}
  </div>;
}
function Journey() {
  return <div className="journey">
    {TOUCHES.map((t, i) => <div key={t.id} className="touch"><span className="touch-day">{Math.abs(t.day)} {t.day === -1 ? "day" : "days"} before cutoff</span><span className="touch-dot" style={{ background: t.color }}>{i + 1}</span><strong>{t.label}</strong><span>{t.action}</span></div>)}
    <div className="conversion"><span>✓</span><strong>Purchase</strong><small>Our company</small></div>
  </div>;
}
function Matrix({ matrices, head, setHead, row, setRow }: { matrices: number[][][]; head: number; setHead: (n: number) => void; row: number; setRow: (n: number) => void }) {
  return <div className="matrix-wrap">
    <div className="matrix-toolbar"><span>Inspect a query row</span><div className="segmented"><button aria-pressed={head === 0} onClick={() => setHead(0)}>Head 1</button><button aria-pressed={head === 1} onClick={() => setHead(1)}>Head 2</button></div></div>
    <div className="matrix-axis">Keys · the touches being read</div>
    <div className="matrix-grid"><span className="query-label">Queries</span>{TOUCHES.map(t => <span className="matrix-col" key={t.id}>{t.shortLabel}</span>)}
      {matrices[head].map((r, i) => <div className={`matrix-row ${row === i ? "selected" : ""}`} key={i}><button aria-label={`Inspect ${TOUCHES[i].label} query`} aria-pressed={row === i} onClick={() => setRow(i)}>{TOUCHES[i].shortLabel} <span>→</span></button>{r.map((v, j) => <button key={j} onClick={() => setRow(i)} aria-label={`${TOUCHES[i].label} query to ${TOUCHES[j].label} key: ${pct(v)}`} style={{ background: `rgba(35, 78, 112, ${0.04 + v * 0.55})` }}>{pct(v)}</button>)}</div>)}
    </div><p className="matrix-caption"><strong>{TOUCHES[row].label}</strong> reads all four touches. This row adds to 100%. A cell is a mixing weight, not yet a touchpoint’s share of conversion credit.</p>
  </div>;
}
const chapterLabels = ["Journey", "Prediction", "Attention", "Credit"];
export default function Home() {
  const [active, setActive] = useState(0);
  const progressRef = useRef<HTMLDivElement>(null);
  const [head, setHead] = useState(0), [row, setRow] = useState(0);
  const base = useMemo(() => traceWholeModel(), []);
  const matrices = base.traces.map(rows => rows.map(trace => trace.weights));
  useEffect(() => {
    const update = () => {
      const chapters = [...document.querySelectorAll<HTMLElement>(".chapter")];
      const midpoint = window.innerHeight * (window.innerWidth <= 760 ? 0.75 : 0.52);
      let next = 0; chapters.forEach((el, i) => { if (el.getBoundingClientRect().top < midpoint) next = i; });
      setActive(next);
      const progress = window.scrollY / Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      if (progressRef.current) progressRef.current.style.transform = `scaleX(${progress})`;
    };
    update(); window.addEventListener("scroll", update, { passive: true }); window.addEventListener("resize", update);
    return () => { window.removeEventListener("scroll", update); window.removeEventListener("resize", update); };
  }, []);
  return <>
    <a className="skip" href="#training-data">Skip to the explainer</a>
    <header className="site-header"><a href="#top" className="brand"><span className="brand-mark">a</span> LiDDA, unpacked</a><nav aria-label="Main navigation"><a href="#training-data">Training data</a><a href="#architecture-check">Attention lesson</a><a href="#attention-readout">Prediction & credit</a><a href="#notes">Paper & glossary</a></nav><div ref={progressRef} className="reading-progress" style={{ width: "100%", transform: "scaleX(0)", transformOrigin: "left" }} /></header>
    <main id="top">
      <section className="hero" aria-labelledby="lesson-title">
        <h1 id="lesson-title">LiDDA</h1>
        <p className="hero-name">LinkedIn Data-Driven Attribution</p>
        <p className="hero-subtitle">An application of attention · POLS 4728</p>
      </section>
      <TrainingExamples />
      <GuideRoadmap />
      <PathRepresentation />
      <section className="scrolly" id="story"><aside className="visual-column" aria-label="Illustration for the current chapter"><div className="visual-sticky"><div className="stage-top"><span className="eyebrow">A single observed journey</span><span className="stage-counter">0{active+1} / 04</span></div><div className="chapter-tabs">{chapterLabels.map((label, i) => <a key={label} aria-current={active === i ? "step" : undefined} href={`#chapter-${i}`}>{label}</a>)}</div>
        <div className="stage-body" key={active}>
          {active === 0 && <><h3>All four ads are ours.</h3><Journey /><div className="observation-box"><span>Observed</span><p>Ad channel · action · campaign · time<br />Customer information available before the outcome<br />Purchase within a defined outcome window</p></div><p className="stage-note">Synthetic example · no customer data</p></>}
          {active === 1 && <><h3>First, learn to predict the outcome.</h3><div className="training-paths">{[true,false,true,false].map((converted,i)=><div key={i}><span>Customer {String.fromCharCode(65+i)}</span><div className="mini-path">{TOUCHES.slice(i%2,4-i%3).map(t=><i key={t.id} style={{background:t.color}}/>)}</div><b className={converted?"yes":"no"}>{converted?"Purchase":"No purchase"}</b></div>)}</div><div className="model-flow"><span>Observed paths + customer context</span><b>↓</b><strong>Response model</strong><b>↓</b><Eq>{"\\hat p(x,S)=P_\\theta(Y=1\\mid x,S)"}</Eq></div><p className="stage-note">Training needs both outcomes. A conversion-only dataset cannot teach the conversion rate.</p></>}
          {active === 2 && <><h3>Attention is a matrix of relationships.</h3><Matrix matrices={matrices} head={head} setHead={setHead} row={row} setRow={setRow}/><div className="stage-formula"><Eq>{"A^{(h)}=\\operatorname{softmax}_{\\rm row}\\!\\left(\\frac{Q^{(h)}K^{(h)\\top}}{\\sqrt{d_k}}\\right)"}</Eq></div></>}
          {active === 3 && <><h3>Turn mixing weights into credit.</h3><div className="aggregate-flow"><span>2 attention heads</span><b>→</b><span>Sum incoming weights</span><b>→</b><strong>Normalize</strong></div><Bars values={base.credit}/><div className="stage-formula"><Eq>{"\\hat\\alpha_j=\\frac{\\alpha_j}{\\sum_k\\alpha_k},\\qquad \\sum_j\\hat\\alpha_j=1"}</Eq></div><p className="stage-note">Illustrative reduction: average incoming attention over query positions and heads. This is our convention; the paper does not fully specify the query-axis reduction.</p></>}
        </div><div className="stage-bottom"><span className="live-dot"/>Illustrative mechanics · not LinkedIn estimates</div></div></aside>
        <div className="story-copy">
          <article className="chapter" id="chapter-0"><span className="chapter-number">01 / Define what we can see</span><h2>A journey inside<br />one company’s view.</h2><p>A customer sees our display ad, clicks our paid social ad, watches our video ad, and clicks our paid search ad. Then they buy from us.</p><p>Each touch is an event, not a different advertiser. The channels may be different ad sellers; the buyer of those ads is always our company.</p><p>Our question is how to divide credit for <em>our</em> purchase among <em>our</em> observed touches. Unseen exposures and prior customer intent are still possible.</p><div className="takeaway"><strong>Fix the measurement window.</strong> Use a consistent lookback and outcome horizon. In the prediction setup here, features precede the outcome window; the purchase is a label, never an input.</div></article>
          <article className="chapter" id="chapter-1"><span className="chapter-number">02 / Learn from both outcomes</span><h2>Prediction supplies<br />the training signal.</h2><p>Write the journey as <Eq>{"S=(t_1,\\ldots,t_n)"}</Eq>, available customer context as <Eq>{"x"}</Eq>, and our purchase indicator as <Eq>{"Y"}</Eq>. A response model estimates <Eq>{"\\hat p(x,S)"}</Eq>.</p><p>Fit on converting and non-converting observation windows. Keep information from after the prediction cutoff out of the features, and evaluate on later held-out data.</p><p>LiDDA uses a conversion classifier with one self-attention layer, multiple heads, and a flattened attention output. Touch representations include campaign and timing information. <Ref /></p><div className="takeaway"><strong>Our adaptation</strong> Use this advertiser’s available customer features and ad history. No cross-advertiser journeys or advertiser embedding are needed. The paper’s member-company feature refers to the customer’s employer, not another advertiser.</div></article>
          <article className="chapter" id="chapter-2"><span className="chapter-number">03 / Look inside the attention layer</span><h2>Who reads<br />information from whom?</h2><p>Within each head, touch representations <Eq>{"X"}</Eq> are projected into queries, keys, and values:</p><Eq block>{"Q=XW_Q,\\qquad K=XW_K,\\qquad V=XW_V"}</Eq><p>Conversion-prediction loss updates these projection parameters through the attention calculation. It needs conversion outcomes, not ground-truth labels for each touch’s credit. Our toy model uses fixed illustrative parameters.</p><p>A query compares with every key; a row-wise softmax turns compatibility scores into positive mixing weights.</p><p>Read a row as: “To update this touch’s representation, how much should the model read from each touch?” Each row totals one. Different heads can learn different relationships.</p><Eq block>{"z_r^{(h)}=\\sum_j A_{rj}^{(h)}v_j^{(h)}"}</Eq><p>The weights multiply the <em>values</em>. Later transformations also matter. A large attention weight alone does not reveal the sign or magnitude of the final prediction change.</p><div className="takeaway"><strong>Try the matrix</strong> Select a query row or switch heads. The column identifies the touch receiving attention; the row identifies the touch doing the reading. <a href="#attention-detail">Work through one row numerically ↓</a></div></article>
          <article className="chapter" id="chapter-3"><span className="chapter-number">04 / Make the allocation choice</span><h2>A normalized weight<br />becomes a share of credit.</h2><p>LiDDA describes head aggregation, zero credit for padding, and normalization within each path. It does not fully specify the query reduction or masking order. The resulting shares are relative touchpoint credits. <Ref /></p><p>For this illustration we average incoming weights across query rows and heads, then allocate one unit of credit to this converted path. That explicit reduction is a teaching convention, not a recovered implementation detail.</p><p>The total is 100% by construction. It does not say that the ads caused 100% of the purchase, nor that this customer had zero chance of converting without them.</p><div className="takeaway"><strong>Credit is relative to the path.</strong> Add or remove a touch and the softmax denominators change. Other touches can receive different weights even when their own recorded features stay fixed.</div></article>
        </div>
      </section>
      <ArchitectureCheckpoint />
      <AttentionWalkthrough />
      <AttentionReadoutBridge />
      <PaperReference />
    </main><footer><span>LiDDA, unpacked · an independent teaching example</span><a href="#top">Back to the beginning ↑</a></footer>
  </>;
}
