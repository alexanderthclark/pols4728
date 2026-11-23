"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { traceLessonShapes } from "../../lib/lesson-shapes";
import { TOUCHES } from "../../lib/attribution-model";
import { MathText } from "./MathText";
import "./model-walkthrough.css";

const lesson = traceLessonShapes();
const names = ["Display", "Instagram", "YouTube", "Search", "PAD"];
const colors = [...TOUCHES.map(touch => touch.color), "#606060"];
const titles = ["Two attention matrices", "Value-mixed outputs", "The same entries, flattened", "One conversion probability", "A separate credit calculation"];
const clamp = (value: number) => Math.max(0, Math.min(1, value));

function AttentionMatrices() {
  return <div className="mw-matrices">
    {lesson.heads.map((matrix, head) => <table className="mw-matrix" key={head}>
      <caption>Head {head + 1} · <MathText>{`A^{(${head + 1})}`}</MathText></caption>
      <thead><tr><th scope="col" aria-label="Query position / source position">i / j</th>{names.map((name, index) => <th scope="col" key={name} title={name}>{index + 1}</th>)}</tr></thead>
      <tbody>{matrix.map((row, query) => <tr key={query}>
        <th scope="row" title={names[query]}>{query + 1}</th>
        {row.map((weight, source) => <td key={source} className={query === 4 || source === 4 ? "mw-masked" : ""} style={{ backgroundColor: query === 4 || source === 4 ? undefined : `rgba(35,78,112,${weight * .65})` }}>
          <span aria-label={weight.toFixed(2)}>{weight === 0 ? "0" : weight.toFixed(2).slice(1)}</span>
        </td>)}
      </tr>)}</tbody>
    </table>)}
  </div>;
}

function OutputCells({ stage }: { stage: number }) {
  return <>
    <div className="mw-output-equation">
      {stage === 1 && <MathText>{"\\bar Z=\\tfrac12\\left(A^{(1)}V^{(1)}+A^{(2)}V^{(2)}\\right)"}</MathText>}
      {stage === 2 && <MathText>{"\\bar Z\\in\\mathbb R^{5\\times2}\\quad\\longrightarrow\\quad\\operatorname{vec}(\\bar Z)\\in\\mathbb R^{10}"}</MathText>}
      {stage === 3 && <MathText>{"\\widehat p=\\sigma\\!\\left(g_\\theta(\\operatorname{vec}(\\bar Z),c)\\right)"}</MathText>}
    </div>
    <div className={`mw-output-canvas${stage === 3 ? " mw-with-classifier" : ""}`} aria-label={stage === 1 ? "Five output rows, two coordinates each. The fifth row is zero padding." : "Ten flattened features in position order. Features nine and ten are zero padding."}>
      <div className="mw-matrix-label">Position</div>
      {names.map((name, index) => <span className="mw-position-label" key={name} style={{ "--row-y": `${48 + index * 46}px` } as CSSProperties}>{index + 1}{index === 4 ? " · PAD" : ""}</span>)}
      <div className="mw-vector-label">10-entry vector</div>
      {lesson.flatSlots.map((slot, index) => {
        const startX = slot.coordinate === 1 ? 34 : 57;
        const startY = 36 + (slot.position - 1) * 46;
        const endX = index * 10;
        const endY = 44;
        const style = {
          "--start-x": `${startX}%`, "--delta-x": `${endX - startX}%`,
          "--start-y": `${startY}px`, "--delta-y": `${endY - startY}px`,
          "--touch-color": colors[slot.position - 1],
        } as CSSProperties;
        return <div key={index} className={`mw-output-cell${slot.padded ? " mw-pad-cell" : ""}`} style={style} data-position={slot.position} data-coordinate={slot.coordinate}>
          <MathText>{slot.padded ? "0" : `\\bar z_{${slot.position}${slot.coordinate}}`}</MathText>
          <small className="mw-flat-index">{index + 1}</small>
        </div>;
      })}
      {stage === 3 && <div className="mw-classifier-flow">
        <span className="mw-readout-arrow" aria-hidden="true">↓</span>
        <span className="mw-context-input">Context <MathText>{"c"}</MathText> →</span>
        <div className="mw-dense-box">Dense classifier</div>
        <span className="mw-probability">↓ <MathText>{"\\widehat p(Y=1)"}</MathText></span>
      </div>}
    </div>
  </>;
}

function CreditReadout() {
  return <div className="mw-credit">
    <MathText block>{"\\alpha_j=\\frac{\\sum_{h=1}^{2}\\sum_{i=1}^{4}A^{(h)}_{ij}}{8}"}</MathText>
    <div className="mw-credit-heading"><span>Source touch</span><span>Column total</span><span>Credit</span></div>
    {names.map((name, index) => <div className="mw-credit-row" key={name}>
      <span>{index + 1} · {name}</span>
      <span>{lesson.incoming[index].toFixed(2)}</span>
      <span className="mw-credit-share"><span style={{ width: `${lesson.credit[index] * 100}%`, borderColor: colors[index] }} /><b>{(lesson.credit[index] * 100).toFixed(1)}%</b></span>
    </div>)}
    <p className="mw-credit-total">8 valid query rows across 2 heads → 100% of credit</p>
  </div>;
}

export function ModelWalkthrough() {
  const [active, setActive] = useState(0);
  const activeRef = useRef(0);
  const sectionRef = useRef<HTMLElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    let frame = 0;
    const update = () => {
      frame = 0;
      const steps = [...(sectionRef.current?.querySelectorAll<HTMLElement>(".mw-step") ?? [])];
      const stage = stageRef.current;
      if (!stage || steps.length !== 5) return;
      const header = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header"));
      const narrow = window.innerWidth <= 900;
      const trigger = narrow ? Math.min(window.innerHeight - 72, header + stage.offsetHeight + 52) : window.innerHeight * .54;
      let next = 0;
      steps.forEach((step, index) => { if (step.getBoundingClientRect().top <= trigger) next = index; });
      if (activeRef.current !== next) { activeRef.current = next; setActive(next); }
      const flattenRect = steps[2].getBoundingClientRect();
      const distance = Math.min(flattenRect.height * .65, window.innerHeight * .5);
      const fraction = next < 2 ? 0 : next > 2 || reduced.matches ? 1 : clamp((trigger - flattenRect.top) / distance);
      stage.style.setProperty("--flat", fraction.toFixed(4));
      // Give every entry its own column before lifting the rows together.
      // This preserves identity without crossing or obscuring other cells.
      stage.style.setProperty("--flat-x", clamp(fraction * 2).toFixed(4));
      stage.style.setProperty("--flat-y", clamp((fraction - .5) * 2).toFixed(4));
    };
    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(update); };
    update();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    reduced.addEventListener("change", schedule);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      reduced.removeEventListener("change", schedule);
    };
  }, []);
  return <section className="mw-section" id="attention-readout" ref={sectionRef} aria-labelledby="mw-title">
    <header className="mw-heading"><h2 id="mw-title">Attention matrices → flattening</h2><p>Customer A · illustrative model: 2 heads, 5 slots, 2 value coordinates per head.</p></header>
    <div className="mw-scrolly">
      <div className="mw-stage" ref={stageRef} data-stage={active}>
        <div className="mw-stage-heading"><span>{active + 1} / 5</span><h3>{titles[active]}</h3></div>
        {active === 0 && <AttentionMatrices />}
        {active >= 1 && active <= 3 && <OutputCells stage={active} />}
        {active === 4 && <CreditReadout />}
        {active < 4 && <ol className="mw-touch-key" aria-label="Customer A’s input positions">{names.map((name, index) => <li key={name}><span style={{ color: colors[index] }}>{index + 1}</span> {name}</li>)}</ol>}
        <p className="mw-stage-note">
          {active === 0 && "Rows: queries. Columns: source touches. Real rows sum to 1 before rounding; PAD is zeroed."}
          {active === 1 && "One row per position; two value coordinates per row."}
          {active === 2 && "Same ten entries, in position order. The final two are padding."}
          {active === 3 && "Context comes from the member/employer interaction branch."}
          {active === 4 && "Relative attention shares, separate from the conversion probability."}
        </p>
      </div>
      <div className="mw-steps">
        <article className="mw-step" id="attention-detail">
          <span className="mw-step-number">01 · Attention</span><h3>Keep one matrix per head.</h3>
          <p>For this sequence, the model gives us two <MathText>{"5\\times5"}</MathText> attention matrices: <MathText>{"A^{(1)}"}</MathText> and <MathText>{"A^{(2)}"}</MathText>.</p>
        </article>
        <article className="mw-step" id="value-outputs">
          <span className="mw-step-number">02 · Outputs</span><h3>Apply the weights to the values.</h3>
          <p>Each <MathText>{"Z^{(h)}=A^{(h)}V^{(h)}"}</MathText> is <MathText>{"5\\times2"}</MathText>. Our illustration averages the two heads’ outputs into one <MathText>{"5\\times2"}</MathText> array, <MathText>{"\\bar Z"}</MathText>.</p>
        </article>
        <article className="mw-step" id="flattening">
          <span className="mw-step-number">03 · Flatten</span><h3>Lay the output rows end to end.</h3>
          <p>Flatten the value-mixed outputs <MathText>{"\\bar Z"}</MathText> into ten features. This only rearranges entries; the attention matrices <MathText>{"A^{(h)}"}</MathText> are not the tensor being flattened.</p>
        </article>
        <article className="mw-step" id="conversion-readout">
          <span className="mw-step-number">04 · Predict</span><h3>Combine sequence and context.</h3>
          <p>The classifier combines the flattened sequence with the covariates to predict conversion. The observed conversion labels provide its training signal.</p>
        </article>
        <article className="mw-step" id="touch-credit">
          <span className="mw-step-number">05 · Credit</span><h3>Return to the attention weights.</h3>
          <p>Our illustrative credit rule sums each source column across heads and real queries, then divides by <MathText>{"2\\times4=8"}</MathText> to get a share.</p>
        </article>
      </div>
    </div>
  </section>;
}
