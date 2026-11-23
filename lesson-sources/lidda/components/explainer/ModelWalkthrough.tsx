"use client";
import { useEffect, useRef, useState } from "react";
import { traceLessonShapes } from "../../lib/lesson-shapes";
import { TOUCHES } from "../../lib/attribution-model";
import { MathText } from "./MathText";
import "./model-walkthrough.css";
const lesson = traceLessonShapes();
const names = ["Display", "Instagram", "YouTube", "Search", "PAD"];
const colors = [...TOUCHES.map(touch => touch.color), "#606060"];
const titles = ["Attention matrices", "Touchpoint credit"];
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

function CreditReadout() {
  return <div className="mw-credit">
    <MathText block>{"\\alpha_j=\\sum_{h=1}^{2}\\sum_{i=1}^{4}A^{(h)}_{ij}"}</MathText>
    <MathText block>{"\\widehat\\alpha_j=\\frac{\\alpha_j}{\\sum_{k=1}^{4}\\alpha_k}=\\frac{\\alpha_j}{8}"}</MathText>
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
    let frame = 0;
    const update = () => {
      frame = 0;
      const steps = [...(sectionRef.current?.querySelectorAll<HTMLElement>(".mw-step") ?? [])];
      const stage = stageRef.current;
      if (!stage || steps.length !== 2) return;
      const headerHeight = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header"));
      const trigger = window.innerWidth <= 900
        ? Math.min(window.innerHeight - 72, headerHeight + stage.offsetHeight + 52)
        : window.innerHeight * 0.54;
      let next = 0;
      steps.forEach((step, index) => { if (step.getBoundingClientRect().top <= trigger) next = index; });
      if (activeRef.current !== next) { activeRef.current = next; setActive(next); }
    };
    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(update); };
    update();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, []);
  return <section className="mw-section" id="attention-readout" ref={sectionRef} aria-labelledby="mw-title">
    <header className="mw-heading">
      <h2 id="mw-title">From attention matrices to credit</h2>
      <p>Customer A · two illustrative heads, four touches, and one PAD slot.</p>
    </header>
    <div className="mw-scrolly">
      <div className="mw-stage" ref={stageRef} data-stage={active}>
        <div className="mw-stage-heading"><span>{active + 1} / 2</span><h3>{titles[active]}</h3></div>
        {active === 0 ? <AttentionMatrices /> : <CreditReadout />}
        {active === 0 && <ol className="mw-touch-key" aria-label="Customer A's input positions">
          {names.map((name, index) => <li key={name}><span style={{ color: colors[index] }}>{index + 1}</span> {name}</li>)}
        </ol>}
        <p className="mw-stage-note">{active === 0
          ? "Rows = queries; columns = source touches. Valid rows sum to 1; PAD is excluded."
          : "Sum incoming attention over heads and valid query positions, then normalize over real touches. This explicit rule is illustrative."}</p>
      </div>
      <div className="mw-steps">
        <article className="mw-step" id="attention-detail">
          <span className="mw-step-number">01 · Attention</span><h3>One matrix per head.</h3>
          <p>Training on conversion labels produces the attention matrices. Here, two illustrative heads each yield a <MathText>{"5\\times5"}</MathText> matrix for the same journey.</p>
        </article>
        <article className="mw-step" id="touch-credit">
          <span className="mw-step-number">02 · Attribution</span><h3>Aggregate by touch and normalize.</h3>
          <p>Sum each source column over both heads and the four real query rows. Divide by <MathText>{"2\\times4=8"}</MathText>, the total weight, to obtain relative shares. Padding gets zero credit.</p>
        </article>
      </div>
    </div>
  </section>;
}
