import { MathText, PaperLink } from "./MathText";
import "./architecture-overview.css";

export function ArchitectureOverview() {
  return (
    <section className="ao-overview" id="architecture-check" aria-labelledby="ao-title">
      <header className="ao-heading">
        <h2 id="ao-title">LiDDA’s conversion model</h2>
        <p>
          Touch-level inputs enter self-attention. Member and employer covariates
          enter through a separate branch.
        </p>
      </header>

      <figure className="ao-figure">
        <div className="ao-flow">
          <div className="ao-branch ao-branch--sequence">
            <div className="ao-node">
              <h3>Per-touch representations</h3>
              <p>Campaign, event, time, position</p>
            </div>
            <span className="ao-arrow" aria-hidden="true" />
            <div className="ao-node ao-node--attention">
              <h3>One multi-head self-attention layer</h3>
              <p>Paper head count not reported</p>
            </div>
            <span className="ao-arrow" aria-hidden="true" />
            <div className="ao-node">
              <h3>Combine and flatten</h3>
              <p>Illustration: average heads</p>
              <p className="ao-shape">
                <MathText>{"5 \\times 2"}</MathText> outputs → 10 features
              </p>
            </div>
            <span className="ao-tail" aria-hidden="true" />
          </div>

          <div className="ao-branch ao-branch--context">
            <div className="ao-node">
              <h3>Member + employer embeddings</h3>
            </div>
            <span className="ao-arrow" aria-hidden="true" />
            <div className="ao-node">
              <h3>Interaction layer</h3>
              <p>Customer-context features</p>
            </div>
            <span className="ao-tail" aria-hidden="true" />
          </div>

          <span className="ao-merge" aria-hidden="true">
            <span className="ao-tip" />
          </span>

          <div className="ao-node ao-classifier">
            <h3>Dense conversion classifier</h3>
            <p>
              Conversion probability <MathText>{"\\widehat p(Y=1)"}</MathText>
            </p>
          </div>
        </div>
        <figcaption>
          Worked illustration: <strong>2 heads, 5 padded slots, 2 value coordinates per head.</strong>{" "}
          Averaging and flattening the value-mixed outputs is our illustrative readout.
        </figcaption>
      </figure>

      <aside className="ao-representation" aria-labelledby="ao-representation-title">
        <h3 id="ao-representation-title">The representation in the paper</h3>
        <MathText block>
          {"\\begin{aligned}&\\operatorname{concat}(E_{\\mathrm{MCID}}, E_T)\\\\&\\quad + E_D + E_{\\mathrm{DOW}} + \\operatorname{tAPE}\\end{aligned}"}
        </MathText>
        <p className="ao-key">
          <b>MCID:</b> campaign; <b>T:</b> event type; <b>D:</b> days relative to conversion;{" "}
          <b>DOW:</b> weekday; <b>tAPE:</b> position.
        </p>
        <p>
          <PaperLink section="S2.F1">Figure 1</PaperLink> places member/employer
          covariates on the separate branch, rather than adding them to each token’s residual stream. Residual/LayerNorm wiring is unspecified.{" "}
          <PaperLink section="S2.SS4">Model §2.4</PaperLink>
        </p>
      </aside>
    </section>
  );
}
