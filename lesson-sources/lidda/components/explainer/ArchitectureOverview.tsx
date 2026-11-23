import { MathText, PaperLink } from "./MathText";
import "./architecture-overview.css";

export function ArchitectureOverview() {
  return (
    <section className="ao-overview" id="architecture-check" aria-labelledby="ao-title">
      <header className="ao-heading">
        <h2 id="ao-title">LiDDA’s conversion model</h2>
        <p>
          A sequence classifier built around multi-head self-attention.
        </p>
      </header>

      <figure className="ao-figure">
        <div className="ao-flow">
          <div className="ao-branch ao-branch--sequence">
            <div className="ao-node">
              <h3>Ordered ad-touch sequence</h3>
              <p>Campaign, event, time, position</p>
            </div>
            <span className="ao-arrow" aria-hidden="true" />
            <div className="ao-node ao-node--attention">
              <h3>Multi-head self-attention</h3>
              <p><strong>1 layer</strong> · head count not reported</p>
            </div>
            <span className="ao-arrow" aria-hidden="true" />
            <div className="ao-node">
              <h3>Combine outputs and flatten</h3>
            </div>
            <span className="ao-tail" aria-hidden="true" />
          </div>

          <div className="ao-branch ao-branch--context">
            <div className="ao-node">
              <h3>Member + employer covariates</h3>
            </div>
            <span className="ao-arrow" aria-hidden="true" />
            <div className="ao-node">
              <h3>Interaction layer</h3>
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
          Covariates join before classification. The paper does not describe
          adding them to each token’s residual stream. <PaperLink section="S2.F1">Figure 1</PaperLink>
        </figcaption>
      </figure>
      <p className="ao-training-signal">Conversion labels train the model, including its attention weights; LiDDA then reuses the learned attention matrices to allocate touchpoint credit.</p>

      <details className="ao-representation">
        <summary>The representation in the paper</summary>
        <MathText block>
          {"\\begin{aligned}&\\operatorname{concat}(E_{\\mathrm{MCID}}, E_T)\\\\&\\quad + E_D + E_{\\mathrm{DOW}} + \\operatorname{tAPE}\\end{aligned}"}
        </MathText>
        <p className="ao-key">
          <b>MCID:</b> campaign; <b>T:</b> event type; <b>D:</b> days relative to conversion;{" "}
          <b>DOW:</b> weekday; <b>tAPE:</b> position.
        </p>
        <p>
          Campaign and event embeddings are concatenated; time and position terms
          are added. Residual connections and normalization are unspecified.{" "}
          <PaperLink section="S2.SS3">Input representation</PaperLink>
        </p>
      </details>
    </section>
  );
}
