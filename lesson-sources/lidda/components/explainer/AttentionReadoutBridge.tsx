"use client";

import { useMemo, useState } from "react";
import { TOUCHES } from "../../lib/attribution-model";
import { traceReadoutBridge, type BridgeReadout } from "../../lib/attention-bridge";
import { MathText as M, PaperLink } from "./MathText";
import "./attention-readout-bridge.css";

const pct = (n: number) => (100 * n).toFixed(1) + "%";
const f = (n: number) => n.toFixed(3);

export function ArchitectureCheckpoint() {
  return <section className="paper-section architecture-checkpoint" id="architecture-check">
    <span className="eyebrow">Pause here for a classroom question</span>
    <h2>Is the model<br />fully specified yet?</h2>
    <p className="section-lede">You have two attention heads, four touchpoints, and one conversion label. Each head carries three value coordinates per touch. The instructions say: “Use attention to predict conversion and assign credit.” What is still missing?</p>
    <div className="ab-question">
      <p><strong>Before reading on:</strong> sketch the dimensions. How do you get one prediction and four credit shares? What happens to the query dimension when you sum the heads?</p>
      <div className="ab-checklist">
        <h3>What a complete implementation must specify</h3>
        <div>
          <ol>
            <li><strong>Choose what the classifier reads.</strong> Each attention matrix is <M>{"4\\times4"}</M>. Each value-mixed output is <M>{"4\\times3"}</M>. These are different objects.</li>
            <li><strong>Combine heads and choose a readout.</strong> Averaging the mixed outputs gives <M>{"4\\times3"}</M>; concatenating them gives <M>{"4\\times6"}</M>. Flattening those produces 12 or 24 features. Pooling or a special classification token would be other explicit designs.</li>
            <li><strong>Define the prediction head.</strong> Map the chosen features and customer context to one logit, then to a conversion probability. Specify the labels, observation cutoff, and training loss.</li>
            <li><strong>Define a separate credit rule.</strong> Summing two attention matrices still leaves a <M>{"4\\times4"}</M> matrix. You must choose how to reduce its query dimension to four source-touch scores, then normalize them.</li>
            <li><strong>Handle variable lengths.</strong> Set a fixed padded/truncated length for a flattened dense readout. Specify masking of padded keys, query outputs, and credit entries. Normalizing alone does not remove padding.</li>
          </ol>
          <p>These choices define how the model turns an input journey into a prediction and a vector of touchpoint credits.</p>
        </div>
      </div>
    </div>
    <p>The next walkthrough supplies the attention arithmetic. <a href="#attention-readout">The readout bridge</a> then connects those tensors to a prediction and a credit vector.</p>
  </section>;
}

export function AttentionReadoutBridge() {
  const [negativeVideo, setNegativeVideo] = useState(false);
  const [readout, setReadout] = useState<BridgeReadout>("mean");
  const [selectedPosition, setSelectedPosition] = useState(0);
  const model = useMemo(() => traceReadoutBridge(negativeVideo, readout), [negativeVideo, readout]);
  const reference = useMemo(() => traceReadoutBridge(false, readout), [readout]);
  const negativeReference = useMemo(() => traceReadoutBridge(true, readout), [readout]);

  return <section className="paper-section attention-readout-bridge" id="attention-readout">
    <span className="eyebrow">From an attention layer to a complete model</span>
    <h2>One set of weights.<br />Two different readouts.</h2>
    <p className="section-lede">Attention produces a matrix for each head. A conversion classifier needs one number per journey. An attribution report needs one share per touch. Those requirements do not specify the same calculation.</p>

    <div className="ab-table-scroll" role="region" aria-label="Attention tensor dimensions" tabIndex={0}>
      <table className="ab-table">
        <caption>Track the objects · one head, one journey, no batch dimension shown</caption>
        <thead><tr><th scope="col">Object</th><th scope="col">Shape</th><th scope="col">Meaning</th></tr></thead>
        <tbody>
          <tr><th scope="row">Scores <M>{"R=QK^\\top/\\sqrt{d_k}"}</M></th><td><M>{"n\\times n"}</M></td><td>Compatibility of each query with each key. Real numbers, possibly negative.</td></tr>
          <tr><th scope="row">Weights <M>{"A=\\operatorname{softmax}_{\\rm row}(R)"}</M></th><td><M>{"n\\times n"}</M></td><td>How each query mixes information from source touches. Each valid row sums to one.</td></tr>
          <tr><th scope="row">Values <M>{"V"}</M></th><td><M>{"n\\times d_v"}</M></td><td>The information carried by each source touch.</td></tr>
          <tr><th scope="row">Mixed outputs <M>{"Z=AV"}</M></th><td><M>{"n\\times d_v"}</M></td><td>A new vector at each query position. Its coordinates need not be positive or sum to one.</td></tr>
        </tbody>
      </table>
    </div>
    <p className="ab-note"><M>{"n"}</M> is the number of retained touches and <M>{"d_v"}</M> is the value width per head. The worked four-touch model has <M>{"H=2"}</M> heads and <M>{"d_v=1"}</M>. With padding, use a fixed maximum length and mask the invalid entries. <a href="https://arxiv.org/html/1706.03762v7#S3.SS2" target="_blank" rel="noreferrer">Standard attention definition</a>.</p>

    <div className="ab-branches">
      <article>
        <span className="ab-kicker">Prediction branch · a specified teaching design</span>
        <h3>Mix values, then read the resulting features.</h3>
        <M block>{"\\bar Z=\\frac{1}{H}\\sum_h A^{(h)}V^{(h)}"}</M>
        <M block>{"\\hat p=\\sigma\\!\\left(g_\\theta(\\operatorname{vec}(\\bar Z),x)\\right)"}</M>
        <p>Here we average heads, retain all position slots by flattening, and let a classifier <M>{"g_\\theta"}</M> combine those features with customer context <M>{"x"}</M>. A dense classifier can put different coefficients on different slots.</p>
      </article>
      <article>
        <span className="ab-kicker">Credit branch · our explicit column-sum rule</span>
        <h3>Reduce the weights to one score per source touch.</h3>
        <M block>{"c_j=\\sum_h\\sum_{r\\in\\mathcal V} A_{rj}^{(h)}"}</M>
        <M block>{"\\hat\\alpha_j=\\frac{c_j}{\\sum_{k\\in\\mathcal V}c_k},\\quad j\\in\\mathcal V"}</M>
        <p><M>{"\\mathcal V"}</M> contains the valid, unpadded positions. Sum down each source column over valid query rows and heads. Then divide by the total incoming mass. Padding receives zero credit.</p>
      </article>
    </div>
    <div className="ab-source-note">
      <strong>What does LiDDA itself specify?</strong>
      <p>The paper specifies flattening for classification, but uses “scores” and “output” inconsistently. Its equation concatenates heads while its prose averages them. It does not fully specify the query reduction for attribution. The two branches above are an explicit teaching construction; they are not a verified reconstruction of the deployed model. <PaperLink section="S2.SS4">Check §2.4</PaperLink></p>
      <p>Flattening the mixed representations <M>{"AV"}</M> and flattening the weights <M>{"A"}</M> would produce different classifier inputs. Our example explicitly chooses the mixed representations, making every numerical step inspectable.</p>
    </div>

    <div className="ab-flatten">
      <h3>Flattening preserves slots. Pooling combines them.</h3>
      <p>Imagine four positions with two output coordinates each. Select a position to follow its coordinates into the flattened vector. These are symbolic slots, separate from the scalar toy below.</p>
      <div className="ab-flatten-visual">
        <div className="ab-position-matrix" role="group" aria-label="Select a position to trace through flattening">
          {[0, 1, 2, 3].map(r => <button key={r} type="button" aria-pressed={selectedPosition === r} onClick={() => setSelectedPosition(r)}>
            <span>Position {r + 1}</span><M>{"z_{" + (r + 1) + "1}"}</M><M>{"z_{" + (r + 1) + "2}"}</M>
          </button>)}
        </div>
        <div className="ab-flat-result">
          <strong>Flatten: eight retained coordinates</strong>
          <ol className="ab-flat-slots" aria-label="Flattened vector in row order">
            {[0, 1, 2, 3].flatMap(r => [1, 2].map(c => <li key={r + "-" + c} className={selectedPosition === r ? "is-highlighted" : ""}><M>{"z_{" + (r + 1) + c + "}"}</M></li>))}
          </ol>
          <p>Coordinates within each position come first; then the next position. Nothing is averaged or normalized.</p>
          <strong>Mean pooling: two averaged coordinates</strong>
          <M block>{"\\left[\\frac14\\sum_{r=1}^4z_{r1},\\;\\frac14\\sum_{r=1}^4z_{r2}\\right]"}</M>
          <p>Pooling removes the separate output slots. Earlier position encodings can still affect the representations; pooling does not necessarily make the entire model insensitive to order.</p>
        </div>
      </div>
      <p><strong>Flattening alone is not a classifier.</strong> A learned head must still map the feature vector to a logit. With a fixed maximum of <M>{"N"}</M> positions, this example’s averaged-head readout has <M>{"Nd_v"}</M> features before customer context is added.</p>
    </div>

    <div className="ab-live" id="attention-values-demo">
      <span className="ab-kicker">Controlled calculation · same attention as the walkthrough</span>
      <h3>Hold attention fixed. Change what it carries.</h3>
      <p>First predict what will happen when the video value changes from +2.3 to −2.3. Will the conversion prediction change? Will any of the attention credit shares change?</p>
      <div className="ab-controls">
        <label>Prediction readout<select aria-label="Prediction readout" value={readout} onChange={e => setReadout(e.target.value as BridgeReadout)}>
          <option value="mean">Mean pooling · original toy</option>
          <option value="position">Flatten · position-specific coefficients</option>
        </select></label>
        <button type="button" aria-pressed={negativeVideo} onClick={() => setNegativeVideo(v => !v)}>{negativeVideo ? "Restore the positive video value" : "Make the video value negative"}</button>
      </div>
      <div className="ab-value-strip" aria-label="Scalar values for the four source touches">
        {TOUCHES.map((t, j) => <div key={t.id}><span>{t.shortLabel} value</span><strong>{model.values[j] > 0 ? "+" : ""}{model.values[j].toFixed(1)}</strong></div>)}
      </div>
      <div className="ab-results" aria-live="polite" aria-atomic="true">
        <div><span>Prediction with positive video value</span><strong>{pct(reference.probability)}</strong></div>
        <div><span>Current prediction</span><strong>{pct(model.probability)}</strong></div>
        <div><span>Video attention credit · unchanged</span><strong>{pct(model.credit[2])}</strong></div>
      </div>
      <div className="ab-table-scroll" role="region" aria-label="Prediction calculation by output position" tabIndex={0}>
        <table className="ab-table">
          <caption>Average the heads, then apply the selected classifier</caption>
          <thead><tr><th scope="col">Output position</th><th scope="col">Averaged output</th><th scope="col">Readout coefficient</th><th scope="col">Term in the logit</th></tr></thead>
          <tbody>{TOUCHES.map((t, r) => <tr key={t.id}><th scope="row">{r + 1} · {t.shortLabel}</th><td>{f(model.features[r])}</td><td>{model.coefficients[r].toFixed(2)}</td><td>{f(model.weightedFeatures[r])}</td></tr>)}</tbody>
        </table>
      </div>
      <M block>{"\\hat p=\\sigma\\!\\left(-2.4+\\sum_{r=1}^4\\beta_r\\bar z_r\\right)"}</M>
      <p>{readout === "mean" ? "All four coefficients are 1/4, so this equals mean pooling and reproduces the original toy. " : "The flattened vector retains all four positions. Coefficients 0.10, 0.20, 0.30, and 0.40 let the classifier weight their outputs differently. "}The coefficients and bias are chosen for illustration. In a fitted classifier, they would be learned.</p>
      <p className="ab-note">The result with this readout: changing the video value from +2.3 to −2.3 lowers the prediction from {pct(reference.probability)} to {pct(negativeReference.probability)}, while its attention credit stays at {pct(reference.credit[2])}. The weights are unchanged; the information they mix has changed.</p>
      <p className="ab-note">Only the internal video value and your selected readout change here. Q, K, and the attention matrices remain fixed. This controlled calculation isolates the role of the values from the role of the attention weights.</p>
    </div>

    <div className="ab-credit">
      <h3>Make the credit normalization auditable.</h3>
      <p>Each of the two heads has four rows that sum to one. Together they contain <M>{"Hn=2\\times4=8"}</M> units of attention mass. Our credit rule collects that mass by source column, then divides by eight.</p>
      <div className="ab-table-scroll" role="region" aria-label="Incoming attention and normalized credit" tabIndex={0}>
        <table className="ab-table">
          <caption>Same four source touches · unchanged by the value edit</caption>
          <thead><tr><th scope="col">Source touch</th><th scope="col">Incoming column total</th><th scope="col">Divide by total mass</th><th scope="col">Credit share</th></tr></thead>
          <tbody>{TOUCHES.map((t, j) => <tr key={t.id}><th scope="row">{t.shortLabel}</th><td>{f(model.incoming[j])}</td><td>{f(model.incoming[j])} / 8</td><td>{pct(model.credit[j])}</td></tr>)}</tbody>
          <tfoot><tr><th scope="row">Total</th><td>8.000</td><td>8 / 8</td><td>100%</td></tr></tfoot>
        </table>
      </div>
      <div className="ab-question"><strong>What if you sum along each row instead?</strong><p>Each row already sums to one. Summing across heads gives two at every query; normalizing gives 25% each in this four-touch example. That operation loses the variation in which source touches receive attention. The axis is a substantive choice.</p></div>
      <div className="ab-padding"><h4>Padding: an explicit convention for this construction</h4><div><p>Mask padded keys before row-softmax, and exclude padded query rows from credit aggregation. Set padded mixed-output rows to zero before a flattened readout. Set padded credits to zero and normalize over valid touches only. Do not apply a softmax to an all-masked empty path; define a separate context-only prediction and no touch credits. The paper does not fully specify this operation order.</p><p>For a journey collected before a fixed prediction cutoff, all observed touches may attend to one another. A causal triangular mask is a separate task choice; it is required only when a position’s prediction must not access later positions.</p></div></div>
    </div>

    <div className="ab-normalizations">
      <h3>Three normalizations you should not confuse.</h3>
      <dl>
        <div><dt>Row-softmax</dt><dd>Turns compatibility scores into mixing weights. One query distributes weight over valid source keys.</dd></div>
        <div><dt>Path-credit normalization</dt><dd>Turns the chosen nonnegative source scores into shares of one allocated conversion. It does not identify how many conversions the advertising caused.</dd></div>
        <div><dt>LayerNorm</dt><dd>Rescales feature coordinates using their mean and variance. It neither allocates credit nor replaces a readout. Its use is not specified in LiDDA’s architecture description.</dd></div>
      </dl>
      <p><strong>Flattening is none of these.</strong> It reshapes an array while preserving its entries. A standard attention layer, a complete conversion classifier, and a justified attribution rule are three different levels of specification.</p>
    </div>
  </section>;
}
