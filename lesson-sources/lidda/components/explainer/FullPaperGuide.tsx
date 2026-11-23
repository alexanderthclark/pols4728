import { MathText as M, PaperLink, PAPER_URL } from "./MathText";
import "./full-paper-guide.css";

const stops = [
  ["01", "Build the input", "Events, embeddings, order, and time", "paths"],
  ["02", "Follow the model", "A journey, a prediction, and a credit allocation", "story"],
  ["03", "Check the architecture", "Identify the tensors and their dimensions", "architecture-check"],
  ["04", "Calculate attention", "Queries, keys, softmax, and weighted values", "attention-detail"],
  ["05", "Trace the readout", "Flattening, prediction, and incoming attention", "attention-readout"],
  ["06", "Return to the source", "Model details and a compact glossary", "notes"],
] as const;

export function GuideRoadmap() {
  return <section className="paper-section guide-roadmap" id="roadmap">
    <div className="guide-heading"><div><span className="eyebrow">The learning path</span><h2>One attention layer.<br />Two distinct uses.</h2></div><p className="section-lede">LiDDA learns from customer journeys and conversion outcomes. We will follow the attention calculation, see how it feeds a prediction, and then read the weights as relative touchpoint credit.</p></div>
    <div className="guide-distinction"><h3>Keep the prediction and the allocation separate.</h3><div>
      <article><span>Conversion prediction</span><M>{"\\hat p(x,S)"}</M><p>How likely is the customer to convert, given context <M>{"x"}</M> and journey <M>{"S"}</M>?</p><small>The attention weights mix values. A classifier reads the resulting representation and produces one probability.</small></article>
      <article><span>Touchpoint credit</span><M>{"\\hat\\alpha_j"}</M><p>What fraction of one converted path’s credit is assigned to touch <M>{"j"}</M>?</p><small>A separate calculation aggregates attention weights and normalizes them across the valid touches.</small></article>
    </div></div>
    <nav className="guide-map" aria-label="LiDDA learning path">{stops.map(([n,title,description,id])=><a href={`#${id}`} key={id}><span>{n}</span><div><strong>{title}</strong><small>{description}</small></div><b aria-hidden="true">↗</b></a>)}</nav>
    <p className="guide-source-note">Scroll through the full explanation, then use the controls to inspect another query or change the prediction readout. The four-touch example uses fixed teaching parameters. Links to the paper identify the published model details; implementation choices in the example are stated explicitly.</p>
  </section>;
}

export function PathRepresentation() {
  return <section className="paper-section path-representation" id="paths">
    <span className="eyebrow">01 / Prepare the model’s input · paper §§2.1–2.3</span>
    <h2>A touch is an event.<br />A journey is an ordered list.</h2>
    <p className="section-lede">Start with one company’s ad events and one purchase outcome. Channel labels tell us where the ad ran. They do not denote different companies selling different products.</p>
    <div className="path-record"><div><span className="record-tag">One event · illustrative</span><dl><div><dt>Customer</dt><dd>Customer 42</dd></div><div><dt>Channel / action</dt><dd>Paid social / click</dd></div><div><dt>Campaign</dt><dd>Our product demonstration</dd></div><div><dt>Time before endpoint</dt><dd>14 days</dd></div></dl></div><div><span className="record-tag">One training example</span><M block>{"(x,S,Y),\\qquad S=(t_1,\\ldots,t_n)"}</M><p><M>{"x"}</M> describes the customer; <M>{"S"}</M> contains their retained touches; <M>{"Y=1"}</M> means the company’s conversion occurred. Include <M>{"Y=0"}</M> examples too.</p></div></div>
    <div className="guide-prose-grid"><div><h3>Construct comparable paths.</h3><p>Define the conversion, lookback period, and outcome window before fitting. Put the events in time order. The paper retains the most recent <M>{"N"}</M> events; shorter paths are padded and the padding is masked. These choices change which information the model can use.</p><p>Separate a customer’s background information from their marketing sequence. The paper uses a member embedding <M>{"E_M"}</M> and an embedding of that member’s employer <M>{"E_C"}</M>. In our setting, use only customer information the advertiser actually has. <PaperLink section="S2.SS1">System overview</PaperLink></p></div><div><h3>Convert events into vectors.</h3><p>An <strong>embedding</strong> is a numerical representation whose coordinates can be learned. Channel/action identifies the event type. Campaign information helps distinguish two ads on the same channel. Time tells the model whether two interactions were close together or weeks apart.</p><p>The paper’s campaign vector comes from an upstream language model using campaign descriptions. This is a way to supply campaign context, not the attention mechanism itself. <PaperLink section="S2.SS2">Path creation</PaperLink></p></div></div>
    <div className="encoding-recipe"><span className="record-tag">The representation in the paper</span><M block>{"\\operatorname{concat}(E_{MCID},E_T)+E_D+E_{DOW}+\\operatorname{tAPE}"}</M><dl><div><dt><M>{"E_{MCID}"}</M> + <M>{"E_T"}</M></dt><dd>Concatenate campaign and event-type embeddings.</dd></div><div><dt><M>{"E_D"}</M></dt><dd>Encode days before the path’s endpoint.</dd></div><div><dt><M>{"E_{DOW}"}</M></dt><dd>Add day-of-week information for regular seasonal patterns.</dd></div><div><dt><M>{"\\operatorname{tAPE}"}</M></dt><dd>Add an encoding of sequence position, adapted to smaller embedding dimensions.</dd></div></dl></div>
    <p><strong>Why both position and time?</strong> The second touch can occur seconds or weeks after the first. Position records order; days before the endpoint locate each event in time. Differences between those distances reveal the spacing between touches. Neither is the conversion outcome. <PaperLink section="S2.SS3">Positional representations</PaperLink></p>
    <details className="paper-details"><summary>Technical detail: positional frequencies and observation cutoffs</summary><div><p>The paper starts from sine/cosine position encodings, then rescales their frequency for the time-series setting:</p><M block>{"\\omega_k=10000^{-2k/d_{\\rm model}},\\qquad \\omega_k^{\\rm new}=\\omega_k\\frac{d_{\\rm model}}{L}"}</M><p>Here <M>{"k"}</M> indexes a frequency, <M>{"d_{\\rm model}"}</M> is the embedding dimension, and <M>{"L"}</M> is the sequence-length scale. The important idea is to give positions distinguishable numerical patterns even with a compact embedding.</p><p><strong>Our timing convention:</strong> the worked predictor uses an observation cutoff before an outcome window. The paper describes days relative to the final conversion state. Its text does not fully specify every negative-path anchoring and evaluation detail. An implementation must define those consistently; future outcomes must not leak into prediction features.</p></div></details>
  </section>;
}

const coverage = [
  ["§§2.1–2.3", "Customer context, event embeddings, campaign information, and time", "paths", "S2.SS1"],
  ["§2.4", "Attention and conversion prediction", "attention-detail", "S2.SS4"],
  ["§2.4", "Head combination, flattening, and normalized touchpoint credit", "attention-readout", "S2.SS4"],
] as const;

export function PaperReference() {
  return <section className="paper-section paper-reference" id="notes">
    <span className="eyebrow">Sources and implementation choices</span><h2>Return to the model<br />in the paper.</h2>
    <p className="section-lede">This lesson explains the attention-based conversion model and credit calculation in <a href={PAPER_URL} target="_blank" rel="noreferrer">LiDDA: Data Driven Attribution at LinkedIn, v3, 27 May 2026</a>, by John Bencina and colleagues. It is a focused teaching adaptation of the model, not a reproduction of the full production system.</p>
    <div className="coverage-table"><table><caption>Primary reading for this lesson</caption><thead><tr><th>Paper</th><th>Topic</th><th>Read</th></tr></thead><tbody>{coverage.map(([section,topic,id,anchor])=><tr key={topic}><th scope="row">{section}</th><td><a href={`#${id}`}>{topic}</a></td><td><PaperLink section={anchor}>Paper</PaperLink></td></tr>)}</tbody></table></div>
    <h3>What the example specifies</h3>
    <p>All four events belong to one advertiser. The two attention heads, scalar values, bias, and readout coefficients are fixed by hand. They illustrate the calculation; they are not fitted parameters or LinkedIn estimates.</p>
    <p>The paper specifies one self-attention layer and a flattened classification representation, but leaves ambiguity about the exact tensor being flattened, the head-combination rule, and the reduction from an attention matrix to one credit per touch. Here we explicitly average heads’ value outputs for prediction, sum incoming attention over heads and valid query positions for credit, and normalize the credits over valid source touches. These conventions make every displayed number reproducible.</p>
    <p>For the underlying attention calculation, see <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Vaswani et al., <em>Attention Is All You Need</em></a>.</p>
    <details className="paper-details"><summary>A short glossary</summary><dl className="glossary">
      <div><dt>Touchpoint</dt><dd>One event in the customer’s recorded journey.</dd></div>
      <div><dt>Embedding</dt><dd>A numerical vector representing an event, its campaign, its timing, or customer context.</dd></div>
      <div><dt>Query and key</dt><dd>Projected vectors whose dot product determines a compatibility score within a head.</dd></div>
      <div><dt>Attention head</dt><dd>One set of learned projections that computes its own mixing weights and value outputs.</dd></div>
      <div><dt>Softmax</dt><dd>Exponentiate the scores in one query row and divide by their sum. The resulting weights total one over the valid keys.</dd></div>
      <div><dt>Value vector</dt><dd>The information being mixed. It is distinct from the attention weight multiplying it.</dd></div>
      <div><dt>Flattening</dt><dd>Reshape a matrix into one feature vector while preserving its entries in a defined order.</dd></div>
      <div><dt>Readout</dt><dd>The learned mapping from the attention representation and context to a conversion prediction.</dd></div>
      <div><dt>Padding mask</dt><dd>A rule excluding unused sequence slots from attention and credit calculations.</dd></div>
      <div><dt>Credit normalization</dt><dd>Divide each valid touch’s aggregated weight by the path total to produce shares that add to one.</dd></div>
    </dl></details>
  </section>;
}
