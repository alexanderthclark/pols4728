import "./training-examples.css";

const examples = [
  { customer: "A", touches: ["Google Display", "Instagram", "YouTube", "Google Search", null], converted: 1 },
  { customer: "B", touches: ["Facebook", "Google Search", null, null, null], converted: 0 },
  { customer: "C", touches: ["Google Search", "Instagram", "YouTube", "Facebook", "Google Search"], converted: 1 },
  { customer: "D", touches: ["YouTube", "Google Display", "Instagram", null, null], converted: 0 },
] as const;

export function TrainingExamples() {
  return <section className="training-examples" id="training-data" aria-labelledby="training-data-title">
    <h2 id="training-data-title">A few training examples</h2>
    <p className="training-intro">Each row is one customer’s recorded ad journey for the same advertiser. Read each sequence from left to right. The label says whether the customer converts in the following seven days.</p>
    <table className="training-table" aria-describedby="training-data-notes">
      <caption>Synthetic data · up to five retained touchpoints per journey</caption>
      <thead><tr>
        <th scope="col">Customer</th>
        <th scope="col">Input: ad touchpoints <span className="training-header-note">Earlier to later, before the prediction cutoff</span></th>
        <th scope="col" className="training-label-heading">Label <var>y</var><span className="training-header-note">Conversion in the next 7 days</span></th>
      </tr></thead>
      <tbody>{examples.map(example => <tr key={example.customer}>
        <th scope="row"><span className="training-customer-prefix">Customer </span>{example.customer}</th>
        <td className="training-input"><ol className="training-sequence" aria-label={`Customer ${example.customer}: five input slots, in time order`}>
          {example.touches.map((touch, position) => <li key={position} className={`training-token${touch === null ? " training-pad" : ""}`}>
            <span className="training-position" aria-hidden="true">{position + 1}</span>
            <span>{touch ?? "PAD"}</span>
          </li>)}
        </ol></td>
        <td className="training-label" aria-label={example.converted ? "1: converted during the following seven days" : "0: no conversion during the following seven days"}>
          <span className="training-mobile-label" aria-hidden="true"><var>y</var> = </span><span className="training-label-value">{example.converted}</span>
        </td>
      </tr>)}</tbody>
    </table>
    <div className="training-notes" id="training-data-notes">
      <p><code>PAD</code> fills an unused slot. It is not an ad exposure and must be masked. Each repeated platform is a separate touch.</p>
      <p><span className="training-outcome-key">1 = converted · 0 = did not convert</span>Both labels use the complete seven-day outcome window. Five slots and seven days are choices for this illustration.</p>
    </div>
  </section>;
}
