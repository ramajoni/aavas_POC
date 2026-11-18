import React, { useEffect, useRef, useState } from 'react';
import { CircleNotch } from '@phosphor-icons/react';
import { api, money, apiOnce } from './api';
import { InfoBox } from './components';
import { Fold } from './Fold';
import { DeepDiveContent } from './DeepDive';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const pts = v => v == null ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}`;
// Two series in fixed categorical order: the general model, then this branch.
const MODEL = '#2a78d6', BRANCH = '#eb6834';

function Curve({ d }) {
  const [hover, setHover] = useState(null);
  const svg = useRef(null);
  const c = d.curve;
  if (!c.length) return null;
  const W = 760, H = 300, L = 46, R = 120, T = 40, B = 34;
  const x0 = c[0].rate, x1 = c[c.length - 1].rate;
  const all = c.flatMap(p => [p.acceptance_model, p.acceptance_branch]);
  // Round the scale to whole steps of 5% (10% on a wide spread) so the axis reads cleanly.
  const step = Math.max(...all) - Math.min(...all) > 0.3 ? 0.1 : 0.05;
  const y0 = Math.max(0, Math.floor((Math.min(...all) - 0.02) / step) * step), y1 = Math.min(1, Math.ceil((Math.max(...all) + 0.02) / step) * step);
  const x = r => L + ((r - x0) / (x1 - x0)) * (W - L - R);
  const y = p => T + (1 - (p - y0) / (y1 - y0)) * (H - T - B);
  const line = key => c.map((p, i) => `${i ? 'L' : 'M'}${x(p.rate).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ');
  const ticksY = []; for (let v = y0; v <= y1 + 1e-9; v += step) ticksY.push(+v.toFixed(2));
  const ticksX = []; for (let r = Math.ceil(x0 * 2) / 2; r <= x1 + 1e-9; r += 0.5) ticksX.push(r);
  const marks = [['RM asks', d.requested_rate, 'ask'], ['Recommended', d.recommended_rate, 'rec']]
    .filter(([, r]) => r != null && r >= x0 && r <= x1);
  const last = c[c.length - 1];
  function move(e) {
    const box = svg.current.getBoundingClientRect();
    const r = x0 + ((((e.clientX - box.left) / box.width) * W - L) / (W - L - R)) * (x1 - x0);
    const near = c.reduce((b, p) => Math.abs(p.rate - r) < Math.abs(b.rate - r) ? p : b, c[0]);
    setHover(r < x0 - 0.05 || r > x1 + 0.05 ? null : near);
  }
  return <figure className="sc-chart">
    <figcaption>Likely to accept at each rate: the general model, and this branch</figcaption>
    <ul className="sc-legend"><li><i style={{ background: MODEL }}/>Model, every branch</li><li><i style={{ background: BRANCH }}/>{d.calibration.branch ? d.city : 'This branch (no separate history)'}</li></ul>
    <div className="sc-plot">
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={move} onPointerLeave={() => setHover(null)}
        aria-label={`At ${pct(d.requested_rate)} the model expects ${share(d.summary.model_at_requested)} to accept; this branch ${share(d.summary.branch_at_requested)}.`}>
        {ticksY.map(v => <g key={v}><line className="sc-grid" x1={L} x2={W - R} y1={y(v)} y2={y(v)}/><text className="sc-axis" x={L - 8} y={y(v) + 4} textAnchor="end">{share(v)}</text></g>)}
        {ticksX.map(r => <text key={r} className="sc-axis" x={x(r)} y={H - B + 18} textAnchor="middle">{r.toFixed(1)}%</text>)}
        {marks.map(([label, r, k], i) => <g key={k}><line className={`sc-mark ${k}`} x1={x(r)} x2={x(r)} y1={T - 10} y2={H - B}/>
          <text className="sc-mark-label" x={x(r) + (i ? 4 : -4)} y={T - 16} textAnchor={i ? 'start' : 'end'}>{label} {pct(r)}</text></g>)}
        <path className="sc-line" stroke={MODEL} d={line('acceptance_model')}/>
        <path className="sc-line" stroke={BRANCH} d={line('acceptance_branch')}/>
        <text className="sc-end" x={x(last.rate) + 8} y={y(last.acceptance_model) + (last.acceptance_model >= last.acceptance_branch ? -4 : 12)}>Model {share(last.acceptance_model)}</text>
        <text className="sc-end" x={x(last.rate) + 8} y={y(last.acceptance_branch) + (last.acceptance_branch > last.acceptance_model ? -4 : 12)}>Branch {share(last.acceptance_branch)}</text>
        {hover && <g pointerEvents="none"><line className="sc-cross" x1={x(hover.rate)} x2={x(hover.rate)} y1={T} y2={H - B}/>
          <circle cx={x(hover.rate)} cy={y(hover.acceptance_model)} r="4.5" fill={MODEL} className="sc-dot"/>
          <circle cx={x(hover.rate)} cy={y(hover.acceptance_branch)} r="4.5" fill={BRANCH} className="sc-dot"/></g>}
        <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="transparent"/>
      </svg>
      {hover && <div className="sc-tip" style={{ left: `${(x(hover.rate) / W) * 100}%` }}><b>At {pct(hover.rate)}</b>
        <span><i style={{ background: MODEL }}/>Model <em>{share(hover.acceptance_model)}</em></span>
        <span><i style={{ background: BRANCH }}/>This branch <em>{share(hover.acceptance_branch)}</em></span></div>}
    </div>
  </figure>;
}

// The fourth review tab: every rate between the ask and the most that may be
// offered, with the general model's prediction beside what this branch's own
// history says, and how much of the difference is the branch's.
// ``curve``: off when the deep dive already draws both lines. ``part``: 'how' for the
// card on how the prediction is made, 'rates' for the table of every rate, or both.
export function ScenarioAnalysis({ requestId, curve = true, part = 'all' }) {
  const [d, setD] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { setD(null); apiOnce(`/pricing/rate-requests/${encodeURIComponent(requestId)}/scenarios`).then(setD).catch(e => setError(e.message)); }, [requestId]);
  // The table part waits quietly: the card above already shows the loading state or the error.
  if (part === 'rates' && (error || !d)) return null;
  if (error) return <InfoBox warning><span role="alert">{error}</span></InfoBox>;
  if (!d) return <section className="rv-card"><p className="rv-muted"><CircleNotch className="spin" size={16}/> Scoring every rate for this file and this branch…</p></section>;
  const s = d.summary, cal = d.calibration, br = cal.branch, hist = d.branch_history;
  const gap = br ? br.gap_pts - cal.all_branches.gap_pts : 0;
  const atAsk = s.branch_at_requested != null && s.model_at_requested != null ? (s.branch_at_requested - s.model_at_requested) * 100 : null;
  const tone = hist?.verdict === 'value creating' ? 'good' : hist?.verdict === 'value destroying' ? 'bad' : 'warn';
  return <div className="rv-stack">
    {part !== 'rates' && <section className="rv-card">
      <h2>How the prediction is made, and how this branch differs</h2>
      <div className="sc-cards">
        <div className="model"><small>General prediction</small>
          <b>{share(s.model_at_requested)} at {pct(d.requested_rate)} · {share(s.model_at_recommended)} at {pct(d.recommended_rate)}</b>
          <span>The acceptance model, trained on every branch’s loans, scored on this file with only the rate changed. It expects the discount to change acceptance by {pts(s.discount_buys_model_pts)} points.</span></div>
        <div className="branch"><small>This branch{br ? ` · ${d.city}` : ''}</small>
          <b>{share(s.branch_at_requested)} at {pct(d.requested_rate)} · {share(s.branch_at_recommended)} at {pct(d.recommended_rate)}</b>
          <span>{br ? <>On {br.files} recent {d.city} loans the model had not seen, customers took the loan {Math.abs(gap).toFixed(1)} points {gap >= 0 ? 'more' : 'less'} often than it predicted, beyond what every branch shows. That difference is carried onto this file: {pts(atAsk)} points at the ask.</>
            : cal.note}</span></div>
        <div className={`decisions ${tone}`}><small>This branch’s own decisions</small>
          <b>{hist ? `${pts(hist.s2d_lift_pts)} pts acceptance · ${hist.value_lift_inr == null ? '—' : `${hist.value_lift_inr < 0 ? '−' : '+'}${money(Math.abs(hist.value_lift_inr))}`} a loan` : 'No comparison'}</b>
          <span>{hist ? <>What past discounts here{hist.label && hist.label !== 'all discounts' ? ' for a lower competitor rate' : ''} actually did against comparable loans left at the recommendation ({hist.files} loans), where the model expects {pts(s.discount_buys_model_pts)} points for this one.</> : 'Not enough history to compare.'}</span></div>
      </div>
      {curve && <Curve d={d}/>}
    </section>}
    {part !== 'how' && <Fold title="Every rate from the ask to the most that may be offered" summary={`${d.table.length} rates · EMI, acceptance (model and branch), expected value, net margin and interest given up at each`}>
      <div className="table-scroll"><table className="sc-table">
        <thead><tr><th rowSpan="2">Rate</th><th rowSpan="2">EMI</th><th colSpan="3">Likely to accept</th><th colSpan="2">Expected value</th><th rowSpan="2">Net margin</th><th rowSpan="2">Interest given up over the loan</th></tr>
          <tr><th>Model</th><th>This branch</th><th>Difference</th><th>Model</th><th>This branch</th></tr></thead>
        <tbody>{d.table.map(r => <tr key={r.rate} className={r.mark || ''}>
          <td><b>{pct(r.rate)}</b>{r.mark && <small>{r.mark === 'requested' ? 'RM asks' : 'Recommended'}</small>}</td>
          <td>{money(r.emi)}</td><td>{share(r.acceptance_model)}</td><td>{share(r.acceptance_branch)}</td>
          <td>{pts((r.acceptance_branch - r.acceptance_model) * 100)}</td>
          <td>{money(r.value_model_inr)}</td><td>{money(r.value_branch_inr)}</td>
          <td>{pct(r.net_margin_pct)}</td><td>{r.interest_given_up_inr > 0 ? money(r.interest_given_up_inr) : '—'}</td></tr>)}</tbody></table></div>
      <p className="rv-note">{d.method} Value changes from the recommendation to the ask: model {s.value_change_model_inr == null ? '—' : money(s.value_change_model_inr)}, this branch {s.value_change_branch_inr == null ? '—' : money(s.value_change_branch_inr)}.</p>
    </Fold>}
  </div>;
}

// Price, acceptance and scenarios on one tab: how the prediction is made and how this
// branch differs from the model first, then the floor, the ask, the recommendation
// and the ceiling on one curve for every branch and this branch, how the floor is
// built, and every rate in between. Both scenario parts share one cached request.
export function PriceScenarios({ requestId }) {
  return <div className="rv-stack">
    <ScenarioAnalysis requestId={requestId} curve={false} part="how"/>
    <DeepDiveContent requestId={requestId} title="Price and acceptance: every branch against this branch"/>
    <ScenarioAnalysis requestId={requestId} curve={false} part="rates"/>
  </div>;
}
