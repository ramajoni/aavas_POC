import React, { useEffect, useRef, useState } from 'react';
import { CircleNotch } from '@phosphor-icons/react';
import { api, money } from './api';
import { InfoBox, Modal } from './components';
import './deep-dive.css';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;

// The floor, built from transparent components, from a decision's value
// decomposition: funding, servicing and capital from the pricing policy, and the
// expected credit loss from the file's probability of default and loss given default.
export function buildUp(v) {
  if (!v) return null;
  const parts = [['Cost of funds', v.less_funding_cost], ['Operating cost (servicing)', v.less_servicing_cost],
    ['Capital charge', v.less_capital_charge], ['Expected credit loss', v.less_expected_credit_loss]].filter(([, x]) => x != null);
  if (!parts.length) return null;
  const components = parts.map(([label, x]) => ({ label, pct: Math.abs(Number(x)) }));
  const breakeven = components.reduce((sum, c) => sum + c.pct, 0);
  return { components, breakeven_rate: breakeven, recommended_rate: v.customer_rate ?? null,
    margin_over_breakeven_pct: v.customer_rate != null ? Number(v.customer_rate) - breakeven : null };
}

export function FloorTable({ b }) {
  if (!b) return null;
  return <><table className="dd-table"><tbody>
    {b.components.map(c => <tr key={c.label}><td>{c.label}</td><td>{c.pct.toFixed(2)}%</td></tr>)}
    <tr className="total"><td>Break-even rate</td><td>{b.breakeven_rate.toFixed(2)}%</td></tr>
    {b.recommended_rate != null && <tr><td>Recommended rate</td><td>{pct(b.recommended_rate)}</td></tr>}
    {b.margin_over_breakeven_pct != null && <tr className="margin"><td>Margin above break-even</td><td>{b.margin_over_breakeven_pct.toFixed(2)}%</td></tr>}
  </tbody></table>
  <p className="dd-note">Every component is a stated rate: funding, servicing and capital from the pricing policy, the expected loss from this file’s probability of default and loss given default.</p></>;
}

// The acceptance curve for one file: how likely the customer is to take the loan at
// each rate, with the floor (break-even), the RM's ask, the recommendation and the
// ceiling (the highest rate the customer is still likely to accept) marked on it.
function AcceptanceCurve({ curve, marks }) {
  const W = 720, H = 290, L = 46, R = 18, T = 54, B = 38;
  const [hover, setHover] = useState(null);
  const svg = useRef(null);
  if (!curve.length) return null;
  const x0 = curve[0].rate, x1 = curve[curve.length - 1].rate;
  const x = r => L + ((r - x0) / Math.max(x1 - x0, 1e-9)) * (W - L - R);
  const y = p => T + (1 - p) * (H - T - B);
  const path = curve.map((p, i) => `${i ? 'L' : 'M'}${x(p.rate).toFixed(1)},${y(p.acceptance).toFixed(1)}`).join(' ');
  const ticks = []; for (let r = Math.ceil(x0 * 2) / 2; r <= x1 + 1e-9; r += 0.5) ticks.push(+r.toFixed(2));
  const at = r => curve.reduce((best, p) => (Math.abs(p.rate - r) < Math.abs(best.rate - r) ? p : best), curve[0]);
  // Marks at the same rate share one line and one label; labels are placed on the
  // first of three rows where they do not run into the label before them.
  const merged = [];
  for (const m of marks.filter(m => m.rate != null && m.rate >= x0 && m.rate <= x1)) {
    const same = merged.find(o => Math.abs(o.rate - m.rate) < 0.005);
    if (same) { same.label = `${same.label} · ${m.label}`; same.key = same.key === 'rec' || m.key === 'rec' ? 'rec' : same.key; }
    else merged.push({ ...m });
  }
  const ends = [-Infinity, -Infinity, -Infinity];
  const visible = merged.sort((a, b) => a.rate - b.rate).map(m => {
    const text = `${m.label} ${pct(m.rate)}`, w = text.length * 6.4, cx = x(m.rate);
    const anchor = cx - w / 2 < 4 ? 'start' : cx + w / 2 > W - 4 ? 'end' : 'middle';
    const left = anchor === 'start' ? cx : anchor === 'end' ? cx - w : cx - w / 2;
    let row = ends.findIndex(e => left > e + 8); if (row < 0) row = ends.indexOf(Math.min(...ends));
    ends[row] = left + w;
    return { ...m, text, anchor, row };
  });
  const floor = marks.find(m => m.key === 'floor')?.rate, ceiling = marks.find(m => m.key === 'ceiling')?.rate;

  function move(e) {
    const box = svg.current.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const r = x0 + ((px - L) / (W - L - R)) * (x1 - x0);
    setHover(at(Math.min(Math.max(r, x0), x1)));
  }
  return <figure className="dd-chart">
    <figcaption>Likely to accept, by offered rate</figcaption>
    <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Acceptance falls from ${share(curve[0].acceptance)} at ${pct(x0)} to ${share(curve[curve.length - 1].acceptance)} at ${pct(x1)}`}
      onPointerMove={move} onPointerLeave={() => setHover(null)}>
      {floor != null && ceiling != null && <rect className="dd-room" x={x(Math.max(floor, x0))} y={T} width={Math.max(x(Math.min(ceiling, x1)) - x(Math.max(floor, x0)), 0)} height={H - T - B}/>}
      {[0, 0.25, 0.5, 0.75, 1].map(p => <g key={p}><line className="dd-grid" x1={L} x2={W - R} y1={y(p)} y2={y(p)}/>
        <text className="dd-axis" x={L - 8} y={y(p) + 4} textAnchor="end">{share(p)}</text></g>)}
      {ticks.map(r => <text key={r} className="dd-axis" x={x(r)} y={H - B + 18} textAnchor="middle">{r.toFixed(1)}%</text>)}
      <path className="dd-line" d={path}/>
      {visible.map(m => <g key={m.key}>
        <line className={`dd-mark ${m.key}`} x1={x(m.rate)} x2={x(m.rate)} y1={T - 6 - m.row * 16} y2={H - B}/>
        <text className="dd-mark-label" x={x(m.rate)} y={T - 10 - m.row * 16} textAnchor={m.anchor}>{m.text}</text>
        <circle className="dd-dot" cx={x(m.rate)} cy={y(at(m.rate).acceptance)} r="4.5"/>
      </g>)}
      {hover && <g pointerEvents="none">
        <line className="dd-cross" x1={x(hover.rate)} x2={x(hover.rate)} y1={T} y2={H - B}/>
        <circle className="dd-dot hover" cx={x(hover.rate)} cy={y(hover.acceptance)} r="5"/>
      </g>}
      <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="transparent"/>
    </svg>
    {hover && <div className="dd-tip" style={{ left: `${(x(hover.rate) / W) * 100}%` }}>
      <b>{pct(hover.rate)}</b><span>{share(hover.acceptance)} likely to accept</span>
      {hover.expected_value_inr != null && <span>Expected value {money(hover.expected_value_inr)}</span>}</div>}
  </figure>;
}

export function DeepDive({ requestId, onClose }) {
  const [d, setD] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { api(`/pricing/rate-requests/${encodeURIComponent(requestId)}/deep-dive`).then(setD).catch(e => setError(e.message)); }, [requestId]);
  const accept = r => (d?.curve?.length ? d.curve.reduce((b, p) => (Math.abs(p.rate - r) < Math.abs(b.rate - r) ? p : b), d.curve[0]).acceptance : null);
  const marks = d ? [
    { key: 'floor', label: 'Break-even', rate: d.breakeven_rate },
    { key: 'ask', label: 'RM asks', rate: d.requested_rate },
    { key: 'rec', label: 'Recommended', rate: d.recommended_rate },
    { key: 'ceiling', label: 'Likely to accept up to', rate: d.ceiling_rate },
  ] : [];
  return <Modal wide title={d ? `Deep dive · ${d.applicant_name || d.loan_id}` : 'Deep dive'} onClose={onClose}>
    {error && <InfoBox warning><span role="alert">{error}</span></InfoBox>}
    {!d && !error && <p className="dd-loading"><CircleNotch className="spin" size={16}/> Drawing the acceptance curve…</p>}
    {d && <div className="dd">
      <div className="dd-points">
        <div className="floor"><small>Floor · break-even</small><b>{pct(d.breakeven_rate)}</b><em>Below this the loan loses money</em></div>
        <div className="ask"><small>RM asks</small><b>{pct(d.requested_rate)}</b><em>{share(accept(d.requested_rate))} likely to accept</em></div>
        <div className="rec"><small>Recommended</small><b>{pct(d.recommended_rate)}</b><em>{share(accept(d.recommended_rate))} likely to accept</em></div>
        <div className="ceiling"><small>Ceiling</small><b>{pct(d.ceiling_rate)}</b><em>{d.ceiling_capped_by === 'quoted_range' ? 'Top of the range quoted to the customer' : d.ceiling_capped_by === 'policy' ? 'Policy ceiling' : '7 in 10 similar customers still accept'}</em></div>
      </div>
      <AcceptanceCurve curve={d.curve} marks={marks}/>
      <div className="dd-grid2">
        {d.build_up && <section><h3>How the floor is built</h3><FloorTable b={d.build_up}/></section>}
        <section><h3>Why the rate and the curve look like this</h3>
          {d.sensitivity?.label && <p className="dd-note">This customer is <b>{d.sensitivity.label}</b>{d.sensitivity.points_lost_per_25bps == null ? '.' : d.sensitivity.points_lost_per_25bps < 0.05 ? ': at this rate, 0.25% more barely changes whether they accept.' : `: about ${d.sensitivity.points_lost_per_25bps.toFixed(1)} points of acceptance lost for every 0.25% added.`}</p>}
          <ul className="dd-reasons">{d.reasons.map(r => <li key={r.label}><span>{r.label}{typeof r.value === 'string' && r.value && !['True', 'False'].includes(r.value) ? ` (${r.value.replaceAll('_', ' ')})` : ''}</span>
            <b className={r.direction === 'decrease' ? 'down' : 'up'}>{r.direction === 'decrease' ? '−' : '+'}{Math.abs(r.impact_bps || 0).toFixed(0)} bps</b></li>)}</ul>
          {d.risk?.grade && <p className="dd-note">Risk grade {d.risk.grade}: about {d.risk.annual_pd != null ? (d.risk.annual_pd * 1000).toFixed(1) : '—'} in 1,000 similar borrowers default each year.</p>}
        </section>
      </div>
    </div>}
  </Modal>;
}
