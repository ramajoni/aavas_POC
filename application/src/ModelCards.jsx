import React, { useRef, useState } from 'react';
import { ArrowRight } from '@phosphor-icons/react';
import { Panel } from './components';
import { number } from './api';
import './model-cards.css';

const pct = (v, d = 2) => v == null ? '—' : `${Number(v).toFixed(d)}%`;
const share = (v, d = 0) => v == null ? '—' : `${(v * 100).toFixed(d)}%`;
// Categorical slots in fixed order (validated: adjacent CVD ΔE ≥ 9.1). Two sit
// below 3:1 on white, so every series is also named in the legend and the table.
const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];

// How a price is built, left to right, with the numbers each step adds.
export function RiskToPrice({ business }) {
  const floor = business?.floor, segs = business?.segments || [];
  if (!floor) return null;
  const el = segs.map(s => s.expected_loss_pct).filter(v => v != null);
  const fixed = (floor.cost_of_funds_pct || 0) + (floor.operating_cost_pct || 0) + (floor.capital_charge_pct || 0);
  const steps = [
    ['Credit risk', 'Default rate × loss if it defaults = expected credit loss a year',
      el.length ? `${pct(Math.min(...el))} to ${pct(Math.max(...el))} a year by segment` : null],
    ['Break-even floor', `Cost of funds ${pct(floor.cost_of_funds_pct)} + operating cost ${pct(floor.operating_cost_pct)} + capital ${pct(floor.capital_charge_pct)} + expected credit loss`,
      `${pct(fixed)} before credit loss`],
    ['Benchmark rate', 'Where verified files like this one were sanctioned', 'A reference, not the price'],
    ['Acceptance curve', 'How likely the customer is to take the loan at each rate', 'Differs by segment, below'],
    ['Recommended rate', 'The best value above the floor, inside the range quoted at Stage 1 and policy', 'An RM or manager decides'],
  ];
  return <section className="rtp" aria-label="From risk to price">
    <h3>From risk to price</h3>
    <ol>{steps.map(([t, d, n], i) => <li key={t}>
      <span className="rtp-n">{i + 1}</span><b>{t}</b><span>{d}</span>{n && <em>{n}</em>}
      {i < steps.length - 1 && <ArrowRight className="rtp-arrow" size={14} aria-hidden="true"/>}</li>)}</ol>
    <p className="rtp-note">Each component of the floor is a stated rate (RBI’s draft on transparent spreads). Source: {floor.source}; credit loss from the Credit risk models.</p>
  </section>;
}

// Each segment's risk beside the rate it starts at and the rate it is booked at.
export function SegmentCard({ segments }) {
  if (!segments?.length) return null;
  return <Panel className="mp-card full" title="Segments: risk and starting rate" action={<span className="mc-src">loan book · policy</span>}>
    <p className="mp-question">What does each customer segment cost us in credit loss, and where does its rate start?</p>
    <div className="table-scroll"><table className="mc-table">
      <thead><tr><th>Segment</th><th>Starting rate</th><th>Typical sanctioned</th><th>Loans</th><th>Defaulted</th><th>Per year</th><th>Loss if it defaults</th><th>Expected credit loss a year</th><th>Likely to accept</th></tr></thead>
      <tbody>{segments.map(s => <tr key={s.segment}>
        <td><b>{s.label}</b></td><td>{pct(s.starting_rate)}</td><td>{pct(s.typical_rate)}</td><td>{number(s.loans)}</td>
        <td>{share(s.default_rate, 1)}</td><td>{share(s.default_rate_annual, 2)}</td><td>{share(s.lgd, 0)}</td>
        <td><b>{pct(s.expected_loss_pct)}</b></td><td>{share(s.acceptance, 0)}</td></tr>)}</tbody></table></div>
    <p className="mc-note">Starting rate: the policy’s card-rate floor plus the segment’s base spread, before risk grade, LTV and property loadings. Defaulted: share of disbursed loans that defaulted over the book’s four-year window; per year is that over four. Expected credit loss a year = default rate a year × loss if it defaults; it is the credit part of each file’s break-even floor. Likely to accept: share of the segment’s offers that were taken.</p>
  </Panel>;
}

// Acceptance against rate, one line per segment, with a crosshair that reads
// every segment at the hovered rate.
export function AcceptanceBySegment({ curves }) {
  const [hover, setHover] = useState(null);
  const svg = useRef(null);
  const series = curves?.series || [];
  if (!series.length) return null;
  const W = 560, H = 280, L = 44, R = 14, T = 14, B = 34;
  const [x0, x1] = curves.rates;
  const x = r => L + ((r - x0) / (x1 - x0)) * (W - L - R);
  const y = p => T + (1 - p) * (H - T - B);
  const rates = series[0].points.map(p => p.rate);
  const at = (s, r) => { const pts = s.points; let best = pts[0]; for (const p of pts) if (Math.abs(p.rate - r) < Math.abs(best.rate - r)) best = p; return best.acceptance; };
  const ticks = []; for (let r = Math.ceil(x0); r <= x1 + 1e-9; r += 1) ticks.push(r);
  function move(e) {
    const box = svg.current.getBoundingClientRect();
    const r = x0 + ((((e.clientX - box.left) / box.width) * W - L) / (W - L - R)) * (x1 - x0);
    const clamped = Math.min(Math.max(r, x0), x1);
    setHover(rates.reduce((b, v) => Math.abs(v - clamped) < Math.abs(b - clamped) ? v : b, rates[0]));
  }
  const readout = hover == null ? null : series.map((s, i) => ({ label: s.label, color: SERIES[i % SERIES.length], v: at(s, hover) })).sort((a, b) => b.v - a.v);
  return <figure className="mc-curves">
    <figcaption>Likely to accept, by rate and segment</figcaption>
    <ul className="mc-legend">{series.map((s, i) => <li key={s.segment}><i style={{ background: SERIES[i % SERIES.length] }}/>{s.label}<small>typical {pct(s.typical_rate)}</small></li>)}</ul>
    <div className="mc-plot">
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={move} onPointerLeave={() => setHover(null)}
        aria-label={series.map(s => `${s.label}: ${share(at(s, s.typical_rate))} at its typical ${pct(s.typical_rate)}`).join('; ')}>
        {[0, 0.25, 0.5, 0.75, 1].map(p => <g key={p}><line className="mc-grid" x1={L} x2={W - R} y1={y(p)} y2={y(p)}/>
          <text className="mc-axis" x={L - 8} y={y(p) + 4} textAnchor="end">{share(p)}</text></g>)}
        {ticks.map(r => <text key={r} className="mc-axis" x={x(r)} y={H - B + 18} textAnchor="middle">{r}%</text>)}
        {series.map((s, i) => <path key={s.segment} className="mc-line" stroke={SERIES[i % SERIES.length]}
          d={s.points.map((p, j) => `${j ? 'L' : 'M'}${x(p.rate).toFixed(1)},${y(p.acceptance).toFixed(1)}`).join(' ')}/>)}
        {series.map((s, i) => s.typical_rate != null && <circle key={s.segment} className="mc-dot" fill={SERIES[i % SERIES.length]} cx={x(s.typical_rate)} cy={y(at(s, s.typical_rate))} r="4.5"/>)}
        {hover != null && <g pointerEvents="none"><line className="mc-cross" x1={x(hover)} x2={x(hover)} y1={T} y2={H - B}/>
          {series.map((s, i) => <circle key={s.segment} className="mc-dot" fill={SERIES[i % SERIES.length]} cx={x(hover)} cy={y(at(s, hover))} r="4"/>)}</g>}
        <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="transparent"/>
      </svg>
      {readout && <div className="mc-tip" style={{ left: `${(x(hover) / W) * 100}%` }}><b>At {pct(hover)}</b>
        {readout.map(r => <span key={r.label}><i style={{ background: r.color }}/>{r.label} <em>{share(r.v)}</em></span>)}</div>}
    </div>
    <p className="mc-note">{curves.method} Dots mark each segment’s typical sanctioned rate.</p>
  </figure>;
}

// What a 0.25% discount wins in each segment: measured from the discounts the
// bank gave, beside what the acceptance curve alone implies.
export function DiscountEffect({ effect }) {
  const rows = effect?.rows || [];
  if (!rows.length) return null;
  const max = Math.max(...rows.map(r => r.measured_high_pts || 0), 1);
  return <div className="mc-discount">
    <h4>A {effect.discount_bps / 100}% discount, by segment</h4>
    <table className="mc-table">
      <thead><tr><th>Segment</th><th>Likely to accept at typical rate</th><th>More customers won (of 100)</th><th>Curve alone</th></tr></thead>
      <tbody>{rows.map(r => <tr key={r.segment}>
        <td><b>{r.label}</b><small>typical {pct(r.typical_rate)}</small></td>
        <td>{share(r.acceptance_at_typical)}</td>
        <td><span className="mc-range" title={`${r.measured_low_pts}–${r.measured_high_pts} points (10–90%)`}>
          <span className="mc-track"><i className="mc-band" style={{ left: `${(r.measured_low_pts / max) * 100}%`, width: `${((r.measured_high_pts - r.measured_low_pts) / max) * 100}%` }}/>
            <i className="mc-mark" style={{ left: `${(r.measured_pts / max) * 100}%` }}/></span>
          <b>+{r.measured_pts}</b><small>{r.measured_low_pts}–{r.measured_high_pts}</small></span></td>
        <td>{r.curve_pts == null ? '—' : `+${r.curve_pts}`}</td></tr>)}</tbody></table>
    <p className="mc-note">{effect.method} Where the two agree, a discount wins the file; where acceptance is already high, it mostly gives margin away.</p>
  </div>;
}
