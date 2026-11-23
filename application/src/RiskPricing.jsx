import React, { useEffect, useState } from 'react';
import { CircleNotch } from '@phosphor-icons/react';
import { money, apiOnce } from './api';
import { InfoBox } from './components';
import './risk-pricing.css';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const frac = (v, d = 1) => v == null ? '—' : `${(Number(v) * 100).toFixed(d)}%`;
const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const signed = v => v == null ? '' : ` (${v > 0 ? '+' : ''}${v.toFixed(2)})`;
const MARKS = { floor: 'Floor', ask: 'RM asks', recommended: 'Recommended', card: 'Card rate' };

function BandBar({ band, pd }) {
  const { pd_low: lo, pd_high: hi, position_in_band: pos } = band;
  const label = `Yearly default risk ${frac(pd, 2)}, in grade ${band.grade} which covers ${hi == null ? `above ${frac(lo, 1)}` : `${frac(lo, 1)} to ${frac(hi, 1)}`}${pos == null ? '' : `; ${Math.round(pos * 100)}% of the way from the better grade to the worse`}.`;
  return <div className="rk-band">
    <div className="rk-band-ends">
      <span>{band.better_grade ? <>Grade {band.better_grade}<small>card {pct(band.card_rate_better_grade)}{signed(band.card_change_if_better_pts)}</small></> : <small>Best grade</small>}</span>
      <span className="here">Grade {band.grade}<small>card {pct(band.card_rate_this_grade)} at {band.tenure_months} months</small></span>
      <span>{band.worse_grade ? <>Grade {band.worse_grade}<small>card {pct(band.card_rate_worse_grade)}{signed(band.card_change_if_worse_pts)}</small></> : <small>Lowest grade</small>}</span>
    </div>
    <div className="rk-track" role="img" aria-label={label}>
      {pos != null && <i style={{ left: `${Math.min(Math.max(pos, 0), 1) * 100}%` }}/>}
    </div>
    <div className="rk-band-scale"><span>{frac(lo, 1)} a year</span><span>{hi == null ? '' : `${frac(hi, 1)} a year`}</span></div>
    <p className="rv-note">{band.pd_distance_to_better_grade != null && <>{frac(band.pd_distance_to_better_grade, 2)} lower would be grade {band.better_grade}. </>}{band.pd_distance_to_worse_grade != null && <>{frac(band.pd_distance_to_worse_grade, 2)} higher would be grade {band.worse_grade}. </>}{band.borderline && <b>Close to a boundary.</b>}</p>
  </div>;
}

// The risk behind the price: the file's grade and where it sits in it, what moves it,
// and what each rate does to affordability, risk cost, acceptance and value. Decision
// support only: the recommended price, the floor and the band are set by the engine.
export function RiskPricing({ requestId }) {
  const [d, setD] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { setD(null); setError(''); apiOnce(`/pricing/rate-requests/${encodeURIComponent(requestId)}/risk-pricing`).then(setD).catch(e => setError(e.message)); }, [requestId]);
  if (error) return <InfoBox warning><span role="alert">{error}</span></InfoBox>;
  if (!d) return <section className="rv-card"><p className="rv-muted"><CircleNotch className="spin" size={16}/> Working out the risk at each rate…</p></section>;
  const s = d.score, here = d.peers?.this_file;
  const cheaper = v => v == null ? '—' : `${Math.round(100 - v)}%`;
  return <section className="rv-card rk" aria-labelledby="rk-title">
    <h2 id="rk-title">Risk-aware pricing</h2>
    <p className={`rk-verdict ${d.verdict.tone}`}>{d.verdict.text}</p>
    <div className="rk-tiles">
      <div><small>Risk grade</small><b>{s.risk_grade ?? '—'}</b><em>{s.source === 'model' ? 'From the risk model' : 'Fixed on the file'}</em></div>
      <div><small>Chance of default</small><b>{frac(s.pd_annual, 2)} a year</b><em>{frac(s.pd_lifetime, 1)} over the loan</em></div>
      <div><small>Loss if it defaults</small><b>{frac(s.lgd, 0)}</b><em>of the balance</em></div>
      <div><small>Expected loss</small><b>{s.expected_loss_pct_per_year == null ? '—' : `${s.expected_loss_pct_per_year.toFixed(2)}% a year`}</b><em>{s.expected_loss_inr_over_life == null ? '—' : `${money(s.expected_loss_inr_over_life)} over ${s.expected_life_years?.toFixed(1)} years`}</em></div>
      <div><small>Break-even rate</small><b>{pct(s.breakeven_rate)}</b><em>Below this the loan loses money</em></div>
    </div>
    <div className="rk-grid2">
      <section><h3>Where this file sits in its grade</h3><BandBar band={d.band} pd={s.pd_annual}/></section>
      <section><h3>What moves this file’s risk</h3>
        {d.drivers.length ? <ul className="rk-drivers">{d.drivers.map(x => <li key={x.feature}>
          <span>{x.label}<small>{x.this_value} here, {x.typical_value} typical</small></span>
          <b className={x.direction === 'raises risk' ? 'up' : 'down'}>{x.pd_change_pts > 0 ? '+' : '−'}{Math.abs(x.pd_change_pts).toFixed(2)} pts<small>{x.direction}</small></b></li>)}</ul>
          : <p className="rv-note">Not available without the risk model.</p>}
        <p className="rv-note">Each figure is the change in yearly default risk against a typical borrower in the book, one factor at a time.</p>
      </section>
    </div>
    <h3>Each rate, with the risk at that rate</h3>
    <div className="table-scroll"><table className="sc-table rk-table">
      <caption className="sr-only">Affordability, risk, margin, acceptance and value at each rate</caption>
      <thead><tr><th scope="col">Rate</th><th scope="col">EMI</th><th scope="col">Share of income to EMIs</th><th scope="col">If rates rose {d.stress_add_pct ?? 2} points (limit)</th><th scope="col">Default risk</th><th scope="col">Expected loss</th><th scope="col">Break-even</th><th scope="col">Margin over risk cost</th><th scope="col">Likely to accept</th><th scope="col">Risk-adjusted value</th></tr></thead>
      <tbody>{d.rates.map(r => <tr key={r.rate} className={r.marks.map(m => `m-${m}`).join(' ')}>
        <th scope="row"><b>{pct(r.rate)}</b>{r.marks.map(m => <small key={m}>{MARKS[m]}</small>)}{!r.in_band && <small className="bad">outside the policy band for this file</small>}</th>
        <td>{money(r.emi)}</td><td>{frac(r.foir)}</td>
        <td className={r.stressed_foir_ok ? '' : 'bad'}>{frac(r.stressed_foir)} ({frac(r.foir_cap, 0)})<small>{r.stressed_foir_ok ? 'passes' : 'fails'}</small></td>
        <td>{frac(r.pd_annual, 2)}{r.risk_grade_at_rate !== d.band.grade && <small>grade {r.risk_grade_at_rate}</small>}</td>
        <td>{r.expected_loss_pct.toFixed(2)}%</td><td>{pct(r.breakeven_rate)}</td><td>{r.margin_over_risk_cost.toFixed(2)} pts</td>
        <td>{share(r.acceptance_branch ?? r.acceptance_model)}</td><td>{money(r.value_branch_inr ?? r.value_model_inr)}</td></tr>)}</tbody></table></div>
    <p className="rv-note">Acceptance and value are for this branch, as in the table of every rate. {d.pd_note}</p>
    <h3>This grade against the book</h3>
    {d.peers?.grades?.length ? <>
      <div className="table-scroll"><table className="sc-table rk-table">
        <caption className="sr-only">Each risk grade in the loan book</caption>
        <thead><tr><th scope="col">Grade</th><th scope="col">Loans</th><th scope="col">Default risk a year</th><th scope="col">Card rate</th><th scope="col">Lowest rate allowed</th><th scope="col">Typical rate (middle half)</th><th scope="col">Defaulted in the book</th></tr></thead>
        <tbody>{d.peers.grades.map(g => <tr key={g.grade} className={g.this_file ? 'm-this' : ''}>
          <th scope="row"><b>{g.grade}</b>{g.this_file && <small>this file</small>}</th><td>{g.files.toLocaleString('en-IN')}</td>
          <td>{frac(g.pd_low, 1)} to {g.pd_high == null ? '—' : frac(g.pd_high, 1)}</td><td>{pct(g.card_rate)}</td><td>{pct(g.floor_rate)}</td>
          <td>{pct(g.median_rate)} ({pct(g.rate_p25)} to {pct(g.rate_p75)})</td>
          <td>{frac(g.realised_default_rate, 1)}<small>{frac(g.realised_default_rate_per_year, 2)} a year</small></td></tr>)}</tbody></table></div>
      {here && <p className="rk-pctl">Among grade {here.grade} loans, the ask is cheaper than {cheaper(here.ask_percentile_same_grade)} and the recommendation cheaper than {cheaper(here.recommended_percentile_same_grade)} of what the book charged{here.ask_percentile_same_grade_and_tenure != null ? `; among those at ${here.tenure_band}, ${cheaper(here.ask_percentile_same_grade_and_tenure)} and ${cheaper(here.recommended_percentile_same_grade_and_tenure)}` : ''}.</p>}
    </> : <p className="rv-note">The loan book is not available to compare against.</p>}
    {d.limitations?.length > 0 && <ul className="rk-limits">{d.limitations.map(t => <li key={t}>{t}</li>)}</ul>}
    <p className="rv-note">{d.note}</p>
  </section>;
}
