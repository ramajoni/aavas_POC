import React, { useMemo } from 'react';
import { Minus, Plus, Target } from '@phosphor-icons/react';
import { emi, money, percent } from './api';
import './scenario.css';

const nearest = (rows, rate) => rows.reduce((best, s, i) => (Math.abs(s.rate - rate) < Math.abs(rows[best].rate - rate) ? i : best), 0);
const lakh = n => `₹${(n / 100000).toFixed(2)} L`;
const rupees = v => `${v > 0 ? '+' : v < 0 ? '−' : '±'}₹${Math.abs(Math.round(v)).toLocaleString('en-IN')}`;
const signed = (v, unit, digits = 1) => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(digits)}${unit}`;

// Rate vs outcome, as a slider over the priced scenarios. Moving it sets the
// proposed rate the override form uses, so what is read here is what gets proposed.
export function ScenarioSlider({ scenarios, recommended, selected, onSelect, loanAmount, years, competitor }) {
  const rows = useMemo(() => [...scenarios].sort((a, b) => a.rate - b.rate), [scenarios]);
  if (!rows.length) return null;
  const recIndex = nearest(rows, recommended);
  const index = nearest(rows, Number(selected) || recommended);
  const s = rows[index], r = rows[recIndex];
  const lo = rows[0].rate, hi = rows[rows.length - 1].rate;
  const at = rate => `${((rate - lo) / (hi - lo || 1)) * 100}%`;
  const pick = i => onSelect(rows[Math.max(0, Math.min(rows.length - 1, i))].rate);
  const sEmi = emi(loanAmount, s.rate, years), rEmi = emi(loanAmount, r.rate, years);

  const metrics = [
    ['Customer acceptance', percent(s.conversion_probability * 100, 1), signed((s.conversion_probability - r.conversion_probability) * 100, ' pts'), s.conversion_probability >= r.conversion_probability],
    ['Net annual margin', percent(s.net_margin_pct), signed(s.net_margin_pct - r.net_margin_pct, ' pts', 2), s.net_margin_pct >= r.net_margin_pct],
    ['Monthly EMI', money(sEmi), rupees(sEmi - rEmi), sEmi <= rEmi],
    ['Expected value', lakh(s.expected_value_inr), signed((s.expected_value_inr - r.expected_value_inr) / 1000, 'k'), s.expected_value_inr >= r.expected_value_inr],
    ['Balance-transfer risk', percent(s.bt_out_probability * 100, 1), signed((s.bt_out_probability - r.bt_out_probability) * 100, ' pts'), s.bt_out_probability <= r.bt_out_probability],
    ['Expected life', `${s.expected_life_years.toFixed(1)} yrs`, signed(s.expected_life_years - r.expected_life_years, ' yrs'), s.expected_life_years >= r.expected_life_years],
  ];

  return <div className="scenario-slider">
    <div className="scenario-head">
      <div>
        <small>Proposed rate</small>
        <strong>{percent(s.rate)}</strong>
        {index === recIndex
          ? <span className="scenario-tag rec"><Target size={14}/> Recommended</span>
          : <span className="scenario-tag">{signed((s.rate - r.rate) * 100, ' bps', 0)} vs recommended {percent(r.rate)}</span>}
      </div>
      <div className="scenario-steps">
        <button type="button" aria-label="Lower the rate one step" onClick={() => pick(index - 1)} disabled={index === 0}><Minus size={16}/></button>
        <button type="button" onClick={() => pick(recIndex)} disabled={index === recIndex}>Reset to recommended</button>
        <button type="button" aria-label="Raise the rate one step" onClick={() => pick(index + 1)} disabled={index === rows.length - 1}><Plus size={16}/></button>
      </div>
    </div>

    <div className="scenario-track">
      <input type="range" min={0} max={rows.length - 1} step={1} value={index}
        onChange={e => pick(Number(e.target.value))}
        aria-label="Proposed interest rate" aria-valuetext={`${s.rate.toFixed(2)}%`}/>
      <div className="scenario-marks" aria-hidden="true">
        <span className="mark rec" style={{ left: at(r.rate) }}>Recommended {r.rate.toFixed(2)}%</span>
        {competitor && competitor >= lo && competitor <= hi &&
          <span className="mark comp" style={{ left: at(competitor) }}>Competitor {Number(competitor).toFixed(2)}%</span>}
      </div>
      <div className="scenario-ends"><span>{lo.toFixed(2)}%</span><span>{hi.toFixed(2)}%</span></div>
    </div>

    <Curve rows={rows} index={index} recIndex={recIndex}/>

    <div className="scenario-metrics">
      {metrics.map(([label, value, delta, better]) => <div key={label}>
        <small>{label}</small><b>{value}</b>
        {index !== recIndex && <em className={better ? 'up' : 'down'}>{delta}</em>}
      </div>)}
    </div>
    <p className="small-copy">Deltas are against the recommended rate. Moving the slider sets the proposed rate in the override form; rates are recommendations, not approvals.</p>
  </div>;
}

// Acceptance and expected value across the scenario range, with the proposed rate marked.
function Curve({ rows, index, recIndex }) {
  const W = 600, H = 130, P = 8;
  const x = i => P + (i / (rows.length - 1 || 1)) * (W - 2 * P);
  const scale = values => { const min = Math.min(...values), max = Math.max(...values); return v => H - P - ((v - min) / (max - min || 1)) * (H - 2 * P); };
  const conv = scale(rows.map(s => s.conversion_probability));
  const value = scale(rows.map(s => s.expected_value_inr));
  const line = (f, key) => rows.map((s, i) => `${x(i).toFixed(1)},${f(s[key]).toFixed(1)}`).join(' ');
  return <figure className="scenario-curve">
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
      aria-label="Customer acceptance falls and expected value peaks as the rate rises">
      <line x1={x(recIndex)} x2={x(recIndex)} y1={P} y2={H - P} className="rec-line"/>
      <polyline points={line(conv, 'conversion_probability')} className="conv"/>
      <polyline points={line(value, 'expected_value_inr')} className="value"/>
      <line x1={x(index)} x2={x(index)} y1={0} y2={H} className="now-line"/>
      <circle cx={x(index)} cy={conv(rows[index].conversion_probability)} r="5" className="conv-dot"/>
      <circle cx={x(index)} cy={value(rows[index].expected_value_inr)} r="5" className="value-dot"/>
    </svg>
    <figcaption><span><i className="k conv"/> Customer acceptance</span><span><i className="k value"/> Expected value</span><span><i className="k rec"/> Recommended</span></figcaption>
  </figure>;
}
