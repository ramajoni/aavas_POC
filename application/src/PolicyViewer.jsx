import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { CircleNotch } from '@phosphor-icons/react';
import { apiOnce, titleCase } from './api';
import { Badge, InfoBox, Modal } from './components';
import './policy-viewer.css';

// What each part of the policy is, in the words a manager would use.
const SECTIONS = {
  reference: 'Reference rates', economics: 'Cost of a rupee lent', base_spread_bps: 'Base spread by segment',
  risk_grade_loading_bps: 'Loading by risk grade', ltv_loading_bps: 'Loading by loan to value',
  property_type_loading_bps: 'Loading by property type', deviation_loading_bps: 'Loading for credit deviations',
  constraints: 'Hard limits', approval_authority: 'Who may approve how much', override: 'Rules for a different rate',
  pre_assessment: 'Indicative range (Stage 1)', stage1: 'Indicative range (Stage 1)', fair_pricing: 'Fair pricing',
  optimizer: 'How the recommended rate is chosen',
};
const LABELS = {
  plr: 'Prime lending rate', card_rate_floor: 'Lowest rate any customer may get', card_rate_ceiling: 'Highest rate',
  funding_cost_pct: 'Cost of funds', servicing_cost_pct: 'Operating cost (servicing)', acquisition_cost_pct: 'Sourcing cost (one-off)',
  processing_fee_pct: 'Processing fee (one-off income)', capital_charge_pct: 'Capital charge', discount_rate_pct: 'Discount rate for value',
  behavioural_horizon_years: 'Longest modelled loan life', max_ltv: 'Highest loan to value', max_foir: 'Highest share of income on EMIs (FOIR)',
  max_tenure_years: 'Longest tenure', min_credit_score: 'Lowest credit score', new_to_credit_allowed: 'New to credit allowed',
  new_to_credit_min_spread_bps: 'Extra spread for new to credit', max_loan_amount: 'Largest loan', absolute_discount_floor_bps: 'Largest total discount from the card rate',
  max_discount_bps: 'Largest discount', max_premium_bps: 'Largest premium', max_unevidenced_discount_bps: 'Largest discount without evidence',
  bps: 'Loading', honour_indicative_band: 'Stay inside the range quoted at Stage 1', min_conversion_probability: 'Lowest acceptable chance the customer accepts',
  max_bt_out_probability: 'Highest acceptable chance the loan moves away', commercial_uplift_review_bps: 'Commercial uplift that triggers a review',
};
const SEGMENTS = { salaried_formal: 'Salaried', salaried_informal: 'Salaried (informal)', self_employed_professional: 'Self-employed professional', self_employed_non_professional: 'Self-employed' };
const label = k => LABELS[k] || SEGMENTS[k] || titleCase(k);

// Values read the way the file means them: _pct as per cent, _bps as per cent too
// (25 bps -> 0.25%), money in rupees, flags as yes or no.
function show(key, value) {
  if (value == null) return '—';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (Array.isArray(value)) return value.map(v => titleCase(String(v))).join(', ');
  if (typeof value === 'number') {
    if (/_bps$|^bps$|^per_deviation$|^max_total$/.test(key)) return `${(value / 100).toFixed(2)}%`;
    if (/_pct$/.test(key) || ['plr', 'card_rate_floor', 'card_rate_ceiling'].includes(key)) return `${value.toFixed(2)}%`;
    if (/^max_ltv$|^max_foir$|ltv$|foir_cap$/.test(key)) return `${Math.round(value * 100)}%`;
    if (/amount$/.test(key)) return `₹${value.toLocaleString('en-IN')}`;
    if (/_years$/.test(key)) return `${value} years`;
    if (/_days$/.test(key)) return `${value} days`;
  }
  return String(value);
}

function Section({ name, value, unit }) {
  if (Array.isArray(value) && value.every(v => v && typeof v === 'object')) {
    const cols = [...new Set(value.flatMap(v => Object.keys(v)))];
    return <table className="pv-table"><thead><tr>{cols.map(c => <th key={c}>{label(c)}</th>)}</tr></thead>
      <tbody>{value.map((row, i) => <tr key={i}>{cols.map(c => <td key={c}>{c === 'level' ? titleCase(row[c]) : show(c, row[c])}</td>)}</tr>)}</tbody></table>;
  }
  if (value && typeof value === 'object') {
    const rows = Object.entries(value).filter(([, v]) => v === null || typeof v !== 'object' || (Array.isArray(v) && v.every(x => typeof x !== 'object')));
    const nested = Object.entries(value).filter(([, v]) => v && typeof v === 'object' && !(Array.isArray(v) && v.every(x => typeof x !== 'object')));
    return <>{rows.length > 0 && <table className="pv-table"><tbody>{rows.map(([k, v]) => <tr key={k}><td>{label(k)}</td><td>{show(unit && typeof v === 'number' ? unit : k, v)}</td></tr>)}</tbody></table>}
      {nested.map(([k, v]) => <div key={k} className="pv-nested"><h4>{label(k)}</h4><Section name={k} value={v}/></div>)}</>;
  }
  return <p>{show(name, value)}</p>;
}

export function PolicyViewer({ version, onClose }) {
  const [current, setCurrent] = useState(version);
  const [all, setAll] = useState([]);
  const [d, setD] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { apiOnce('/pricing/policy/versions').then(r => setAll(r.versions)).catch(() => {}); }, []);
  useEffect(() => { setD(null); setError(''); apiOnce(`/pricing/policy/versions/${encodeURIComponent(current)}`).then(setD).catch(e => setError(e.message)); }, [current]);
  const p = d?.policy || {};
  const sections = Object.entries(p).filter(([k]) => !['policy_version', 'product', 'effective_from'].includes(k));
  return <Modal wide title={`Pricing policy ${current}`} onClose={onClose}>
    <div className="pv">
      <div className="pv-head">
        <div className="pv-versions" role="tablist" aria-label="Policy versions">{all.map(v =>
          <button key={v.version} type="button" role="tab" aria-selected={v.version === current} className={v.version === current ? 'active' : ''} onClick={() => setCurrent(v.version)}>
            {v.version}{v.current ? ' · current' : ''}</button>)}</div>
        {d && <p className="rv-muted">Effective from {d.effective_from || '—'} · {titleCase(p.product)}{' '}
          {d.current ? <Badge tone="green">Current version</Badge> : <Badge tone="amber">Replaced by {d.next}</Badge>}</p>}
      </div>
      {error && <InfoBox warning><span role="alert">{error}</span></InfoBox>}
      {!d && !error && <p className="rv-muted"><CircleNotch className="spin" size={16}/> Reading the policy…</p>}
      {d && d.previous && <section className="pv-changes"><h3>What changed from {d.previous}</h3>
        <ul>{d.changes_from_previous.filter(c => c.path !== 'policy_version').map(c => <li key={c.path}>
          <b>{c.path.split('.').map(label).join(' › ')}</b>: {c.change === 'added' ? `added (${show(c.path.split('.').pop(), c.after)})`
            : c.change === 'removed' ? `removed (was ${show(c.path.split('.').pop(), c.before)})`
            : `${show(c.path.split('.').pop(), c.before)} → ${show(c.path.split('.').pop(), c.after)}`}</li>)}
          {d.changes_from_previous.filter(c => c.path !== 'policy_version').length === 0 && <li>No rule changed; only the version number.</li>}</ul></section>}
      {d && <div className="pv-grid">{sections.map(([k, v]) => <section key={k} className="pv-section"><h3>{SECTIONS[k] || titleCase(k)}</h3>
        <Section name={k} value={v} unit={/_bps$/.test(k) ? 'bps' : null}/></section>)}</div>}
      {d && <details className="pv-raw"><summary>The policy file as written</summary><pre>{d.raw}</pre></details>}
    </div>
  </Modal>;
}

// A policy version that opens that version of the policy.
export function PolicyLink({ version, children }) {
  const [open, setOpen] = useState(false);
  if (!version) return children || '—';
  return <>
    <button type="button" className="link-button pv-link" onClick={() => setOpen(true)} title={`Read pricing policy ${version}`}>{children || version}</button>
    {/* Rendered at the page root: the link often sits inside a details list whose
        styles would otherwise reach into the viewer. */}
    {open && createPortal(<PolicyViewer version={version} onClose={() => setOpen(false)}/>, document.body)}
  </>;
}
