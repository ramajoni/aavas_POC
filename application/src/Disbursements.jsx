import React, { useEffect, useState } from 'react';
import { ArrowClockwise, ChartBar } from '@phosphor-icons/react';
import { api } from './api';
import { Button, Empty, InfoBox } from './components';
import './disbursements.css';

const crore = n => n == null ? '—' : n >= 1e7 ? `₹${(n / 1e7).toFixed(2)} Cr` : `₹${(n / 1e5).toFixed(1)} L`;
const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const bps = n => n == null ? '—' : `${Math.round(n)} bps`;
const monthName = m => new Date(`${m}-01T00:00:00`).toLocaleDateString('en-IN', { month: 'long', year: 'numeric' });

// Disbursed against target, with a tick where the month's pace says it should be.
function Progress({ value, target, pace }) {
  if (!target) return <span className="muted">No target set</span>;
  const pct = Math.min(value / target, 1);
  return <span className="db-progress" title={`${crore(value)} of ${crore(target)}`}>
    <span className="db-bar"><i style={{ width: `${pct * 100}%` }}/><em style={{ left: `${Math.min(pace, 1) * 100}%` }}/></span>
    <span>{crore(value)} <small>of {crore(target)}</small></span>
  </span>;
}

// Area manager: the area's disbursements this month against target, by RM, with
// each RM's pricing pattern beside it.
export function AreaDisbursements({ user }) {
  const [d, setD] = useState(null);
  const [error, setError] = useState('');
  const load = () => { setError(''); api('/pricing/area/disbursements').then(setD).catch(e => setError(e.message)); };
  useEffect(load, []);
  return <div className="operations db">
    <div className="page-intro"><p className="eyebrow">{user.name} · {user.title}</p><h1>Disbursements</h1></div>
    {error && <InfoBox warning><span role="alert">{error}</span></InfoBox>}
    {d && <>
      <div className="app-kpis">
        <div className="app-kpi primary"><small>Disbursed in {monthName(d.month)}</small><b>{crore(d.area.disbursed_inr)}</b>
          <em>{share(d.area.pct_of_target)} of {crore(d.area.target_inr)} target</em></div>
        <div className="app-kpi"><small>Where the month says we should be</small><b>{crore(d.area.expected_by_now_inr)}</b>
          <em>Day {d.day} of {d.days_in_month}</em></div>
        <div className="app-kpi"><small>Files disbursed</small><b>{d.area.files}</b><em>this month</em></div>
      </div>
      {!d.rms.length ? <Empty icon={ChartBar} title="No relationship managers">Relationship managers in this area appear here.</Empty>
        : <div className="app-table"><table>
          <thead>
            <tr className="group"><th></th><th colSpan="3">Month to date</th><th colSpan="3">Offers</th><th colSpan="3">Lower-rate requests</th></tr>
            <tr><th>RM</th><th>Disbursed against target</th><th>Files</th><th>On pace</th>
              <th>Sent</th><th>Taken</th><th>Below recommendation</th><th>Asked</th><th>Approved</th><th>Average asked</th></tr></thead>
          <tbody>{d.rms.map(r => <tr key={r.username}>
            <td><b>{r.name}</b><small>{r.username}</small></td>
            <td><Progress value={r.disbursed_inr} target={r.target_inr} pace={d.pace}/></td>
            <td>{r.files}</td>
            <td className={r.on_pace === false ? 'late' : ''}>{r.on_pace == null ? '—' : r.on_pace ? 'Yes' : `Behind by ${crore(r.expected_by_now_inr - r.disbursed_inr)}`}</td>
            <td>{r.offers_sent}</td><td>{share(r.acceptance_rate)}</td><td>{bps(r.avg_offer_below_recommendation_bps)}</td>
            <td>{r.requests}{r.pending ? <small>{r.pending} waiting</small> : null}</td>
            <td>{r.approved + r.declined ? `${r.approved} of ${r.approved + r.declined}` : '—'}</td>
            <td>{bps(r.avg_discount_asked_bps)}</td>
          </tr>)}</tbody></table></div>}
      <p className="small-copy">{d.note} “Below recommendation” is how far, on average, an RM’s offers sit under the system recommendation; “Average asked” is the depth of the lower rates they ask you for.</p>
    </>}
    <div className="actions end"><Button onClick={load}><ArrowClockwise/> Refresh</Button></div>
  </div>;
}

// The RM's own month, as a strip at the top of My applications.
export function MyMonth() {
  const [d, setD] = useState(null);
  useEffect(() => { api('/pricing/my/disbursements').then(setD).catch(() => setD(null)); }, []);
  const me = d?.rms?.[0];
  if (!me) return null;
  return <div className="db-strip" aria-label="My month">
    <div><small>Disbursed in {monthName(d.month)}</small><Progress value={me.disbursed_inr} target={me.target_inr} pace={d.pace}/></div>
    <div><small>Sanctions sent</small><b>{me.offers_sent}</b></div>
    <div><small>Conversion</small><b>{share(me.acceptance_rate)}</b><em>offers taken</em></div>
    <div><small>Files disbursed</small><b>{me.files}</b></div>
  </div>;
}
