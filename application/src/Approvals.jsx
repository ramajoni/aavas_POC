import React, { useEffect, useState } from 'react';
import { ArrowClockwise, CheckCircle, Gavel, XCircle } from '@phosphor-icons/react';
import { api, money, titleCase } from './api';
import { Button, DataTable, Empty, InfoBox } from './components';
import { DeepDive } from './DeepDive';
import './applications.css';
import './price-offer.css';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const lakh = n => n == null ? '—' : `₹${(n / 100000).toFixed(1)} L`;
const crore = n => n == null ? '—' : n >= 1e7 ? `₹${(n / 1e7).toFixed(2)} Cr` : lakh(n);
const waiting = iso => { const d = Math.floor((Date.now() - new Date(iso).getTime()) / 86400000); return d <= 0 ? 'Today' : d === 1 ? '1 day' : `${d} days`; };
const when = iso => iso ? new Date(iso).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '';

// Wireframe 05, "Manager · Approval queue": lower rates the relationship managers
// asked for, sorted by the interest a year each would give up.
export function Approvals({ user, onOpen, notify }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState('');
  const [dive, setDive] = useState(null);
  const load = () => { setError(''); api('/pricing/rate-requests').then(r => setRows(r.requests)).catch(e => setError(e.message)); };
  useEffect(load, []);
  const pending = (rows || []).filter(r => r.status === 'pending').sort((a, b) => (b.analysis?.cost_per_year || 0) - (a.analysis?.cost_per_year || 0));
  const decided = (rows || []).filter(r => r.status !== 'pending');
  const month = new Date().toISOString().slice(0, 7);
  const thisMonth = decided.filter(r => (r.decided_at || '').startsWith(month));
  const value = pending.reduce((s, r) => s + (r.loan_amount || 0), 0);

  return <div className="operations my-applications">
    <div className="page-intro"><p className="eyebrow">{user.name} · {user.title}</p><h1>Approval requests</h1></div>
    <div className="app-kpis">
      <div className="app-kpi primary"><small>Waiting for you</small><b>{rows ? `${pending.length} file${pending.length === 1 ? '' : 's'}` : '—'}</b></div>
      <div className="app-kpi"><small>Loans awaiting approval</small><b>{rows ? crore(value) : '—'}</b></div>
      <div className="app-kpi"><small>Approved this month</small><b>{rows ? `${thisMonth.filter(r => r.status === 'approved').length} of ${thisMonth.length}` : '—'}</b></div>
    </div>
    {error && <InfoBox warning><span role="alert">{error}</span></InfoBox>}
    {rows && !pending.length && <Empty icon={Gavel} title="Nothing waiting">Requests appear here when a relationship manager asks for a rate below their authority.</Empty>}
    {pending.length > 0 && <div className="app-table"><table>
      <thead><tr><th>Applicant</th><th>RM</th><th>Loan</th><th>RM asks</th><th>Recommended</th><th className="key">Why a higher rate is possible</th><th>Interest given up a year</th><th>Waiting</th><th></th></tr></thead>
      <tbody>{pending.map(r => <tr key={r.request_id}>
        <td><b>{r.applicant_name || r.loan_id}</b><small>{r.loan_id}</small></td>
        <td>{r.requested_by_name || r.requested_by}</td><td>{lakh(r.loan_amount)}</td><td>{pct(r.requested_rate)}</td><td>{pct(r.recommended_rate)}</td>
        <td className="key">{r.analysis?.headroom_rate ? <>Customer likely to accept up to <b>{pct(r.analysis.headroom_rate)}</b>
          {r.analysis.headroom_capped_by === 'quoted_range' && <small>top of the range quoted to them</small>}</> : '—'}</td>
        <td>{money(r.analysis?.cost_per_year)}</td>
        <td>{waiting(r.requested_at)}</td>
        <td className="mgr-actions"><button className="link-button" onClick={() => setDive(r.request_id)}>Deep dive</button>
          <button className="link-button" onClick={() => onOpen(r)}>Review</button></td>
      </tr>)}</tbody></table></div>}
    {dive && <DeepDive requestId={dive} onClose={() => setDive(null)}/>}
    <p className="small-copy">Sorted by the interest a year each request would give up: the rate reduction times the loan amount. “Likely to accept up to” is the highest rate 7 in 10 similar customers still accept, never above the range the customer was quoted.</p>
    {decided.length > 0 && <>
      <h2 className="mgr-history">Decided</h2>
      <DataTable columns={[
        { label: 'Applicant', render: r => r.applicant_name || r.loan_id },
        { label: 'RM asked', render: r => pct(r.requested_rate) },
        { label: 'Decision', render: r => r.status === 'approved' ? <span className="green-text"><CheckCircle size={14}/> Approved at {pct(r.approved_rate)}</span> : <span><XCircle size={14}/> Declined</span> },
        { label: 'Note', key: 'decision_note' },
        { label: 'When', render: r => when(r.decided_at) }]}
        rows={decided.slice(0, 20).map(r => ({ ...r, id: r.request_id }))}/></>}
    <div className="actions end"><Button onClick={load}><ArrowClockwise/> Refresh</Button></div>
  </div>;
}

// Wireframe 06 header: what the RM asked, what it costs, what it buys.
export function RequestSummary({ request, limits }) {
  const x = request.analysis || {};
  const floor = Math.max(request.recommended_rate - (limits?.approve_below_recommendation_bps ?? 40) / 100, 0);
  return <section className="mgr-request">
    <h2>{request.requested_by_name || request.requested_by} asks {pct(request.requested_rate)}, the system recommends {pct(request.recommended_rate)}.
      {x.cost_per_year != null && <> Approving gives up about {money(x.cost_per_year)} of interest a year.</>}</h2>
    <div className="mgr-tiles">
      <div className="cost"><small>Interest given up over the loan’s life</small><b>{x.cost_over_life != null ? `≈ ${money(x.cost_over_life)}` : '—'}</b><em>{x.expected_life_years ? `Expected life about ${x.expected_life_years} years` : ''}</em></div>
      <div><small>Acceptance</small><b>{share(x.acceptance_at_recommended)} → {share(x.acceptance_at_requested)}</b><em>If approved at {pct(request.requested_rate)}</em></div>
      <div><small>Customer likely to accept up to</small><b>{x.headroom_rate ? pct(x.headroom_rate) : '—'}</b><em>{x.headroom_capped_by === 'quoted_range' ? 'Top of the range quoted to them' : '7 in 10 similar customers accept'}</em></div>
      <div><small>Your limit</small><b>Down to {pct(floor)}</b><em>Below that: credit head</em></div>
    </div>
    <div className="mgr-reason"><small>RM’s reason</small><span>{titleCase(request.reason)}{request.note ? ` · “${request.note}”` : ''}</span>
      {request.reason === 'competitor_lower_rate' && !request.evidence_verified && <small className="warn">Competitor quote not yet evidenced</small>}</div>
  </section>;
}

// Wireframe 06 "Your decision". The rate follows the scenario slider above it.
export function ManagerDecision({ request, rate, setRate, reasons, onDecided, notify }) {
  const [reason, setReason] = useState(request.reason);
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  async function decide(approve) {
    if (!note.trim()) { setError('Add a note to your decision.'); return; }
    setBusy(true); setError('');
    try {
      const r = await api(`/pricing/rate-requests/${request.request_id}/decide`, { approve, rate: approve ? Number(rate) : null, reason: approve ? reason : null, note });
      notify(approve ? `Approved at ${pct(r.approved_rate)}. The offer has been sent.` : `Declined. ${request.requested_by_name || request.requested_by} keeps ${pct(request.recommended_rate)}.`);
      onDecided(r);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }
  return <section className="panel mgr-decision"><div className="panel-head"><h2><Gavel size={21}/> Your decision</h2></div>
    <div className="po-form">
      <label><span>Approved rate (%)</span><input type="number" step="0.01" value={rate} onChange={e => setRate(e.target.value)}/></label>
      <label><span>Reason (required)</span><select value={reason} onChange={e => setReason(e.target.value)}>{reasons.map(r => <option key={r} value={r}>{titleCase(r)}</option>)}</select></label>
      <label><span>Note (required)</span><input value={note} placeholder="e.g. Approve subject to sanction letter before disbursement" onChange={e => setNote(e.target.value)}/></label>
    </div>
    {error && <p className="po-error" role="alert">{error}</p>}
    <div className="po-actions">
      <Button primary busy={busy} onClick={() => decide(true)}><CheckCircle size={17}/> Approve at {pct(rate)}</Button>
      <Button busy={busy} onClick={() => decide(false)}>Decline, keep {pct(request.recommended_rate)}</Button>
      <small>The rate follows the scenario slider. Approving records your override and sends the offer.</small>
    </div>
  </section>;
}
