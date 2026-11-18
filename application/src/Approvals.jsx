import React, { useEffect, useState } from 'react';
import { ArrowClockwise, CheckCircle, Gavel, TrendUp, XCircle } from '@phosphor-icons/react';
import { api, money, titleCase } from './api';
import { Button, DataTable, Empty, InfoBox } from './components';
import { AgingCell, noRoom, StatusBoard, StatusCell, useStatusBoard } from './Status';
import './applications.css';
import './price-offer.css';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const lakh = n => n == null ? '—' : `₹${(n / 100000).toFixed(1)} L`;
const crore = n => n == null ? '—' : n >= 1e7 ? `₹${(n / 1e7).toFixed(2)} Cr` : lakh(n);
const days = iso => Math.floor((Date.now() - new Date(iso).getTime()) / 86400000);
const waiting = iso => { const d = days(iso); return d <= 0 ? 'Today' : d === 1 ? '1 day' : `${d} days`; };
const when = iso => iso ? new Date(iso).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '';

// Wireframe 05, "Manager · Approval queue": lower rates the relationship managers
// asked for, sorted by the interest a year each would give up. The tiles filter the
// list; the row below them narrows it by branch or name.
export function Approvals({ user, onOpen, notify }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState('');
  const [view, setView] = useState('waiting');
  const [city, setCity] = useState('');
  const [query, setQuery] = useState('');
  const load = () => { setError(''); api('/pricing/rate-requests').then(r => setRows(r.requests)).catch(e => setError(e.message)); };
  useEffect(load, []);
  // Every application in the area and its next possible action: a view of this page.
  const board = useStatusBoard();
  const st = Object.fromEntries((board.rows || []).map(b => [b.loan_id, b]));

  const month = new Date().toISOString().slice(0, 7);
  const all = rows || [];
  const q = query.trim().toLowerCase();
  const matches = r => (!city || r.city === city)
    && (!q || `${r.applicant_name || ''} ${r.loan_id}`.toLowerCase().includes(q));
  const given = r => r.analysis?.cost_full_term ?? r.analysis?.cost_per_year ?? 0;
  const pending = all.filter(r => r.status === 'pending').sort((a, b) => given(b) - given(a));
  const decidedThisMonth = all.filter(r => r.status !== 'pending' && (r.decided_at || '').startsWith(month));
  const VIEWS = {
    waiting: { label: 'Waiting for you', rows: pending },
    late: { label: 'Waiting 2+ days', rows: pending.filter(r => days(r.requested_at) >= 2) },
    approved: { label: 'Approved or uplifted', rows: decidedThisMonth.filter(r => r.status === 'approved' || r.status === 'uplifted') },
    declined: { label: 'Declined this month', rows: decidedThisMonth.filter(r => r.status === 'declined') },
    status: { label: 'All applications · status', rows: board.rows || [] },
  };
  const statusView = view === 'status';
  const openById = id => { const r = all.find(x => x.request_id === id); if (r) onOpen(r); else api(`/pricing/rate-requests/${encodeURIComponent(id)}`).then(onOpen).catch(e => notify(e.message, 'error')); };
  const shown = VIEWS[view].rows.filter(matches);
  const sub = { waiting: crore(pending.reduce((t, r) => t + (r.loan_amount || 0), 0)) + ' of loans',
    late: pending.length ? `of ${pending.length} waiting` : 'none waiting',
    approved: `of ${decidedThisMonth.length} decided`, declined: `of ${decidedThisMonth.length} decided`,
    status: board.rows ? `${board.rows.filter(r => r.next?.owner === 'Area manager').length} need you` : '' };
  const options = key => [...new Set(all.map(r => r[key]).filter(Boolean))].sort();
  const filtered = city || q;
  const decidedView = view === 'approved' || view === 'declined';

  return <div className="operations my-applications approvals">
    <div className="page-intro"><p className="eyebrow">{user.name} · {user.title}</p><h1>Approval requests</h1></div>
    <div className="app-kpis five" role="tablist">{Object.entries(VIEWS).map(([key, v]) =>
      <button key={key} type="button" role="tab" aria-selected={view === key} className={`app-kpi tab${view === key ? ' primary' : ''}`} onClick={() => setView(key)}>
        <small>{v.label}</small><b>{(key === 'status' ? board.rows : rows) ? v.rows.length : '—'}</b><em>{rows ? sub[key] : ''}</em></button>)}</div>
    {statusView && <StatusBoard embedded board={board} role={user.role} onOpenRequest={openById}/>}
    {!statusView && <div className="mgr-filters" aria-label="Filter the requests">
      <label><span>Branch</span><select value={city} onChange={e => setCity(e.target.value)}><option value="">All branches</option>{options('city').map(c => <option key={c}>{c}</option>)}</select></label>
      <label className="grow"><span>Applicant or loan ID</span><input type="search" value={query} placeholder="Search" onChange={e => setQuery(e.target.value)}/></label>
      {filtered && <button type="button" className="link-button" onClick={() => { setCity(''); setQuery(''); }}>Clear filters</button>}
    </div>}
    {error && <InfoBox warning><span role="alert">{error}</span></InfoBox>}

    {!statusView && rows && !shown.length && <Empty icon={Gavel} title={filtered ? 'Nothing matches these filters' : view === 'waiting' ? 'Nothing waiting' : `Nothing ${VIEWS[view].label.toLowerCase()}`}>
      {filtered ? 'Clear the filters to see every request.' : 'Requests appear here when a relationship manager asks for a rate below their authority.'}</Empty>}
    {!statusView && !decidedView && shown.length > 0 && <div className="app-table"><table>
      <thead><tr><th>Customer</th><th>RM</th><th>Loan amount</th><th>Indicative range</th><th>RM’s ask → rate to hold</th><th className="key">Upsell room</th><th>Revenue at risk</th><th>Status · next action</th><th>Aging</th><th></th></tr></thead>
      <tbody>{shown.map(r => <tr key={r.request_id}>
        <td><b>{r.applicant_name || r.loan_id}</b><small>{r.loan_id}{r.city ? ` · ${r.city}` : ''}</small></td>
        <td>{r.requested_by_name || r.requested_by}</td><td>{lakh(r.loan_amount)}</td>
        <td>{r.analysis?.indicative_band ? `${pct(r.analysis.indicative_band.min_rate)}–${pct(r.analysis.indicative_band.max_rate)}` : '—'}<small>quoted at application</small></td>
        <td className="mgr-hold"><b>{pct(r.requested_rate)} → {pct(r.recommended_rate)}</b>
          <small className="up">+{Math.round((r.recommended_rate - r.requested_rate) * 100)} bps by holding the risk-adjusted rate</small></td>
        <td className="key">{!r.analysis ? '—' : r.analysis.headroom_rate != null && r.analysis.headroom_rate > r.recommended_rate + 0.005
          ? <>Accepts up to <b>{pct(r.analysis.headroom_rate)}</b><small>+{(r.analysis.headroom_rate - r.recommended_rate).toFixed(2)}% over the rate to hold</small></>
          : <><b>No room</b><small>{r.analysis.headroom_capped_by === 'quoted_range' ? 'the rate to hold is the top of the range quoted to them'
            : r.analysis.headroom_capped_by === 'above_quote' ? 'already above the range quoted to them'
            : 'fewer than 70% of customers like this accept even at the rate to hold'}</small></>}</td>
        <td>{money(r.analysis?.cost_full_term ?? r.analysis?.cost_per_year)}{r.analysis?.tenure_years && <small>over {r.analysis.tenure_years} years</small>}</td>
        <td><StatusCell row={st[r.loan_id]}/></td>
        <td>{st[r.loan_id] ? <AgingCell aging={st[r.loan_id].aging}/> : <span className={days(r.requested_at) >= 2 ? 'late' : ''}>{waiting(r.requested_at)}</span>}</td>
        <td className="mgr-actions"><button className="link-button" onClick={() => onOpen(r)}>Review</button></td>
      </tr>)}</tbody></table></div>}
    {decidedView && shown.length > 0 && <DataTable columns={[
        { label: 'Customer', render: r => <><b>{r.applicant_name || r.loan_id}</b><small className="mp-cell-sub">{r.loan_id}{r.city ? ` · ${r.city}` : ''}</small></> },
        { label: 'RM', render: r => r.requested_by_name || r.requested_by },
        { label: 'Loan amount', render: r => lakh(r.loan_amount) },
        { label: 'RM’s discounted rate', render: r => pct(r.requested_rate) },
        { label: 'Decision', render: r => r.status === 'uplifted' ? <span className="green-text"><TrendUp size={14}/> Uplifted to {pct(r.approved_rate)}</span> : r.status === 'approved' ? <span className="green-text"><CheckCircle size={14}/> Approved at {pct(r.approved_rate)}</span> : <span><XCircle size={14}/> Declined, kept {pct(r.recommended_rate)}</span> },
        { label: 'Note', key: 'decision_note' },
        { label: 'When', render: r => when(r.decided_at) }]}
        rows={shown.map(r => ({ ...r, id: r.request_id }))}/>}
    {!statusView && !decidedView && shown.length > 0 && <p className="small-copy">Sorted by revenue at risk: the interest the bank gives up over the whole loan if the discount is approved (the drop in EMI, over every EMI of the tenure). “Indicative range” is what the customer was quoted on their own figures; “Rate to hold” is the system’s risk-adjusted recommendation on the verified file. “Upsell room” is the highest rate at which 70% of similar customers still accept, never above the range the customer was quoted.</p>}
    <div className="actions end"><Button onClick={load}><ArrowClockwise/> Refresh</Button></div>
  </div>;
}

// Wireframe 06 header: what the RM asked, what it costs, what it buys.
// The request read from the upsell side: holding the model's rate instead of the
// RM's ask, what it adds to the rate, what it costs in acceptance, and what it earns.
export function RequestSummary({ request }) {
  const x = request.analysis || {};
  const ask = request.requested_rate, reco = request.recommended_rate;
  const upBps = Math.round((reco - ask) * 100);
  const roomUp = x.headroom_rate != null && x.headroom_rate - reco > 0.005;
  const atAsk = x.acceptance_at_requested, atReco = x.acceptance_at_recommended;
  const drop = atAsk != null && atReco != null ? Math.round((atAsk - atReco) * 100) : null;
  const dropText = drop == null ? '' : drop <= 0 ? 'with no drop in acceptance' : `with ${drop <= 3 ? 'only ' : ''}${drop} point${drop === 1 ? '' : 's'} lower acceptance`;
  const gain = x.cost_full_term;
  return <section className="mgr-request">
    <h2>Holding the system recommendation of {pct(reco)} instead of {request.requested_by_name || request.requested_by}’s {pct(ask)} is a {upBps} bps upsell {dropText}
      {gain != null ? <>: about {money(gain)} more interest over the full {x.tenure_years}-year loan.</> : '.'}</h2>
    <div className="mgr-tiles">
      <div className="upsell-table"><table>
        <thead><tr><th></th><th>Rate</th><th>Acceptance</th></tr></thead>
        <tbody>
          <tr><td>RM’s ask</td><td>{pct(ask)}</td><td>{share(atAsk)}</td></tr>
          <tr className="reco"><td>System recommendation</td><td>{pct(reco)}</td><td>{share(atReco)}</td></tr>
          <tr className="diff"><td>Difference</td><td>+{(reco - ask).toFixed(2)}%</td><td>{drop == null ? '—' : `${drop > 0 ? '−' : drop < 0 ? '+' : ''}${Math.abs(drop)} pts`}</td></tr>
        </tbody></table></div>
      <div className="upsell"><small>Upsell by holding the system recommendation</small><b>+{upBps} bps</b><em>{dropText ? dropText.charAt(0).toUpperCase() + dropText.slice(1) : ''}</em></div>
      <div className="earn"><small>Extra interest over the full loan</small><b>{money(gain ?? x.cost_over_life)}</b><em>{x.tenure_years && x.emi_recommended != null ? `${x.tenure_years * 12} EMIs of ${money(x.emi_recommended - x.emi_requested)} more` : ''}</em></div>
      <div><small>Further upsell room</small><b>{roomUp ? `Up to ${pct(x.headroom_rate)}` : 'No room'}</b><em>{roomUp ? `+${Math.round((x.headroom_rate - reco) * 100)} bps over the system recommendation; 70% of customers like this still accept${x.headroom_capped_by === 'quoted_range' ? ' (top of the range quoted to them)' : ''}` : noRoom(x.headroom_capped_by)}</em></div>
    </div>
    <div className="mgr-reason"><small>RM’s reason</small><span>{titleCase(request.reason)}{request.note ? ` · “${request.note}”` : ''}</span>
      {request.reason === 'competitor_lower_rate' && !request.evidence_verified && <small className="warn">Competitor quote not yet evidenced</small>}</div>
  </section>;
}

// Wireframe 06 "Your decision". The rate follows the scenario slider above it.
export function ManagerDecision({ request, rate, setRate, reasons, upliftReasons = [], onDecided, notify, side, preset }) {
  const [reason, setReason] = useState(request.reason);
  const [upReason, setUpReason] = useState(upliftReasons[0] || '');
  useEffect(() => { if (!upReason && upliftReasons.length) setUpReason(upliftReasons[0]); }, [upliftReasons.length]);
  const [note, setNote] = useState('');
  // A suggestion the manager chose to use fills the form; they still decide.
  // Holding the model rate is a decline: the rate is left as asked and Decline leads.
  useEffect(() => { if (preset) { if (preset.action !== 'hold') setRate(preset.rate); setNote(preset.note);
    if (preset.reason) (preset.action === 'uplift' ? setUpReason : setReason)(preset.reason); } }, [preset]);
  // A request below the lowest rate policy allows: say what can still be approved.
  const an = request.analysis || {};
  const rec = request.recommended_rate;
  // Price uplift: decline the discount and price above the recommendation, up to the
  // rate customers like this still accept, never above the range they were quoted.
  const room = an.headroom_rate != null && an.headroom_rate > rec + 0.005 ? an.headroom_rate : null;
  const isUplift = Number(rate) > rec + 1e-9;
  const accept = useAcceptanceAt(request.loan_id, isUplift ? Number(rate) : null);
  const floorNote = an.nothing_approvable
    ? `${pct(rec)} is already the lowest rate policy allows for this file${an.card_rate != null ? ` (card rate ${pct(an.card_rate)} less the largest discount policy allows)` : ''}: nothing below it can be approved. Decline to keep it${room ? ', or uplift the price' : ''}.`
    : an.below_policy_floor ? `Asked ${pct(request.requested_rate)}, below ${pct(an.policy_floor)}, the lowest rate policy allows for this file. You can approve ${pct(an.manager_floor)} or above, or decline.` : null;
  const declineLeads = !isUplift && (preset?.action === 'hold' || Boolean(an.nothing_approvable));
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  async function decide(approve) {
    if (!note.trim()) { setError('Add a note to your decision.'); return; }
    setBusy(true); setError('');
    try {
      const uplift = approve && isUplift;
      const r = await api(`/pricing/rate-requests/${request.request_id}/decide`, { approve, uplift, rate: approve ? Number(rate) : null,
        reason: approve ? (uplift ? upReason : reason) : null, note });
      notify(uplift ? `Uplifted to ${pct(r.approved_rate)}. The offer has been sent.` : approve ? `Approved at ${pct(r.approved_rate)}. The offer has been sent.`
        : `Declined. ${request.requested_by_name || request.requested_by} keeps ${pct(rec)}.`);
      onDecided(r);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }
  const showRate = !an.nothing_approvable || room;
  return <section id="mgr-decision" className={`panel mgr-decision${side ? ' side' : ''}${preset ? ' preset' : ''}`}><div className="panel-head"><h2>{!side && <Gavel size={21}/>} Your decision</h2></div>
    {floorNote && <InfoBox warning>{floorNote}</InfoBox>}
    {room && <div className={`mgr-uplift${isUplift ? ' on' : ''}`}><TrendUp size={16}/><span>Price uplift: customers like this still accept up to <b>{pct(room)}</b>{an.headroom_capped_by === 'quoted_range' ? ', the top of the range quoted to them' : ''}.</span>
      {!isUplift && <button type="button" className="link-button" onClick={() => setRate(String(room))}>Uplift to {pct(room)}</button>}</div>}
    <div className="po-form">
      {showRate && <label><span>{isUplift ? 'Uplifted rate (%)' : 'Approved rate (%)'}</span><input type="number" step="0.01" value={rate} onChange={e => setRate(e.target.value)}/>
        {isUplift && accept != null && <small className="mgr-accept">{Math.round(accept * 100)}% likely to accept at {pct(rate)}</small>}</label>}
      <label><span>{isUplift ? 'Reason for the uplift' : 'Reason (required)'}</span>{isUplift
        ? <select value={upReason} onChange={e => setUpReason(e.target.value)}>{upliftReasons.map(r => <option key={r} value={r}>{titleCase(r)}</option>)}</select>
        : <select value={reason} onChange={e => setReason(e.target.value)}>{reasons.map(r => <option key={r} value={r}>{titleCase(r)}</option>)}</select>}</label>
      <label><span>Note (required)</span>{side
        ? <textarea rows="3" value={note} placeholder={isUplift ? 'e.g. Customer accepts the quoted rate; no discount needed.' : 'e.g. Approve subject to sanction letter before disbursement.'} onChange={e => setNote(e.target.value)}/>
        : <input value={note} placeholder="e.g. Approve subject to sanction letter before disbursement" onChange={e => setNote(e.target.value)}/>}</label>
    </div>
    {error && <p className="po-error" role="alert">{error}</p>}
    <div className="po-actions">
      {declineLeads && <Button primary busy={busy} onClick={() => decide(false)}>Decline, keep {pct(rec)}</Button>}
      {isUplift ? <Button primary busy={busy} onClick={() => decide(true)}><TrendUp size={17}/> Uplift to {pct(rate)}</Button>
        : !an.nothing_approvable && <Button primary={!declineLeads} busy={busy} onClick={() => decide(true)}><CheckCircle size={17}/> Approve at {pct(rate)}</Button>}
      {!declineLeads && <Button busy={busy} onClick={() => decide(false)}>Decline, keep {pct(rec)}</Button>}
      <small>{isUplift ? 'Uplifting declines the discount and sends the offer at the higher rate; it is recorded as a premium override.'
        : side ? 'Approving records your override and sends the offer to the customer.' : 'The rate follows the scenario slider. Approving records your override and sends the offer.'}</small>
    </div>
  </section>;
}

// How likely the customer is to accept at one rate, for the uplift preview.
function useAcceptanceAt(loanId, rate) {
  const [p, setP] = useState(null);
  useEffect(() => {
    setP(null);
    if (!(rate > 0 && rate < 30)) return undefined;
    const t = setTimeout(() => api(`/pricing/offers/acceptance?loan_id=${encodeURIComponent(loanId)}&rates=${rate.toFixed(2)}`)
      .then(r => setP(r.acceptance[0]?.acceptance ?? null)).catch(() => {}), 300);
    return () => clearTimeout(t);
  }, [loanId, rate]);
  return p;
}
