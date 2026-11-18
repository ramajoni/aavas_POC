import React, { useEffect, useMemo, useState } from 'react';
import { ArrowClockwise, CaretDown, CheckCircle, Circle, CircleNotch, MagnifyingGlass, MinusCircle, WarningCircle } from '@phosphor-icons/react';
import { api } from './api';
import { Badge, Button, Empty, InfoBox } from './components';
import './status.css';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const lakh = n => n == null ? '—' : `₹${(n / 100000).toFixed(n % 100000 ? 1 : 0)} lakh`;
const day = iso => iso ? new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }) : null;
const ICON = { done: CheckCircle, current: Circle, warn: WarningCircle, skipped: MinusCircle, todo: Circle };
const OWNER_TONE = { RM: 'blue', 'Area manager': 'amber', Customer: 'green' };

// The next possible action on a file and whose move it is.
export function NextAction({ next, compact }) {
  if (!next) return null;
  return <div className={`st-next${compact ? ' compact' : ''}`}><span className={`st-owner ${OWNER_TONE[next.owner] || 'blue'}`}>{next.owner}</span>
    <span>{compact ? '' : <b>Next: </b>}{next.text}</span></div>;
}

// The steps of one file, left to right: what is done, what it waits on, what is next.
export function Timeline({ steps }) {
  return <ol className="st-steps">{steps.map((s, i) => {
    const Icon = ICON[s.state] || Circle;
    return <li key={s.key} className={s.state} aria-current={s.state === 'current' ? 'step' : undefined}>
      <span className="st-mark">{s.state === 'todo' ? i + 1 : <Icon size={16} weight={s.state === 'current' ? 'bold' : 'fill'}/>}</span>
      <b>{s.label}</b>
      {s.detail && <small>{s.detail}</small>}
      {s.at && <em>{day(s.at)}</em>}
    </li>; })}</ol>;
}

// Where one application stands, for the RM's file and the area manager's review.
// ``refresh`` changes whenever something on the file may have moved it on.
export function ApplicationStatus({ loanId, refresh }) {
  const [d, setD] = useState(null);
  useEffect(() => {
    let live = true;
    api(`/applications/${encodeURIComponent(loanId)}/status`).then(r => { if (live) setD(r); }).catch(() => { if (live) setD(false); });
    return () => { live = false; };
  }, [loanId, refresh]);
  if (d === false) return null;
  if (!d) return <section className="st-card"><p className="muted"><CircleNotch className="spin" size={14}/> Reading where this file stands…</p></section>;
  return <section className="st-card" aria-label="Application status">
    <div className="st-head"><h2>Application status</h2><Badge tone={d.status.tone}>{d.status.label}</Badge></div>
    <NextAction next={d.next}/>
    <Timeline steps={d.steps}/>
  </section>;
}

// Every application and where it stands, read once by the page that shows it (the
// RM's My Applications, the area manager's Approval requests) so its tile can count it.
export function useStatusBoard() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const load = () => { setBusy(true); setError(''); api('/applications/status-board').then(r => setRows(r.applications)).catch(e => setError(e.message)).finally(() => setBusy(false)); };
  useEffect(load, []);
  return { rows, error, busy, load };
}

// Every application and where it stands, with the prices so far; a row opens to
// its timeline. ``role`` decides which next actions get a button: the RM opens the
// file, the area manager reviews a request waiting for them.
export function StatusBoard({ board, role, onOpenFile, onOpenRequest, embedded }) {
  const { rows, error, busy, load } = board;
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(null);
  const groups = useMemo(() => {
    const out = {};
    for (const r of rows || []) out[r.status.label] = (out[r.status.label] || 0) + 1;
    return Object.entries(out).sort((x, y) => y[1] - x[1]);
  }, [rows]);
  const q = search.trim().toLowerCase();
  const manager = ['area_manager', 'administrator'].includes(role);
  const act = r => r.next?.kind === 'manager_review' && manager && r.next.request_id
    ? <button type="button" className="button primary st-go" onClick={() => onOpenRequest(r.next.request_id)}>Review</button>
    : role === 'relationship_manager' && r.next?.kind !== 'manager_review'
      ? <button type="button" className="button st-go" onClick={() => onOpenFile(r.loan_id)}>Open file</button> : null;
  const shown = (rows || []).filter(r => (filter === 'all' || r.status.label === filter)
    && (!q || `${r.applicant_name} ${r.loan_id} ${r.city || ''} ${r.entered_by || ''}`.toLowerCase().includes(q)));

  return <div className="st-board">
    {embedded ? <p className="small-copy">Every application and where it stands: the indicative range quoted, the risk-adjusted rate on the verified file,
        how likely the customer is to accept, and the next possible action.</p>
      : <div className="page-intro"><h1>Application status</h1>
      <p>Every application and where it stands: the indicative range quoted, the risk-adjusted rate on the verified file,
        how likely the customer is to accept, and the next possible action.</p></div>}
    {error && <InfoBox warning><span role="alert">{error}</span></InfoBox>}
    <div className="st-filters">
      <div className="st-chips" role="tablist" aria-label="Filter by status">
        <button role="tab" aria-selected={filter === 'all'} className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>All <small>{rows?.length ?? '—'}</small></button>
        {groups.map(([label, n]) => <button key={label} role="tab" aria-selected={filter === label} className={filter === label ? 'active' : ''} onClick={() => setFilter(label)}>{label} <small>{n}</small></button>)}
      </div>
      <label className="st-search"><MagnifyingGlass size={16}/><input aria-label="Search applications" placeholder="Customer, application ID, city" value={search} onChange={e => setSearch(e.target.value)}/></label>
      <Button busy={busy} onClick={load}><ArrowClockwise size={16}/> Refresh</Button>
    </div>
    {!rows && !error && <p className="muted"><CircleNotch className="spin" size={14}/> Reading every file…</p>}
    {rows && !shown.length && <Empty title="Nothing to show">{rows.length ? 'No file matches this filter.' : 'Applications appear here once they are submitted.'}</Empty>}
    {shown.length > 0 && <div className="app-table st-table"><table>
      <thead><tr><th>Customer</th><th>Entered by</th><th>Loan</th><th>Indicative range</th><th>Risk-adjusted rate</th><th>Acceptance</th><th>Status</th><th>Next possible action</th><th></th></tr></thead>
      <tbody>{shown.map(r => <React.Fragment key={r.loan_id}>
        <tr className={open === r.loan_id ? 'open' : ''}>
          <td><b>{r.applicant_name}</b><small>{r.loan_id}{r.city ? ` · ${r.city}` : ''}</small></td>
          <td>{r.entered_by || 'Applied online'}</td>
          <td>{lakh(r.loan_amount)}</td>
          <td>{r.indicative ? `${pct(r.indicative.min_rate)}–${pct(r.indicative.max_rate)}` : '—'}</td>
          <td>{r.risk_adjusted ? <><b>{pct(r.risk_adjusted.rate)}</b><small>risk grade {r.risk_adjusted.risk_grade || '—'}</small></> : <span className="muted">Not yet</span>}</td>
          <td>{r.acceptance != null ? `${Math.round(r.acceptance * 100)}%` : <span className="muted">Not yet</span>}</td>
          <td><Badge tone={r.status.tone}>{r.status.label}</Badge><small>Updated {day(r.updated_at)}</small></td>
          <td className="st-next-cell"><NextAction next={r.next} compact/></td>
          <td className="st-actions">{act(r)}<button type="button" className="link-button st-toggle" aria-expanded={open === r.loan_id} onClick={() => setOpen(open === r.loan_id ? null : r.loan_id)}>
            {open === r.loan_id ? 'Hide' : 'Steps'}<CaretDown size={14}/></button></td>
        </tr>
        {open === r.loan_id && <tr className="st-detail"><td colSpan="9"><Timeline steps={r.steps}/></td></tr>}
      </React.Fragment>)}</tbody></table></div>}
  </div>;
}

// The board on a page of its own (an old link to #/status still lands somewhere).
export function StatusPage(props) {
  const board = useStatusBoard();
  return <StatusBoard board={board} {...props}/>;
}

// Cells the worklists share: the status with the next action, how long the file has
// waited, and the file from the upsell side.
const daysText = n => n == null ? '—' : n === 0 ? 'today' : n === 1 ? '1 day' : `${n} days`;
export function StatusCell({ row }) {
  if (!row) return <span className="muted">—</span>;
  return <div className="st-cell"><Badge tone={row.status.tone}>{row.status.label}</Badge><NextAction next={row.next} compact/></div>;
}
export function AgingCell({ aging, late = 2 }) {
  if (!aging) return <span className="muted">—</span>;
  const slow = aging.days_in_step != null && aging.days_in_step >= late;
  return <div className={`st-aging${slow ? ' late' : ''}`}>
    <b>{aging.days_in_step === 0 ? 'Today' : daysText(aging.days_in_step)}</b><small>{aging.step}</small>
    <small>Open {aging.days_open === 0 ? 'since today' : daysText(aging.days_open)}</small></div>;
}
// Why there is no upsell room above the system recommendation, from the limit that bound it.
export const noRoom = capped => capped === 'quoted_range' ? 'The system recommendation is the top of the range quoted to them'
  : capped === 'above_quote' ? 'Already above the range quoted to them' : capped === 'policy' ? 'The system recommendation is the policy ceiling'
  : 'Fewer than 70% of customers like this accept even at the system recommendation';
export function UpsellCell({ upsell }) {
  if (!upsell) return <span className="muted">—</span>;
  const head = upsell.kind === 'hold' ? `+${upsell.bps} bps` : upsell.kind === 'room' ? `Up to ${Number(upsell.rate).toFixed(2)}%`
    : upsell.kind === 'sent' ? `${upsell.bps > 0 ? '+' : upsell.bps < 0 ? '−' : ''}${Math.abs(upsell.bps)} bps` : upsell.kind === 'quote' ? `Aim ${Number(upsell.rate).toFixed(2)}%` : 'No room';
  const tone = upsell.kind === 'none' || (upsell.kind === 'sent' && upsell.bps < 0) ? 'flat' : 'up';
  return <div className={`st-upsell ${tone}`}><b>{head}</b><small>{upsell.text}</small></div>;
}
