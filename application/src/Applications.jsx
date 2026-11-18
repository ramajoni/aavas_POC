import React, { useEffect, useState } from 'react';
import { ArrowClockwise, ChatCircleText, FileText } from '@phosphor-icons/react';
import { api, titleCase } from './api';
import { Button, Empty, InfoBox, Modal } from './components';
import { ApplicationDocuments } from './Documents';
import { MyMonth } from './Disbursements';
import { AgingCell, NextAction, StatusBoard, StatusCell, UpsellCell, useStatusBoard } from './Status';
import './applications.css';

const lakh = n => n == null ? '—' : `₹${(n / 100000).toFixed(1)} L`;
const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const day = iso => iso ? new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }) : '—';
const daysSince = iso => iso ? Math.max(Math.floor((Date.now() - new Date(iso).getTime()) / 86400000), 0) : null;
function ago(days, lateFrom = 2) {
  if (days == null) return { text: '—', late: false };
  return { text: days === 0 ? 'Today' : days === 1 ? '1 day' : `${days} days`, late: days >= lateFrom };
}
const STALLED = { documents_pending: 'Documents pending', quote_not_applied: 'Quote taken, not applied', dropped_off: 'Dropped off' };
const CHANNELS = { whatsapp: 'WhatsApp', sms: 'SMS' };

// Remind a customer whose file has stalled. The message is shown before it is
// recorded; this service has no SMS or WhatsApp gateway, so nothing is delivered.
function Nudge({ item, channels, onClose, onSent, notify }) {
  const [channel, setChannel] = useState(channels[0] || 'sms');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function send() {
    setBusy(true); setError('');
    try {
      await api('/applications/nudges', { ref: item.ref, channel });
      notify(`Nudge recorded for ${item.name} (${CHANNELS[channel] || channel}). Delivery needs a messaging gateway, which this demo does not have.`);
      onSent();
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }
  return <Modal title={`Nudge ${item.name}`} onClose={onClose}>
    <p className="muted">{STALLED[item.stage]} · {item.detail}</p>
    <div className="nudge-channels" role="radiogroup" aria-label="Channel">{channels.map(c =>
      <label key={c}><input type="radio" name="channel" checked={channel === c} onChange={() => setChannel(c)}/> {CHANNELS[c] || titleCase(c)}</label>)}</div>
    <blockquote className="nudge-message">{item.preview}</blockquote>
    {item.last_nudge && <p className="small-copy">Last nudged {day(item.last_nudge.sent_at)} by {CHANNELS[item.last_nudge.channel] || item.last_nudge.channel}.</p>}
    {error && <p className="form-error" role="alert">{error}</p>}
    <div className="modal-actions"><Button onClick={onClose}>Cancel</Button><Button primary busy={busy} onClick={send}><ChatCircleText/> Send nudge</Button></div>
  </Modal>;
}

// Wireframe 03a, "RM · My applications": the month against target, then every
// file in exactly one stage - read from the server, so a file the RM sends to the
// manager leaves "Ready for final offer" for "Pending rate approval" at once.
export function Applications({ user, onOpen, notify }) {
  const [rows, setRows] = useState(null);
  const [progress, setProgress] = useState(null);
  const [offers, setOffers] = useState([]);
  const [requests, setRequests] = useState([]);
  const [tab, setTab] = useState('ready');
  const board = useStatusBoard();
  // Each file's status, next action, aging and upsell view, for the columns below.
  const st = Object.fromEntries((board.rows || []).map(r => [r.loan_id, r]));
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [docsFor, setDocsFor] = useState(null);
  const [nudgeFor, setNudgeFor] = useState(null);

  function load() {
    setBusy(true); setError('');
    Promise.all([api('/applications'), api('/applications/in-progress').catch(() => null),
                 api('/pricing/offers').catch(() => ({ offers: [] })),
                 api('/pricing/rate-requests?status=pending').catch(() => ({ requests: [] }))])
      .then(([a, p, o, r]) => { setRows(a.applications); setProgress(p); setOffers(o.offers); setRequests(r.requests); })
      .catch(e => setError(e.message)).finally(() => setBusy(false));
  }
  useEffect(load, []);

  async function open(loanId) {
    try { const detail = await api(`/applications/${encodeURIComponent(loanId)}`); onOpen(detail.applicant); }
    catch (e) { notify(e.message, 'error'); }
  }

  const all = rows || [];
  const byLoan = Object.fromEntries(all.map(r => [r.loan_id, r]));
  const inStage = s => all.filter(r => r.stage === s);
  const ready = inStage('ready');
  const pending = inStage('pending_rate_approval');
  const sentLoans = new Set(inStage('offer_sent').map(r => r.loan_id));
  const sent = offers.filter(o => o.status === 'sent' && (sentLoans.has(o.loan_id) || !byLoan[o.loan_id]));
  const stalled = progress?.items || [];
  const requestFor = loanId => requests.find(r => r.loan_id === loanId);
  // Sample leads have no stored file: their next move follows from where they stopped.
  const sampleNext = i => i.stage === 'dropped_off' ? { owner: 'RM', text: `Call the customer: they stopped at ${String(i.stopped_at || 'the application').toLowerCase()}.` }
    : i.stage === 'quote_not_applied' ? { owner: 'Customer', text: 'Apply on the quote while it is valid; the RM can nudge them.' }
    : { owner: 'Customer', text: `Upload the ${i.detail ? i.detail.replace(/^Waiting for /, '') : 'documents'}.` };
  // Where every file stands, with its next possible action: a view of this page, not a menu of its own.
  const tabs = [['progress', 'In progress', stalled.length], ['ready', 'Ready for final offer', ready.length],
                ['mgr', 'Pending rate approval', pending.length], ['sent', 'Offer sent, awaiting customer', sent.length],
                ['status', 'All applications · status', board.rows ? board.rows.length : null]];
  const name = loanId => byLoan[loanId]?.applicant_name || loanId;
  // An assisted application this RM entered: they upload what the customer hands over.
  const mine = r => r.source === 'assisted' && r.created_by === user.username;
  const sourceLine = r => `${r.loan_id}${r.city ? ` · ${r.city}` : ''} · ${r.source === 'assisted' ? `Entered by ${r.created_by === user.username ? 'you' : r.created_by}` : 'Applied online'}${r.customer_login ? ` · login ${r.customer_login}` : ''}`;

  return <div className="operations my-applications">
    <div className="page-intro"><p className="eyebrow">{user.name} · {user.title}</p><h1>My applications</h1></div>
    <MyMonth/>
    <div className="app-kpis five" role="tablist">{tabs.map(([k, label, n]) =>
      <button key={k} type="button" role="tab" aria-selected={tab === k} className={`app-kpi tab${tab === k ? ' primary' : ''}`} onClick={() => setTab(k)}>
        <small>{label}</small><b>{(k === 'status' ? board.rows : rows) && n != null ? n : '—'}</b></button>)}</div>
    {tab === 'status' && <StatusBoard embedded board={board} role={user.role} onOpenFile={open} onOpenRequest={() => {}}/>}
    {error && <InfoBox warning><span role="alert">{error}</span></InfoBox>}

    {tab === 'progress' && rows && (stalled.length ? <><div className="app-table"><table>
      <thead><tr><th>Applicant</th><th>Loan asked</th><th>Where it stopped</th><th>Status · next action</th><th>Aging</th><th>Upsell</th><th>Last nudged</th><th>Action</th></tr></thead>
      <tbody>{stalled.map(i => { const w = ago(i.days_since_activity, 3); return <tr key={i.ref}>
        <td><b>{i.name}{i.sample && <span className="sample-tag">Sample</span>}</b>
          <small>{i.sample ? `${i.ref}${i.city ? ` · ${i.city}` : ''} · ${i.mobile}` : sourceLine({ ...i, loan_id: i.ref })}</small></td>
        <td>{lakh(i.loan_amount)}</td>
        <td><b>{STALLED[i.stage] || titleCase(i.stage)}</b><small>{i.detail}</small></td>
        <td>{i.sample ? <NextAction compact next={sampleNext(i)}/> : <StatusCell row={st[i.ref]}/>}</td>
        <td>{i.sample || !st[i.ref] ? <span className={w.late ? 'late' : ''}>{w.text}<small>since the last activity</small></span> : <AgingCell aging={st[i.ref].aging} late={3}/>}</td>
        <td>{i.sample ? (i.quoted ? <UpsellCell upsell={{ kind: 'quote', rate: i.quoted.max_rate, text: `Quoted ${pct(i.quoted.min_rate)}–${pct(i.quoted.max_rate)}: aim for the top` }}/> : <span className="muted">—</span>) : <UpsellCell upsell={st[i.ref]?.upsell}/>}</td>
        <td>{i.last_nudge ? <>{day(i.last_nudge.sent_at)}<small>{CHANNELS[i.last_nudge.channel] || i.last_nudge.channel}{i.nudges > 1 ? ` · ${i.nudges} so far` : ''}</small></> : <span className="muted">Not yet</span>}</td>
        <td className="row-actions"><button type="button" className="link-button" onClick={() => setNudgeFor(i)}>Nudge customer</button>
          {!i.sample && byLoan[i.ref] && (mine(byLoan[i.ref]) ? <button type="button" className="link-button" onClick={() => setDocsFor(byLoan[i.ref])}>Upload</button>
            : i.documents ? <button type="button" className="link-button" onClick={() => setDocsFor(byLoan[i.ref])}>Documents</button> : null)}
          {!i.sample && byLoan[i.ref] && <button type="button" className="link-button" onClick={() => open(i.ref)}>Open</button>}</td>
      </tr>; })}</tbody></table></div>
      <p className="small-copy">Files reach “Ready for final offer” once the {progress.required_text} are in. {progress.note} Sample rows stand in for the rest of a branch pipeline.</p></>
      : <Empty icon={FileText} title="Nothing stalled">Files waiting on documents, and customers who stopped part-way, appear here.</Empty>)}

    {tab === 'ready' && rows && (ready.length ? <div className="app-table"><table>
      <thead><tr><th>Applicant</th><th>Sanctioned amount</th><th>Documents</th><th>Suggested offer</th><th>Upsell</th><th>Status · next action</th><th>Aging</th><th>Action</th></tr></thead>
      <tbody>{ready.map(r => { const w = ago(daysSince(r.verified_at)); const s = r.suggested; return <tr key={r.loan_id}>
        <td><b>{r.applicant_name}</b><small>{sourceLine(r)}</small></td>
        <td>{lakh(s?.sanctioned_amount ?? r.loan_amount)}{s && !s.amount_verified && <small>as requested</small>}</td>
        <td><button type="button" className="link-button doc-ok" onClick={() => setDocsFor(r)}>Verified ({r.documents})</button></td>
        <td>{r.readiness && !r.readiness.done?.risk ? <span className="muted">{r.readiness.next_label}: {r.readiness.due} to fill</span>
          : s?.rate != null ? <span className="suggested"><b>{pct(s.rate)}</b>{s.acceptance != null ? <> · {Math.round(s.acceptance * 100)}% likely to accept</> : <small>Acceptance after the customer interaction is recorded</small>}</span>
          : <span className="muted">Open to price</span>}
          {r.manager_declined && <small>Manager declined {pct(r.manager_declined.requested_rate)}</small>}</td>
        <td><UpsellCell upsell={st[r.loan_id]?.upsell}/></td>
        <td><StatusCell row={st[r.loan_id]}/></td>
        <td>{st[r.loan_id] ? <AgingCell aging={st[r.loan_id].aging}/> : <span className={w.late ? 'late' : ''}>{w.text}</span>}</td>
        <td><button type="button" className="link-button" onClick={() => open(r.loan_id)}>{r.readiness && !r.readiness.complete ? 'Complete file' : 'Open'}</button></td></tr>; })}</tbody></table></div>
      : <Empty icon={FileText} title="Nothing ready">A file lands here once its documents are in. Enter the verified figures and record the customer interaction to get the rate the engine suggests and send it.</Empty>)}

    {tab === 'mgr' && (pending.length ? <div className="app-table"><table>
      <thead><tr><th>Applicant</th><th>Sanctioned amount</th><th>Recommended</th><th>Requested</th><th>Upsell</th><th>Reason</th><th>Next action</th><th>Aging</th></tr></thead>
      <tbody>{pending.map(f => { const r = requestFor(f.loan_id); return <tr key={f.loan_id}>
        <td><b>{f.applicant_name || f.loan_id}</b><small>{f.loan_id}</small></td><td>{lakh(r?.loan_amount ?? f.loan_amount)}</td>
        <td>{pct(r?.recommended_rate)}</td><td>{pct(r?.requested_rate)}</td><td><UpsellCell upsell={st[f.loan_id]?.upsell}/></td>
        <td>{r ? titleCase(r.reason) : <span className="muted">Asked by another RM</span>}</td>
        <td><StatusCell row={st[f.loan_id]}/></td>
        <td>{st[f.loan_id] ? <AgingCell aging={st[f.loan_id].aging}/> : r ? ago(daysSince(r.requested_at)).text : '—'}</td></tr>; })}</tbody></table></div>
      : <Empty icon={FileText} title="Nothing pending">Files you send to the area manager for a lower rate wait here until the manager decides.</Empty>)}

    {tab === 'sent' && (sent.length ? <div className="app-table"><table>
      <thead><tr><th>Applicant</th><th>Sanctioned amount</th><th>Offered rate</th><th>Upsell</th><th>Route</th><th>Next action</th><th>Aging</th><th>Action</th></tr></thead>
      <tbody>{sent.map(o => <tr key={o.offer_id}>
        <td><b>{name(o.loan_id)}</b><small>{o.loan_id}</small></td><td>{lakh(byLoan[o.loan_id]?.loan_amount)}</td>
        <td>{pct(o.rate)}<small>sent {day(o.sent_at)}</small></td><td><UpsellCell upsell={st[o.loan_id]?.upsell}/></td>
        <td>{o.route === 'manager_approved' ? 'Approved by manager' : o.route === 'manager_uplifted' ? 'Uplifted by manager' : o.route === 'with_reason' ? `Below recommendation · ${titleCase(o.reason)}` : 'At or above recommendation'}</td>
        <td>{st[o.loan_id] ? <StatusCell row={st[o.loan_id]}/> : <>Awaiting<small>The customer accepts in their portal</small></>}</td>
        <td>{st[o.loan_id] ? <AgingCell aging={st[o.loan_id].aging} late={3}/> : '—'}</td>
        <td>{byLoan[o.loan_id] ? <button type="button" className="link-button" onClick={() => open(o.loan_id)}>Record decision</button> : '—'}</td></tr>)}</tbody></table></div>
      : <Empty icon={FileText} title="No offers waiting">Offers you send appear here until the customer decides.</Empty>)}

    {tab !== 'status' && <div className="actions end"><Button busy={busy} onClick={() => { load(); board.load(); }}><ArrowClockwise/> Refresh</Button></div>}
    {docsFor && <Modal wide title={`Documents · ${docsFor.applicant_name} · ${docsFor.loan_id}`} onClose={() => setDocsFor(null)}>
      <ApplicationDocuments loanId={docsFor.loan_id} notify={notify} canUpload={mine(docsFor)} onChange={load}/>
    </Modal>}
    {nudgeFor && <Nudge item={nudgeFor} channels={progress?.channels || ['sms']} notify={notify}
      onClose={() => setNudgeFor(null)} onSent={() => { setNudgeFor(null); load(); }}/>}
  </div>;
}
