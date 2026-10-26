import React, { useEffect, useState } from 'react';
import { ArrowClockwise, ChatCircleText, FileText } from '@phosphor-icons/react';
import { api, titleCase } from './api';
import { Button, Empty, InfoBox, Modal } from './components';
import { ApplicationDocuments } from './Documents';
import { MyMonth } from './Disbursements';
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
  const tabs = [['progress', 'In progress', stalled.length], ['ready', 'Ready for final offer', ready.length],
                ['mgr', 'Pending rate approval', pending.length], ['sent', 'Offer sent, awaiting customer', sent.length]];
  const name = loanId => byLoan[loanId]?.applicant_name || loanId;
  // An assisted application this RM entered: they upload what the customer hands over.
  const mine = r => r.source === 'assisted' && r.created_by === user.username;
  const sourceLine = r => `${r.loan_id}${r.city ? ` · ${r.city}` : ''} · ${r.source === 'assisted' ? `Entered by ${r.created_by === user.username ? 'you' : r.created_by}` : 'Applied online'}${r.customer_login ? ` · login ${r.customer_login}` : ''}`;

  return <div className="operations my-applications">
    <div className="page-intro"><p className="eyebrow">{user.name} · {user.title}</p><h1>My applications</h1></div>
    <MyMonth/>
    <div className="app-kpis four" role="tablist">{tabs.map(([k, label, n]) =>
      <button key={k} type="button" role="tab" aria-selected={tab === k} className={`app-kpi tab${tab === k ? ' primary' : ''}`} onClick={() => setTab(k)}>
        <small>{label}</small><b>{rows ? n : '—'}</b></button>)}</div>
    {error && <InfoBox warning><span role="alert">{error}</span></InfoBox>}

    {tab === 'progress' && rows && (stalled.length ? <><div className="app-table"><table>
      <thead><tr><th>Applicant</th><th>Loan asked</th><th>Where it stopped</th><th>Last activity</th><th>Last nudged</th><th>Action</th></tr></thead>
      <tbody>{stalled.map(i => { const w = ago(i.days_since_activity, 3); return <tr key={i.ref}>
        <td><b>{i.name}{i.sample && <span className="sample-tag">Sample</span>}</b>
          <small>{i.sample ? `${i.ref}${i.city ? ` · ${i.city}` : ''} · ${i.mobile}` : sourceLine({ ...i, loan_id: i.ref })}</small></td>
        <td>{lakh(i.loan_amount)}</td>
        <td><b>{STALLED[i.stage] || titleCase(i.stage)}</b><small>{i.detail}</small></td>
        <td className={w.late ? 'late' : ''}>{w.text}</td>
        <td>{i.last_nudge ? <>{day(i.last_nudge.sent_at)}<small>{CHANNELS[i.last_nudge.channel] || i.last_nudge.channel}{i.nudges > 1 ? ` · ${i.nudges} so far` : ''}</small></> : <span className="muted">Not yet</span>}</td>
        <td className="row-actions"><button type="button" className="link-button" onClick={() => setNudgeFor(i)}>Nudge customer</button>
          {!i.sample && byLoan[i.ref] && (mine(byLoan[i.ref]) ? <button type="button" className="link-button" onClick={() => setDocsFor(byLoan[i.ref])}>Upload</button>
            : i.documents ? <button type="button" className="link-button" onClick={() => setDocsFor(byLoan[i.ref])}>Documents</button> : null)}</td>
      </tr>; })}</tbody></table></div>
      <p className="small-copy">Files reach “Ready for final offer” once the {progress.required_text} are in. {progress.note} Sample rows stand in for the rest of a branch pipeline.</p></>
      : <Empty icon={FileText} title="Nothing stalled">Files waiting on documents, and customers who stopped part-way, appear here.</Empty>)}

    {tab === 'ready' && rows && (ready.length ? <div className="app-table"><table>
      <thead><tr><th>Applicant</th><th>Sanctioned amount</th><th>Documents</th><th>Suggested offer</th><th>Days since verification</th><th>Action</th></tr></thead>
      <tbody>{ready.map(r => { const w = ago(daysSince(r.verified_at)); const s = r.suggested; return <tr key={r.loan_id}>
        <td><b>{r.applicant_name}</b><small>{sourceLine(r)}</small></td>
        <td>{lakh(s?.sanctioned_amount ?? r.loan_amount)}{s && !s.amount_verified && <small>as requested</small>}</td>
        <td><button type="button" className="link-button doc-ok" onClick={() => setDocsFor(r)}>Verified ({r.documents})</button></td>
        <td>{s?.rate != null ? <span className="suggested"><b>{pct(s.rate)}</b>{s.acceptance != null && <> · {Math.round(s.acceptance * 100)}% likely to accept</>}</span> : <span className="muted">Open to price</span>}
          {r.manager_declined && <small>Manager declined {pct(r.manager_declined.requested_rate)}</small>}</td>
        <td className={w.late ? 'late' : ''}>{w.text}</td>
        <td><button type="button" className="link-button" onClick={() => open(r.loan_id)}>Open</button></td></tr>; })}</tbody></table></div>
      : <Empty icon={FileText} title="Nothing ready">A file lands here once its documents are in and verified, with the rate the engine suggests.</Empty>)}

    {tab === 'mgr' && (pending.length ? <div className="app-table"><table>
      <thead><tr><th>Applicant</th><th>Sanctioned amount</th><th>Recommended</th><th>Requested</th><th>Reason</th><th>With</th><th>Days waiting</th></tr></thead>
      <tbody>{pending.map(f => { const r = requestFor(f.loan_id); return <tr key={f.loan_id}>
        <td><b>{f.applicant_name || f.loan_id}</b><small>{f.loan_id}</small></td><td>{lakh(r?.loan_amount ?? f.loan_amount)}</td>
        <td>{pct(r?.recommended_rate)}</td><td>{pct(r?.requested_rate)}</td><td>{r ? titleCase(r.reason) : <span className="muted">Asked by another RM</span>}</td><td>Area manager</td>
        <td>{r ? ago(daysSince(r.requested_at)).text : '—'}</td></tr>; })}</tbody></table></div>
      : <Empty icon={FileText} title="Nothing pending">Files you send to the area manager for a lower rate wait here until the manager decides.</Empty>)}

    {tab === 'sent' && (sent.length ? <div className="app-table"><table>
      <thead><tr><th>Applicant</th><th>Sanctioned amount</th><th>Offered rate</th><th>Sent</th><th>Route</th><th>Customer response</th><th>Action</th></tr></thead>
      <tbody>{sent.map(o => <tr key={o.offer_id}>
        <td><b>{name(o.loan_id)}</b><small>{o.loan_id}</small></td><td>{lakh(byLoan[o.loan_id]?.loan_amount)}</td>
        <td>{pct(o.rate)}</td><td>{day(o.sent_at)}</td>
        <td>{o.route === 'manager_approved' ? 'Approved by manager' : o.route === 'with_reason' ? `Below recommendation · ${titleCase(o.reason)}` : 'At or above recommendation'}</td>
        <td>Awaiting<small>The customer accepts in their portal</small></td>
        <td>{byLoan[o.loan_id] ? <button type="button" className="link-button" onClick={() => open(o.loan_id)}>Record decision</button> : '—'}</td></tr>)}</tbody></table></div>
      : <Empty icon={FileText} title="No offers waiting">Offers you send appear here until the customer decides.</Empty>)}

    <div className="actions end"><Button busy={busy} onClick={load}><ArrowClockwise/> Refresh</Button></div>
    {docsFor && <Modal wide title={`Documents · ${docsFor.applicant_name} · ${docsFor.loan_id}`} onClose={() => setDocsFor(null)}>
      <ApplicationDocuments loanId={docsFor.loan_id} notify={notify} canUpload={mine(docsFor)} onChange={load}/>
    </Modal>}
    {nudgeFor && <Nudge item={nudgeFor} channels={progress?.channels || ['sms']} notify={notify}
      onClose={() => setNudgeFor(null)} onSent={() => { setNudgeFor(null); load(); }}/>}
  </div>;
}
