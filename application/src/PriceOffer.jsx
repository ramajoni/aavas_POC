import React, { useEffect, useRef, useState } from 'react';
import { CheckCircle, CircleNotch, FileText, PaperPlaneTilt, PencilSimple, UserSwitch } from '@phosphor-icons/react';
import { api, emi, hasFile, money, percent, titleCase } from './api';
import { Button, InfoBox, Modal } from './components';
import { ApplicationDocuments } from './Documents';
import { CompetitorCheck } from './CompetitorCheck';
import { CustomerLoginControl } from './CustomerLogin';
import './price-offer.css';

const lakh = n => n == null ? '—' : `₹${(n / 100000).toFixed(n % 100000 ? 1 : 0)} lakh`;
const pct = r => `${Number(r).toFixed(2)}%`;
const when = iso => iso ? new Date(iso).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '';
const SEGMENTS = { salaried_formal: 'Salaried', salaried_informal: 'Salaried (informal)', self_employed_professional: 'Self-employed professional', self_employed_non_professional: 'Self-employed' };
const CHANNELS = { branch_walkin: 'Branch walk-in', dsa: 'DSA', connector: 'Connector', digital: 'Digital' };
const PROPERTY = { resale_flat: 'Resale flat', builder_flat: 'Builder flat', self_construction: 'Self construction', plot_plus_construction: 'Plot + construction' };
const STATUS = { in_progress: 'In progress · documents due', ready: 'Verified · ready for final offer', pending_rate_approval: 'Pending rate approval', offer_sent: 'Offer sent · awaiting customer', accepted: 'Accepted by the customer', not_accepted: 'Not taken by the customer' };
const yesNo = v => v == null ? '—' : v ? 'Yes' : 'No';
const given = v => v !== '' && v != null;
const check = flag => !given(flag) ? 'Not checked yet' : Number(flag) ? 'Issue flagged' : 'Clear';
// A figure only the assessment can supply: say so when it has not been entered.
const verified = (v, fmt = x => x) => given(v) ? fmt(v) : 'Not verified yet';
const shown = v => v == null || v === '' ? '—' : v;

// Where the file stands, in the order the bank works it: the loan is sanctioned
// on verified documents, then the rate goes to the customer as the final offer.
const STEPS = [['application', 'Application'], ['indicative', 'Indicative range'], ['documents', 'Documents'],
  ['sanction', 'Verification & sanction'], ['offer', 'Final offer'], ['decision', 'Customer decision']];
const STAGE_AT = { in_progress: 2, ready: 4, pending_rate_approval: 4, offer_sent: 5, accepted: 6, not_accepted: 6 };
function FileStages({ file }) {
  if (!file?.stage) return null;
  const at = STAGE_AT[file.stage] ?? 0;
  const note = { documents: file.stage === 'in_progress' && file.missing_documents?.length ? `${file.missing_documents.length} still due` : null,
    offer: file.stage === 'pending_rate_approval' ? 'With the area manager' : file.stage === 'ready' ? 'Ready to send' : null,
    decision: file.stage === 'offer_sent' ? 'Waiting for the customer' : file.stage === 'accepted' ? 'Accepted' : file.stage === 'not_accepted' ? 'Not taken' : null,
    indicative: file.indicative ? `${pct(file.indicative.min_rate)}–${pct(file.indicative.max_rate)}` : null };
  return <ol className="po-stages" aria-label="Where this file stands">{STEPS.map(([k, label], i) =>
    <li key={k} className={i < at ? 'done' : i === at ? 'current' : ''} aria-current={i === at ? 'step' : undefined}>
      <span>{i < at ? <CheckCircle size={15} weight="fill"/> : i + 1}</span><b>{label}</b>{note[k] && <small>{note[k]}</small>}</li>)}</ol>;
}

// Wireframe 04, "RM · Price and send offer". The relationship manager sees the
// price, its band and how customers respond - not the analysis behind it, which
// the server does not send to this role.
export function PriceOffer({ state, setState, runFinal, busy, notify, navigate, onEdit, user }) {
  const { applicant: a, pre, final } = state;
  const tried = useRef(null); const key = JSON.stringify(a);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!hasFile(a) || final || busy || tried.current === key) return;
    tried.current = key; setFailed(false);
    runFinal({ quiet: true }).then(ok => { if (!ok) setFailed(true); });
  }, [key, final, busy]);

  const [reasons, setReasons] = useState([]);
  useEffect(() => { api('/pricing/override/reasons').then(r => setReasons(r.allowed_reasons || [])).catch(() => {}); }, []);
  const [history, setHistory] = useState({ offers: [], requests: [] });
  const loadHistory = () => Promise.all([
    api(`/pricing/offers?loan_id=${encodeURIComponent(a.loan_id)}`).catch(() => ({ offers: [] })),
    api(`/pricing/rate-requests?loan_id=${encodeURIComponent(a.loan_id)}`).catch(() => ({ requests: [] })),
  ]).then(([o, r]) => setHistory({ offers: o.offers, requests: r.requests }));
  useEffect(() => { loadHistory(); }, [a.loan_id]);
  // The stored file: status, who entered it and its documents.
  const [file, setFile] = useState(undefined);
  // The list first: a file never submitted would be an expected 404, which the
  // browser logs as an error.
  const loadFile = () => api('/applications')
    .then(({ applications }) => applications.some(x => x.loan_id === a.loan_id) ? api(`/applications/${encodeURIComponent(a.loan_id)}`) : null)
    .then(setFile).catch(() => setFile(null));
  useEffect(() => { if (hasFile(a)) loadFile(); }, [a.loan_id]);

  const rec = final?.recommendation?.recommended_rate;
  const [rate, setRate] = useState('');
  const [reason, setReason] = useState('');
  const [note, setNote] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [check, setCheck] = useState(null);
  useEffect(() => { if (rec != null) setRate(String(rec)); }, [rec]);

  if (!hasFile(a)) return <div className="price-offer"><div className="po-head"><h1>Price and send offer</h1></div>
    <section className="po-card po-pending"><p>No file is open. Open one from My applications, or enter a new one.</p>
      <div className="po-actions"><Button primary onClick={() => navigate('applications')}>My applications</Button><Button onClick={() => navigate('new-application')}>New application</Button></div></section></div>;

  const header = <div className="po-head">
    <span className="po-crumbs"><button className="link-button" onClick={() => navigate('applications')}>My applications</button> / Price and send offer</span>
    <h1>{a.name} · {lakh(a.sanctioned_loan_amount || a.requested_loan_amount)} home loan</h1>
    <span className="po-meta">{[a.loan_id, SEGMENTS[a.segment] || titleCase(a.segment), a.city, titleCase(a.property_type),
      `${a.tenure_years || a.requested_tenure_years} years`,
      a.estimated_property_value ? `LTV ${Math.round(((a.sanctioned_loan_amount || a.requested_loan_amount) / (a.appraised_property_value || a.estimated_property_value)) * 100)}%` : null,
      a.bureau_score > 0 ? `Credit score ${a.bureau_score}` : 'New to credit'].filter(Boolean).join(' · ')}</span>
    <FileStages file={file}/>
  </div>;

  const details = <ApplicationDetails a={a} file={file} user={user} onEdit={onEdit} notify={notify} onDocuments={loadFile}/>;

  if (!final) return <div className="price-offer">{header}{details}
    <section className="po-card po-pending">{failed
      ? <InfoBox warning>Pricing did not complete. <button className="link-button" onClick={() => { tried.current = null; setFailed(false); }}>Try again</button></InfoBox>
      : <p><CircleNotch className="spin" size={18}/> Pricing this file…</p>}</section></div>;

  const limitBps = user.limits?.offer_below_recommendation_bps ?? 15;
  // No offer above the top of the range the customer was quoted at Stage 1.
  const quoted = final.indicative_band;
  const quotedCap = quoted?.max_rate != null && quoted.max_rate < final.envelope?.max_rate && quoted.max_rate >= rec;
  const floor = final.envelope?.min_rate, ceiling = quotedCap ? quoted.max_rate : final.envelope?.max_rate;
  const reasonFloor = Math.max(+(rec - limitBps / 100).toFixed(2), floor ?? 0);
  const value = Number(rate);
  const below = value < rec - 1e-9, needsManager = value < reasonFloor - 1e-9;
  const outside = value > ceiling + 1e-9 || value < floor - 1e-9;
  const loan = a.sanctioned_loan_amount || a.requested_loan_amount;
  const years = a.tenure_years || a.requested_tenure_years;
  const comp = a.sales_context || {};
  const accept = final.recommendation.conversion_probability;

  const openOffer = history.offers.find(o => o.status === 'sent');
  const accepted = history.offers.find(o => o.status === 'accepted');
  const pending = history.requests.find(r => r.status === 'pending');
  const lastRequest = history.requests[0];

  async function send() {
    setSending(true); setError('');
    try {
      const r = await api('/pricing/offers', { loan_id: a.loan_id, rate: value, reason: below ? reason : null, note: note || null });
      notify(`Offer sent at ${pct(r.rate)}.`); await loadHistory(); loadFile();
    } catch (e) { setError(e.message); } finally { setSending(false); }
  }
  async function askManager() {
    if (!reason) { setError('Give a reason for the lower rate before sending it to the area manager.'); return; }
    setSending(true); setError('');
    try {
      await api('/pricing/rate-requests', { loan_id: a.loan_id, rate: value, reason, note: note || null });
      notify(`Sent to the area manager: ${pct(value)} requested.`); await loadHistory(); loadFile();
    } catch (e) { setError(e.message); } finally { setSending(false); }
  }

  // The authority bar: widths follow the rate span, with a minimum so every band reads.
  const span = Math.max(ceiling - floor, 0.01);
  const w = (lo, hi) => Math.max(((hi - lo) / span) * 100, 14);
  const bands = [
    { key: 'mgr', label: `Below ${pct(reasonFloor)}: area manager`, width: w(floor, reasonFloor) },
    { key: 'reason', label: `${pct(reasonFloor)}–${pct(rec - 0.01)}: with a reason`, width: w(reasonFloor, rec) },
    { key: 'direct', label: `${pct(rec)}–${pct(ceiling)}: send directly`, width: w(rec, ceiling) },
  ];
  const here = outside ? null : needsManager ? 'mgr' : below ? 'reason' : 'direct';
  // A lower rate given for a competitor's offer, against what the fact check says that offer can justify.
  const cap = check?.score?.max_discount_bps;
  const claimShort = below && reason === 'competitor_lower_rate' && cap != null && (rec - value) * 100 > cap + 1e-9
    ? (cap === 0 ? `The competitor fact check scores this quote ${check.score.points}/100 (${check.score.label}): it cannot justify a lower rate.`
       : `The competitor fact check (${check.score.points}/100, ${check.score.label}) supports at most ${(cap / 100).toFixed(2)}% off the recommendation for this quote.`)
    : null;

  return <div className="price-offer">{header}{details}
    <div className="po-layout">
      <section className="po-card">
        <div className="po-top">
          <div><span className="po-label">Recommended rate</span><strong className="po-rate">{pct(rec)}</strong></div>
          <div className="po-facts">
            {accept != null && <b>{Math.round(accept * 100)} in 100 similar customers accept</b>}
            <span>EMI {money(emi(loan, rec, years))}{comp.competitor_rate ? <> · Competitor quote {pct(comp.competitor_rate)}, {comp.competitor_evidence_verified ? 'verified' : 'unverified'}</> : null}</span>
            {quoted ? <span>Quoted at Stage 1: {pct(quoted.min_rate)}–{pct(quoted.max_rate)}{quoted.within ? ' · the recommendation is inside it' : ' · policy moved the rate outside it'}</span>
              : pre?.band && <span>Indicative range at application: {pct(pre.band.min_rate)}–{pct(pre.band.max_rate)}</span>}
          </div>
        </div>

        <CompetitorCheck applicant={a} onResult={setCheck}/>

        <div className="po-authority">
          <span className="po-label strong">Your pricing authority</span>
          <div className="po-bands">{bands.map(b => <div key={b.key} className={`po-band ${b.key}${here === b.key ? ' here' : ''}`} style={{ flex: b.width }}>{b.label}</div>)}</div>
          <small>Below the recommendation, say why a higher rate was not possible. {quotedCap
            ? <>Above {pct(ceiling)} is above the range quoted to the customer; below {pct(floor)} is outside policy.</>
            : <>Outside {pct(floor)}–{pct(ceiling)} is outside policy.</>}</small>
        </div>

        {openOffer && <InfoBox success><CheckCircle size={16}/> Offer sent at <b>{pct(openOffer.rate)}</b> on {when(openOffer.sent_at)}{openOffer.route === 'manager_approved' ? ', approved by the area manager' : ''}. Awaiting the customer.</InfoBox>}
        {pending && <InfoBox><UserSwitch size={16}/> With the area manager since {when(pending.requested_at)}: you asked for <b>{pct(pending.requested_rate)}</b> ({titleCase(pending.reason)}).</InfoBox>}
        {!pending && lastRequest?.status === 'declined' && !openOffer && <InfoBox warning>The area manager declined {pct(lastRequest.requested_rate)}: “{lastRequest.decision_note}”. Send at the recommendation or above.</InfoBox>}

        {accepted && <InfoBox success><CheckCircle size={16}/> The customer accepted the offer at <b>{pct(accepted.rate)}</b> ({when(accepted.customer_decided_at)}). Nothing more to do on this file.</InfoBox>}
        {accepted ? null : openOffer
          ? <CustomerDecision offer={openOffer} onDone={() => { loadHistory(); loadFile(); notify('Customer decision recorded.'); }}/>
          : !pending && <>
            <div className="po-form">
              <label><span>Offer rate (%)</span><input type="number" step="0.01" min={floor} max={ceiling} value={rate} onChange={e => { setRate(e.target.value); setError(''); }}/></label>
              <label><span>Reason if lower</span><select value={reason} onChange={e => setReason(e.target.value)} disabled={!below}>
                <option value="">{below ? 'Choose a reason' : 'Not needed at this rate'}</option>
                {reasons.map(r => <option key={r} value={r}>{titleCase(r)}</option>)}</select></label>
              <label><span>Note</span><input value={note} placeholder="Optional" onChange={e => setNote(e.target.value)}/></label>
            </div>
            {claimShort && <p className="po-warn">{claimShort}</p>}
            {error && <p className="po-error" role="alert">{error}</p>}
            <div className="po-actions">
              <Button primary busy={sending} disabled={outside || needsManager || !value || (below && !reason)} onClick={send}><PaperPlaneTilt size={18}/> Send offer at {value ? pct(value) : '—'}</Button>
              <Button busy={sending} disabled={!needsManager || outside} onClick={askManager}>Send to manager for a lower rate</Button>
            </div>
            {outside && <small className="po-error">{quotedCap && value > ceiling
              ? `${pct(value)} is above the range quoted to the customer (up to ${pct(ceiling)}).`
              : `${pct(value)} is outside policy (${pct(floor)}–${pct(ceiling)}).`}</small>}
          </>}
      </section>
    </div>
  </div>;
}

// Everything on the file, read only; "Edit details" changes it and reprices.
function ApplicationDetails({ a, file, user, onEdit, notify, onDocuments }) {
  const [docsOpen, setDocsOpen] = useState(false);
  const sc = a.sales_context || {};
  const loan = a.sanctioned_loan_amount || a.requested_loan_amount;
  const value = a.appraised_property_value || a.estimated_property_value;
  const mine = file?.source === 'assisted' && file?.created_by === user.username;
  // Four columns; the last holds the market context and the file itself.
  const columns = [
    [['Borrower', [
      ['Name', a.name], ['Mobile', a.mobile], ['Email', a.email], ['Age', a.applicant_age],
      ['Employment', SEGMENTS[a.segment] || titleCase(a.segment)], ['City', a.city],
    ]]],
    [['Income and credit', [
      ['Declared monthly income', money(a.declared_monthly_income)], ['Verified monthly income', verified(a.verified_monthly_income, money)],
      ['Monthly obligations', money(a.declared_obligations)],
      ['Credit score', a.new_to_credit || a.bureau_score < 0 ? 'New to credit' : a.bureau_score],
      ['Credit history', verified(a.bureau_vintage_months, v => `${v} months`)],
      ['Banking behaviour', verified(a.banking_behaviour_score, v => `${v} / 100`)],
      ['Delinquencies, last 24 months', verified(a.delinquency_count_24m)], ['Credit deviations', verified(a.deviation_count)],
    ]]],
    [['Loan and property', [
      ['Requested loan', money(a.requested_loan_amount)], ['Assessed loan', verified(a.sanctioned_loan_amount, money)],
      ['Requested tenure', a.requested_tenure_years && `${a.requested_tenure_years} years`],
      ['Assessed tenure', verified(a.tenure_years, v => `${v} years`)],
      ['Property type', PROPERTY[a.property_type] || titleCase(a.property_type)],
      ['Estimated property value', money(a.estimated_property_value)], ['Appraised value', verified(a.appraised_property_value, money)],
      ['Loan to value', loan && value ? `${Math.round((loan / value) * 100)}%` : null],
      ['Legal check', check(a.legal_flag)], ['Technical check', check(a.technical_flag)],
    ]]],
    [['Market context', [
      ['Competitor quote', sc.competitor_rate ? `${pct(sc.competitor_rate)}${sc.competitor_name ? ` · ${sc.competitor_name}` : ''}` : 'None'],
      ['Evidence', sc.competitor_rate ? (sc.competitor_evidence_verified ? 'Sanction letter seen' : 'Not yet seen') : null],
      ['Customer negotiating', yesNo(sc.customer_negotiating)], ['Negotiation rounds', sc.prior_negotiation_rounds],
      ['Existing customer', yesNo(sc.existing_relationship)],
    ]], ['Application', [
      ['Application ID', a.loan_id],
      ['Source', file ? (file.source === 'assisted' ? `Entered by ${mine ? 'you' : file.created_by}` : 'Applied online') : null],
      ['Channel', CHANNELS[a.channel] || titleCase(a.channel)],
      ['Submitted', when(file?.created_at)],
      ['Status', file === null ? 'Not submitted yet' : file && (STATUS[file.stage] || titleCase(file.stage || file.status))],
      ['Documents', file ? (file.documents ? `Received (${file.documents})` : 'None yet') : null],
    ]]],
  ];
  return <section className="po-card po-details">
    <div className="po-details-head"><h2>Application details</h2>
      <div className="po-actions">
        {file && <button className="link-button po-edit" onClick={() => setDocsOpen(true)}><FileText size={14}/> {mine ? 'Documents and upload' : 'View documents'}</button>}
        <button className="link-button po-edit" onClick={onEdit}><PencilSimple size={14}/> Edit details</button>
      </div></div>
    <div className="po-sections">{columns.map(groups => <div key={groups[0][0]}>{groups.map(([title, rows]) => <div key={title} className="po-group">
      <h3>{title}</h3>
      <dl className="po-dl">{rows.map(([label, v]) => <div key={label}><dt>{label}</dt><dd>{shown(v)}</dd></div>)}
        {title === 'Application' && file && <div><dt>Customer login</dt><dd><CustomerLoginControl loanId={a.loan_id}
          username={file.customer_login} name={a.name} notify={notify} onChange={onDocuments}/></dd></div>}</dl>
    </div>)}</div>)}</div>
    {a.conversation_notes && <div className="po-notes"><h3>Notes from the conversation</h3><p>{a.conversation_notes}</p></div>}
    {docsOpen && <Modal wide title={`Documents · ${a.name} · ${a.loan_id}`} onClose={() => setDocsOpen(false)}>
      <ApplicationDocuments loanId={a.loan_id} notify={notify} canUpload={mine} onChange={onDocuments}/>
    </Modal>}
  </section>;
}

function CustomerDecision({ offer, onDone }) {
  const [accepted, setAccepted] = useState('yes');
  const [finalRate, setFinalRate] = useState(String(offer.rate));
  const [lossReasons, setLossReasons] = useState([]);
  const [loss, setLoss] = useState('');
  const [lostTo, setLostTo] = useState('');
  const [lostRate, setLostRate] = useState('');
  const [busy, setBusy] = useState(false); const [error, setError] = useState('');
  useEffect(() => { api('/pricing/outcome/loss-reasons').then(r => setLossReasons(r.reasons || [])).catch(() => {}); }, []);
  async function save() {
    setBusy(true); setError('');
    try {
      await api(`/pricing/offers/${offer.offer_id}/customer-decision`, {
        accepted: accepted === 'yes', final_rate: Number(finalRate) || null,
        loss_reason: accepted === 'yes' ? null : (loss || 'no_reason_recorded'),
        lost_to: accepted === 'yes' ? null : (lostTo || null), lost_to_rate: accepted === 'yes' ? null : (Number(lostRate) || null) });
      onDone();
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }
  return <div className="po-decision">
    <h3>Customer decision</h3>
    <div className="po-form four">
      <label><span>Decision</span><select value={accepted} onChange={e => setAccepted(e.target.value)}><option value="yes">Accepted</option><option value="no">Not accepted</option></select></label>
      <label><span>Final rate (%)</span><input type="number" step="0.01" value={finalRate} onChange={e => setFinalRate(e.target.value)}/></label>
      <label><span>Reason, if not accepted</span><select value={loss} disabled={accepted === 'yes'} onChange={e => setLoss(e.target.value)}>
        <option value="">Not applicable</option>{lossReasons.map(r => <option key={r.code} value={r.code}>{r.label}</option>)}</select></label>
      <label><span>Other lender and rate, if any</span><div className="po-pair"><input value={lostTo} placeholder="Lender" disabled={accepted === 'yes'} onChange={e => setLostTo(e.target.value)}/><input type="number" step="0.01" value={lostRate} placeholder="%" disabled={accepted === 'yes'} onChange={e => setLostRate(e.target.value)}/></div></label>
    </div>
    {error && <p className="po-error" role="alert">{error}</p>}
    <div className="po-actions"><Button primary busy={busy} onClick={save}>Save decision</Button>
      <small>Customer decisions are recorded against the offer and used to improve future recommendations.</small></div>
  </div>;
}
