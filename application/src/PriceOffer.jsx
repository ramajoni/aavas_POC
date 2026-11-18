import React, { useEffect, useRef, useState } from 'react';
import { Briefcase, CalendarBlank, CheckCircle, CircleNotch, Envelope, FileText, HouseLine, LockSimple, MapPin, NotePencil, PaperPlaneTilt, PencilSimple, Phone, UserSwitch, Wallet, WarningCircle } from '@phosphor-icons/react';
import { api, emi, hasFile, money, percent, titleCase } from './api';
import { Badge, Button, Empty, Field, InfoBox, Modal, Panel } from './components';
import { ApplicationDocuments } from './Documents';
import { CustomerLoginControl } from './CustomerLogin';
import { PolicyLink } from './PolicyViewer';
import { ApplicationStatus } from './Status';
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

const RM_TABS = ['Overview', 'Assessment Details', 'Risk & Underwriting', 'Customer Interaction'];
const STAGE_BADGE = { in_progress: 'Documents due', ready: 'Ready for final offer', pending_rate_approval: 'Pending rate approval',
  offer_sent: 'Offer sent', accepted: 'Accepted', not_accepted: 'Not taken' };

// The three steps a file is filled in before it is priced and sent, from the
// server's readiness check: the customer's own figures (the indicative range), every
// verified figure and both checks (the risk-adjusted rate), then the customer's
// interaction and negotiation (the acceptance rate). Each unlocks the next; nothing is sent until
// all three are done.
function FileSteps({ ready, indicative, onEdit }) {
  if (!ready) return <section className="po-card po-steps"><p className="muted"><CircleNotch className="spin" size={16}/> Checking the file…</p></section>;
  return <section className="po-card po-steps" aria-label="What this file needs before it is priced and sent">
    <div className="po-details-head"><h2>Before pricing and sending</h2>
      {!ready.complete && <Button primary onClick={onEdit}><PencilSimple size={16}/> {(ready.steps.find(st => st.key === ready.next) || {}).action || 'Fill in the file'}</Button>}</div>
    <ol className="po-step-list">{ready.steps.map((st, i) => {
      const state = st.done ? 'done' : st.locked ? 'locked' : 'current';
      const due = [...st.missing.map(m => m.label)];
      return <li key={st.key} className={state}>
        <div className="po-step-head"><span className="po-step-mark">{st.done ? <CheckCircle size={18} weight="fill"/> : st.locked ? <LockSimple size={15}/> : i + 1}</span>
          <div><b>{st.label}</b><small>{st.done ? 'Done' : st.locked ? `Waiting for step ${i}` : `Unlocks ${st.unlocks}`}</small></div></div>
        {st.key === 'indicative' && st.done && indicative && <p className="po-step-note">Indicative range {pct(indicative.min_rate)}–{pct(indicative.max_rate)}</p>}
        {!st.done && !st.locked && due.length > 0 && <p className="po-step-due"><b>Still to fill ({due.length}):</b> {due.join(' · ')}</p>}
        {st.problems.map(p => <p key={p} className="po-step-problem"><WarningCircle size={15}/>{p}</p>)}
        {st.warnings.map(w => <p key={w} className="po-step-warning">{w}</p>)}
      </li>; })}</ol>
  </section>;
}

// The final acceptance rate: how likely the customer is to accept at each rate on
// the screen (the recommendation, the rate being offered, one already sent or asked
// for), scored on the file as last priced, as every other screen scores it.
function useAcceptance(loanId, rates, on) {
  const [scores, setScores] = useState({});
  const key = [...new Set(rates.filter(r => Number(r) > 0 && Number(r) < 30).map(r => Number(r).toFixed(2)))].join(',');
  useEffect(() => {
    if (!on || !key) return;
    let live = true;
    const t = setTimeout(() => api(`/pricing/offers/acceptance?loan_id=${encodeURIComponent(loanId)}&rates=${key}`)
      .then(r => { if (live) setScores(s => ({ ...s, ...Object.fromEntries(r.acceptance.map(x => [x.rate.toFixed(2), x.acceptance])) })); })
      .catch(() => {}), 300);
    return () => { live = false; clearTimeout(t); };
  }, [loanId, key, on]);
  useEffect(() => { setScores({}); }, [loanId, on]);
  return r => (r == null || !(Number(r) > 0) ? null : scores[Number(r).toFixed(2)] ?? null);
}
const inHundred = p => `${Math.round(p * 100)}%`;

// Wireframe 04, "RM · Price and send offer". The relationship manager sees the
// price, its band and how customers respond - not the analysis behind it, which
// the server does not send to this role.
export function PriceOffer({ state, setState, runFinal, busy, notify, navigate, onEdit, user }) {
  const { applicant: a, pre, final } = state;
  const tried = useRef(null); const key = JSON.stringify(a);
  const [failed, setFailed] = useState(false);
  // What the file still needs, read again whenever its details change.
  const [ready, setReady] = useState(null);
  useEffect(() => {
    if (!hasFile(a)) return;
    let live = true; setReady(null);
    api('/applications/readiness', { applicant: a }).then(r => { if (live) setReady(r); }).catch(() => { if (live) setReady(false); });
    return () => { live = false; };
  }, [key]);
  const [history, setHistory] = useState({ offers: [], requests: [] });
  // A file already with the customer or the area manager is past the steps.
  const pastSteps = history.offers.some(o => ['sent', 'accepted'].includes(o.status)) || history.requests.some(r => r.status === 'pending');
  const canPrice = pastSteps || ready?.done?.risk || ready === false;
  // The risk-adjusted rate is worked out only once every verified figure is in.
  useEffect(() => {
    if (!hasFile(a) || final || busy || !canPrice || tried.current === key) return;
    tried.current = key; setFailed(false);
    runFinal({ quiet: true }).then(ok => { if (!ok) setFailed(true); });
  }, [key, final, busy, canPrice]);

  const [reasons, setReasons] = useState([]);
  useEffect(() => { api('/pricing/override/reasons').then(r => setReasons(r.allowed_reasons || [])).catch(() => {}); }, []);
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
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { if (rec != null) setRate(String(rec)); }, [rec]);
  // Scored once the customer interaction is recorded (step 3), and again as the rate changes.
  const scoreOn = Boolean(final?.decision_id) && (pastSteps || ready === false || Boolean(ready?.complete));
  const sentOrAsked = [...history.offers.filter(o => o.status === 'sent').map(o => o.rate), ...history.requests.filter(r => r.status === 'pending').map(r => r.requested_rate)];
  const acceptanceAt = useAcceptance(a.loan_id, [rec, Number(rate), ...sentOrAsked], scoreOn);
  // The same tabs as the area manager's file, without the analysis and the audit trail.
  const [tab, setTab] = useState('Overview');
  const [noteOpen, setNoteOpen] = useState(false);
  const [noteText, setNoteText] = useState('');

  if (!hasFile(a)) return <div className="price-offer"><div className="po-head"><h1>Price and send offer</h1></div>
    <section className="po-card po-pending"><p>No file is open. Open one from My applications, or enter a new one.</p>
      <div className="po-actions"><Button primary onClick={() => navigate('applications')}>My applications</Button><Button onClick={() => navigate('new-application')}>New application</Button></div></section></div>;

  const initials = String(a.name || '').split(' ').filter(Boolean).map(n => n[0]).join('').slice(0, 2).toUpperCase();
  const header = <>
    <div className="breadcrumb">Home Loans <span>›</span> <button className="link-button" onClick={() => navigate('applications')}>My applications</button> <span>›</span><b>{a.loan_id}</b></div>
    <div className="application-title"><div><div className="title-line"><h1>Application #{a.loan_id}</h1>
      <Badge tone={file?.stage === 'accepted' ? 'green' : 'blue'}>{STAGE_BADGE[file?.stage] || (final ? 'Under assessment' : 'New application')}</Badge></div>
      <p>Assigned to: {user.name} <span>|</span> Branch: {a.city || '—'} <span>|</span> Channel: {CHANNELS[a.channel] || titleCase(a.channel)}</p></div>
      <div className="actions"><Button onClick={() => setNoteOpen(true)}><NotePencil size={18}/> Add Note</Button>
        {busy && <span className="auto-pricing" role="status"><CircleNotch className="spin" size={16}/> Pricing this file…</span>}</div></div>
    <ApplicationStatus loanId={a.loan_id} refresh={`${file?.updated_at}|${final?.decision_id}|${history.offers.length}|${history.requests.map(r => r.status).join()}|${history.offers.map(o => o.status).join()}`}/>
    <nav className="tabs" aria-label="Application views">{RM_TABS.map(t =>
      <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>)}</nav>
    <div className="customer-strip panel">
      <div className="customer-name"><div className="avatar pale">{initials}</div><div><h3>{a.name}</h3>
        <small><Phone size={12}/>{a.mobile || '—'}</small><small><Envelope size={12}/>{a.email || '—'}</small></div></div>
      {[[CalendarBlank, 'Age', a.applicant_age ? `${a.applicant_age} years` : '—'], [Briefcase, 'Employment', SEGMENTS[a.segment] || '—'],
        [Wallet, 'Monthly Income', money(a.declared_monthly_income)], [HouseLine, 'Requested Loan', money(a.requested_loan_amount)],
        [MapPin, 'Location', a.city || '—']].map(([Icon, t, v]) =>
        <div className="customer-detail" key={t}><Icon size={23}/><div><small>{t}</small><b>{v}</b></div></div>)}
      <div className="credit-score"><strong>{a.new_to_credit || !(a.bureau_score > 0) ? 'NTC' : a.bureau_score}</strong><small>Bureau Score</small>
        <progress max="900" value={Math.max(Number(a.bureau_score) || 0, 0)}/></div>
      <button className="icon-button" aria-label="Edit applicant details" onClick={onEdit} disabled={busy}><PencilSimple size={18}/></button>
    </div>
  </>;

  const details = <ApplicationDetails a={a} file={file} user={user} onEdit={onEdit} notify={notify} onDocuments={loadFile}/>;
  const steps = !pastSteps && ready !== false && !ready?.complete
    ? <FileSteps ready={ready} indicative={file?.indicative || pre?.band} onEdit={onEdit}/> : null;
  const noteModal = noteOpen && <Modal title="Add customer note" onClose={() => setNoteOpen(false)}>
    <form onSubmit={e => { e.preventDefault(); setState(s => ({ ...s, notes: [...(s.notes || []), { at: new Date().toISOString(), text: noteText }] })); setNoteText(''); setNoteOpen(false); setTab('Customer Interaction'); notify('Note added to this file in your browser.'); }}>
      <Field label="Conversation note"><textarea required rows="5" value={noteText} onChange={e => setNoteText(e.target.value)} placeholder="What did the customer share?"/></Field>
      <div className="modal-actions"><Button primary type="submit">Save Note</Button></div></form></Modal>;
  // Tabs other than the overview: the verified figures, the risk grade and policy
  // check the RM is shown, and the conversation with the customer.
  const other = tab === 'Assessment Details' ? <AssessmentTab a={a} file={file} onEdit={onEdit}/>
    : tab === 'Risk & Underwriting' ? <RiskTab final={final} busy={busy}/>
    : tab === 'Customer Interaction' ? <InteractionTab notes={state.notes || []} onAdd={() => setNoteOpen(true)} file={file}/> : null;
  if (other) return <div className="price-offer rm-file">{header}{other}{noteModal}</div>;

  if (!canPrice) return <div className="price-offer rm-file">{header}{steps}{details}{noteModal}</div>;

  if (!final) return <div className="price-offer rm-file">{header}{steps}{details}
    <section className="po-card po-pending">{failed
      ? <InfoBox warning>Pricing did not complete. <button className="link-button" onClick={() => { tried.current = null; setFailed(false); }}>Try again</button></InfoBox>
      : <p><CircleNotch className="spin" size={18}/> Pricing this file…</p>}</section>{noteModal}</div>;

  const limitBps = user.limits?.offer_below_recommendation_bps ?? 15;
  // No offer above the top of the range the customer was quoted at Stage 1.
  const quoted = final.indicative_band;
  const quotedCap = quoted?.max_rate != null && quoted.max_rate < final.envelope?.max_rate && quoted.max_rate >= rec;
  const floor = final.envelope?.min_rate, ceiling = quotedCap ? quoted.max_rate : final.envelope?.max_rate;
  const reasonFloor = Math.max(+(rec - limitBps / 100).toFixed(2), floor ?? 0);
  const value = Number(rate);
  const below = value < rec - 1e-9, needsManager = value < reasonFloor - 1e-9;
  // Too high cannot be sent; too low goes to the area manager, who decides it.
  const outside = value > ceiling + 1e-9;
  const underFloor = floor != null && value > 0 && value < floor - 1e-9;
  const loan = a.sanctioned_loan_amount || a.requested_loan_amount;
  const years = a.tenure_years || a.requested_tenure_years;
  const comp = a.sales_context || {};
  // The acceptance rate waits for the customer interaction (step 3), and so does sending.
  const behaviourIn = pastSteps || ready === false || ready?.complete;
  const accept = behaviourIn ? (acceptanceAt(rec) ?? final.recommendation.conversion_probability) : null;
  // The final acceptance rate: at the rate in the offer box, beside the recommendation's.
  const finalAccept = behaviourIn && value > 0 ? acceptanceAt(value) : null;
  const vsRec = finalAccept != null && accept != null ? Math.round(finalAccept * 100) - Math.round(accept * 100) : null;

  const openOffer = history.offers.find(o => o.status === 'sent');
  const accepted = history.offers.find(o => o.status === 'accepted');
  const pending = history.requests.find(r => r.status === 'pending');
  const lastRequest = history.requests[0];

  async function send() {
    setSending(true); setError('');
    try {
      const r = await api('/pricing/offers', { loan_id: a.loan_id, rate: value, reason: below ? reason : null, note: null });
      notify(`Offer sent at ${pct(r.rate)}.`); await loadHistory(); loadFile();
    } catch (e) { setError(e.message); } finally { setSending(false); }
  }
  async function askManager() {
    if (!reason) { setError('Give a reason for the lower rate before sending it to the area manager.'); return; }
    setSending(true); setError('');
    try {
      await api('/pricing/rate-requests', { loan_id: a.loan_id, rate: value, reason, note: null });
      notify(`Sent to the area manager: ${pct(value)} requested.`); await loadHistory(); loadFile();
    } catch (e) { setError(e.message); } finally { setSending(false); }
  }

  // The authority bar: widths follow the rate span, with a minimum so every band reads.
  const span = Math.max(ceiling - floor, 0.01);
  const w = (lo, hi) => Math.max(((hi - lo) / span) * 100, 14);
  // Only the bands this file has: when the recommendation already sits at the policy
  // floor there is nothing below it for anyone to approve.
  const atFloor = floor != null && rec <= floor + 1e-9;
  const bands = [
    { key: 'mgr', label: `Below ${pct(reasonFloor)}: area manager`, width: w(floor, reasonFloor) },
    rec > reasonFloor + 1e-9 && { key: 'reason', label: `${pct(reasonFloor)}–${pct(rec - 0.01)}: with a reason`, width: w(reasonFloor, rec) },
    { key: 'direct', label: `${pct(rec)}–${pct(ceiling)}: send directly`, width: w(rec, ceiling) },
  ].filter(Boolean);
  const card = final.envelope?.card_rate;
  const floorWhy = card != null && floor != null ? ` (card rate ${pct(card)} less the largest discount policy allows, ${(card - floor).toFixed(2)}%)` : '';
  const here = outside ? null : needsManager ? 'mgr' : below ? 'reason' : 'direct';

  return <div className="price-offer rm-file">{header}{steps}{details}{noteModal}
    <div className="po-layout">
      <section className="po-card">
        <div className="po-top">
          <div><span className="po-label">Recommended rate</span><strong className="po-rate">{pct(rec)}</strong></div>
          <div className="po-facts">
            {accept != null ? <b>{inHundred(accept)} of similar customers accept at {pct(rec)}</b>
              : <b className="muted">Acceptance rate: record the customer interaction (step 3) to see it</b>}
            <span>EMI {money(emi(loan, rec, years))}{comp.competitor_rate ? <> · Competitor quote {pct(comp.competitor_rate)}, {comp.competitor_evidence_verified ? 'verified' : 'unverified'}</> : null}</span>
            {quoted ? <span>Quoted at Stage 1: {pct(quoted.min_rate)}–{pct(quoted.max_rate)}{quoted.within ? ' · the recommendation is inside it' : ' · policy moved the rate outside it'}</span>
              : pre?.band && <span>Indicative range at application: {pct(pre.band.min_rate)}–{pct(pre.band.max_rate)}</span>}
          </div>
        </div>


        <div className="po-authority">
          <span className="po-label strong">Your pricing authority</span>
          <div className="po-bands">{bands.map(b => <div key={b.key} className={`po-band ${b.key}${here === b.key ? ' here' : ''}`} style={{ flex: b.width }}>{b.label}</div>)}</div>
          <small>{atFloor ? <>{pct(rec)} is the lowest rate policy allows for this file{floorWhy}. A lower rate can still go to the area manager, who decides it. </>
            : <>Below the recommendation, say why a higher rate was not possible. Below {pct(floor)}, the lowest rate policy allows, the area manager decides. </>}{quotedCap
            ? <>Above {pct(ceiling)} is above the range quoted to the customer.</> : <>Above {pct(ceiling)} is outside policy.</>}</small>
        </div>

        {openOffer && <InfoBox success><CheckCircle size={16}/> Offer sent at <b>{pct(openOffer.rate)}</b> on {when(openOffer.sent_at)}{openOffer.route === 'manager_approved' ? ', approved by the area manager' : openOffer.route === 'manager_uplifted' ? ', uplifted by the area manager' : ''}. Awaiting the customer.{acceptanceAt(openOffer.rate) != null && <> Final acceptance rate at this rate: <b>{inHundred(acceptanceAt(openOffer.rate))}</b>.</>}</InfoBox>}
        {pending && <InfoBox><UserSwitch size={16}/> With the area manager since {when(pending.requested_at)}: you asked for <b>{pct(pending.requested_rate)}</b> ({titleCase(pending.reason)}).{acceptanceAt(pending.requested_rate) != null && <> Final acceptance rate at that rate: <b>{inHundred(acceptanceAt(pending.requested_rate))}</b>.</>}</InfoBox>}
        {!pending && lastRequest?.status === 'declined' && !openOffer && <InfoBox warning>The area manager declined {pct(lastRequest.requested_rate)}: “{lastRequest.decision_note}”. Send at the recommendation or above.</InfoBox>}

        {accepted && <InfoBox success><CheckCircle size={16}/> The customer accepted the offer at <b>{pct(accepted.rate)}</b> ({when(accepted.customer_decided_at)}). Nothing more to do on this file.</InfoBox>}
        {accepted ? null : openOffer
          ? <CustomerDecision offer={openOffer} onDone={() => { loadHistory(); loadFile(); notify('Customer decision recorded.'); }}/>
          : !pending && !behaviourIn ? <InfoBox><LockSimple size={16}/> Record the customer interaction and negotiation (step 3) to see the acceptance rate and send this file, to the customer or to the area manager.</InfoBox>
          : !pending && <>
            <div className="po-form">
              <label><span>Offer rate (%)</span><input type="number" step="0.01" max={ceiling} value={rate} onChange={e => { setRate(e.target.value); setError(''); }}/></label>
              <label><span>Reason if lower</span><select value={reason} onChange={e => setReason(e.target.value)} disabled={!below}>
                <option value="">{below ? 'Choose a reason' : 'Not needed at this rate'}</option>
                {reasons.map(r => <option key={r} value={r}>{titleCase(r)}</option>)}</select></label>
              <div className="po-final-accept" aria-live="polite"><span>Final acceptance rate</span>
                <b>{finalAccept != null ? inHundred(finalAccept) : value > 0 ? '…' : '—'}</b>
                <small>{!(value > 0) ? 'Enter a rate' : `likely to accept at ${pct(value)}${vsRec == null || Math.abs(value - rec) < 0.005 ? '' : vsRec === 0 ? ', same as at the recommendation' : `, ${Math.abs(vsRec)} point${Math.abs(vsRec) === 1 ? '' : 's'} ${vsRec > 0 ? 'higher' : 'lower'} than at ${pct(rec)}`}`}</small></div>
            </div>
            {error && <p className="po-error" role="alert">{error}</p>}
            <div className="po-actions">
              <Button primary busy={sending} disabled={outside || needsManager || !value || (below && !reason)} onClick={send}><PaperPlaneTilt size={18}/> Send offer at {value ? pct(value) : '—'}</Button>
              <Button busy={sending} disabled={!needsManager || outside} onClick={askManager}>Send to manager for a lower rate</Button>
            </div>
            {outside && <small className="po-error">{quotedCap
              ? `${pct(value)} is above the range quoted to the customer (up to ${pct(ceiling)}).`
              : `${pct(value)} is above the policy ceiling of ${pct(ceiling)}.`}</small>}
            {underFloor && <small className="po-warn">{pct(value)} is below {pct(floor)}, the lowest rate policy allows for this file{floorWhy}. You can send it to the area manager: {atFloor
              ? `nothing below ${pct(rec)} is within policy here, so expect a decline.`
              : `they can approve ${pct(floor)} or above, or decline.`}</small>}
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

// The figures the assessment verifies; until one is, pricing uses the declared figure.
function AssessmentTab({ a, file, onEdit }) {
  const loan = a.sanctioned_loan_amount || a.requested_loan_amount;
  const value = a.appraised_property_value || a.estimated_property_value;
  const rows = [['Verified monthly income', verified(a.verified_monthly_income, money)], ['Appraised property value', verified(a.appraised_property_value, money)],
    ['Assessed loan amount', verified(a.sanctioned_loan_amount, money)], ['Loan to value', loan && value ? `${Math.round((loan / value) * 100)}%` : '—'],
    ['Assessed tenure', verified(a.tenure_years, v => `${v} years`)], ['Credit history', verified(a.bureau_vintage_months, v => `${v} months`)],
    ['Banking behaviour', verified(a.banking_behaviour_score, v => `${v} / 100`)], ['Delinquencies, last 24 months', verified(a.delinquency_count_24m)],
    ['Credit deviations', verified(a.deviation_count)], ['Legal check', check(a.legal_flag)], ['Technical check', check(a.technical_flag)],
    ['Documents', file ? (file.missing_documents?.length ? `${file.documents} in · ${file.missing_documents.length} still due` : `Verified (${file.documents})`) : '—']];
  return <Panel title="Verified assessment details" action={<Button onClick={onEdit}><PencilSimple size={16}/> Edit details</Button>}>
    <div className="details-grid">{rows.map(([t, v]) => <div key={t}><span>{t}</span><b>{v}</b></div>)}</div>
    <InfoBox>The risk-adjusted rate is worked out once every figure here is verified and both checks are recorded. Edit details to record what the documents show; the file is priced again.</InfoBox>
  </Panel>;
}

// What the RM is shown of risk: the grade, the authority the price needs and the policy check.
function RiskTab({ final, busy }) {
  if (!final) return <Panel title="Risk & underwriting review"><Empty title={busy ? 'Pricing this file' : 'Assessment pending'}>The file is priced when it opens; the risk grade and policy check appear here.</Empty></Panel>;
  const p = final.policy || {}, env = final.envelope || {};
  const rows = [['Risk grade', final.risk?.risk_grade || '—'], ['Required authority', titleCase(p.required_approval) || '—'],
    ['Policy check', p.allowed === false ? 'Outside policy' : 'Inside policy'], ['Permissible range', env.min_rate != null ? `${pct(env.min_rate)}–${pct(env.max_rate)}` : '—'],
    ['Card rate', env.card_rate != null ? pct(env.card_rate) : '—'], ['Policy', <PolicyLink version={p.policy_version}/>]];
  return <Panel title="Risk & underwriting review">
    <div className="details-grid">{rows.map(([t, v]) => <div key={t}><span>{t}</span><b>{v}</b></div>)}</div>
    {(p.violations || []).map(v => <InfoBox warning key={v}>{v}</InfoBox>)}
    <p className="muted">The analysis behind the grade and the price is reviewed by the area manager and risk.</p>
  </Panel>;
}

function InteractionTab({ notes, onAdd, file }) {
  return <Panel title="Customer interaction notes" action={<Button onClick={onAdd}><NotePencil size={16}/> Add Note</Button>}>
    {notes.length ? notes.map((n, i) => <div className="note" key={i}><small>Note #{i + 1} · {new Date(n.at).toLocaleString()}</small><p>{n.text}</p></div>)
      : <Empty title="No notes yet">Add notes from your conversation with the customer. Notes stay in this browser.</Empty>}
    {file?.customer_login && <p className="muted">The customer follows this file with the login {file.customer_login}.</p>}
  </Panel>;
}
