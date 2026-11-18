import React, { useEffect, useState } from 'react';
import { CircleNotch } from '@phosphor-icons/react';
import { api, money, titleCase } from './api';
import { Button, InfoBox } from './components';
import { ApplicationDocuments } from './Documents';
import { RequestSummary, ManagerDecision } from './Approvals';
import { PriceScenarios } from './Scenarios';
import { Fold } from './Fold';
import { PolicyLink } from './PolicyViewer';
import { DecisionSupport } from './DecisionSupport';
import { useHistory, HistoryGate, SubTabs, FileUpselling, FileCompetitors, FileNotes, BranchUpselling, CompetitorStrength, SalesNotes } from './History';
import { ApprovalBriefing } from './Briefing';
import { openExplain } from './LoanExplain';
import { ApplicationStatus } from './Status';
import './review.css';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const lakh = n => n == null ? '—' : `₹${(n / 100000).toFixed(n % 100000 ? 1 : 0)} lakh`;
const SEGMENT = { salaried_formal: 'Salaried, formal employer', salaried_informal: 'Salaried, informal employer',
  self_employed_professional: 'Self-employed professional', self_employed_non_professional: 'Self-employed' };
const SHORT = { salaried_formal: 'Salaried', salaried_informal: 'Salaried (informal)',
  self_employed_professional: 'Self-employed professional', self_employed_non_professional: 'Self-employed' };
const PROPERTY = { resale_flat: 'Resale flat', builder_flat: 'Builder flat', self_construction: 'Self construction', plot_plus_construction: 'Plot + construction' };
export const REVIEW_TABS = [['decision', 'Decision'], ['support', 'Decision support'], ['application', 'Application'], ['branch', 'Branch details'], ['price', 'Pricing intelligence']];
const CHANNEL = { branch_walkin: 'Branch walk-in', dsa: 'DSA', connector: 'Connector', digital: 'Digital' };

// A reason code in the words a manager would use: "Loan is 69% of property value".
function factor(r) {
  const v = r.value;
  switch (r.feature) {
    case 'risk_grade': return `Risk grade ${v}`;
    case 'segment': return SEGMENT[v] || titleCase(v);
    case 'ltv': return typeof v === 'number' ? `Loan is ${Math.round(v * 100)}% of property value` : r.label;
    case 'bureau_score': return v == null || v < 0 ? 'New to credit' : `Credit score ${Math.round(v)}`;
    case 'bureau_vintage_months': return v ? `Credit history of ${Math.round(v)} months` : 'Thin credit history';
    case 'property_type': return PROPERTY[v] || titleCase(v);
    default: return typeof v === 'string' && !['True', 'False'].includes(v) ? `${r.label} (${titleCase(v)})` : r.label;
  }
}

function PricingRationale({ a, final, busy, failed, onRetry }) {
  if (!final) return <section className="rv-card"><h2>Pricing rationale</h2>
    {failed ? <InfoBox warning>Pricing did not complete. <button className="link-button" onClick={onRetry}>Try again</button></InfoBox>
      : <p className="rv-muted"><CircleNotch className="spin" size={16}/> {busy ? 'Pricing this file…' : 'Waiting for the price…'}</p>}</section>;
  const r = final.risk || {}, loan = a.sanctioned_loan_amount || a.requested_loan_amount;
  const facts = [SEGMENT[a.segment], r.risk_grade && `Risk grade ${r.risk_grade}`,
    r.pd != null && `About ${(r.pd * 100).toFixed(1)}% of similar borrowers default each year`,
    r.lgd != null && `Loss if default about ${Math.round(r.lgd * 100)}%`,
    r.expected_loss_pct != null && loan && `Expected loss about ${money(Math.round(r.expected_loss_pct / 100 * loan))} a year`].filter(Boolean);
  const first = String(a.name || 'The customer').split(' ')[0];
  const given = v => v !== '' && v != null;
  const record = [given(a.delinquency_count_24m) ? `${a.delinquency_count_24m} missed payments in 24 months` : 'payment history not verified yet',
    given(a.banking_behaviour_score) ? `banking behaviour ${a.banking_behaviour_score} / 100` : null,
    given(a.deviation_count) ? `${a.deviation_count} credit deviations` : null].filter(Boolean).join(', ');
  const codes = [...(final.explanation?.reason_codes || [])].sort((x, y) => (y.impact_bps || 0) - (x.impact_bps || 0)).slice(0, 4);
  const top = Math.max(1, ...codes.map(c => c.impact_bps || 0));
  const review = final.explanation?.fair_pricing_review;
  return <section className="rv-card">
    <h2>Pricing rationale</h2>
    <p className="rv-facts">{facts.join(' · ')}</p>
    <p className="rv-muted">{first}: {record}</p>
    <div className="rv-factors">{codes.map(c => <div key={c.code} className="rv-factor">
      <span>{factor(c)}</span>
      <i><b className={c.direction === 'increase' ? 'up' : 'down'} style={{ width: `${(c.impact_bps / top) * 100}%` }}/></i>
      <em className={c.direction === 'increase' ? 'up' : 'down'}>{c.direction === 'increase' ? '+' : '−'}{(Math.abs(c.impact_bps) / 100).toFixed(2)}%</em>
    </div>)}</div>
    <p className="rv-note">Each factor’s effect on the system’s benchmark rate; the card rate is {pct(final.envelope?.card_rate)}. Fair-pricing check: {review?.flagged ? 'flagged for review' : 'passed'}.</p>
    {final.decision_id && <button className="link-button rv-link" onClick={() => openExplain(final.decision_id)}>See full pricing rationale</button>}
  </section>;
}

const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const emiAt = (amount, rate, months) => { const r = rate / 1200; return r ? amount * r * (1 + r) ** months / ((1 + r) ** months - 1) : amount / months; };

// The loan this request was priced on, and what it costs the customer each month
// at the RM's ask and the system recommendation.
function LoanEmi({ a, request }) {
  const loan = request.loan_amount || a.sanctioned_loan_amount || a.requested_loan_amount;
  const years = Number(a.tenure_years || a.requested_tenure_years), months = years * 12;
  const value = a.appraised_property_value || a.estimated_property_value;
  const verified = given(a.verified_monthly_income), income = verified ? a.verified_monthly_income : a.declared_monthly_income;
  const other = Number(a.declared_obligations) || 0;
  if (!loan || !months) return null;
  const at = rate => { const emi = Math.round(emiAt(loan, rate, months)); return { emi, interest: emi * months - loan, share: income ? (emi + other) / income : null }; };
  const rows = [['RM’s ask', request.requested_rate, ''], ['System recommendation', request.recommended_rate, 'reco']];
  return <section className="rv-card rv-loan" aria-label="Loan and EMI">
    <h2>Loan and EMI</h2>
    <dl className="rv-loan-facts">
      <div><dt>Loan amount</dt><dd>{money(loan)}</dd></div>
      <div><dt>Tenure</dt><dd>{years} years<small>{months} EMIs</small></dd></div>
      <div><dt>Property</dt><dd>{money(value)}<small>{[PROPERTY[a.property_type] || titleCase(a.property_type), given(a.appraised_property_value) ? 'appraised' : 'customer’s estimate'].filter(Boolean).join(' · ')}</small></dd></div>
      <div><dt>Loan to value</dt><dd>{value ? `${Math.round((loan / value) * 100)}%` : '—'}</dd></div>
      <div><dt>Monthly income</dt><dd>{money(income)}<small>{verified ? 'verified' : 'declared, not verified yet'}</small></dd></div>
      <div><dt>Other EMIs</dt><dd>{money(other)}<small>a month</small></dd></div>
    </dl>
    <table className="rv-emi">
      <thead><tr><th></th><th>Rate</th><th>EMI a month</th><th>Total interest over {years} years</th><th>All EMIs as share of income</th></tr></thead>
      <tbody>
        {rows.map(([label, rate, cls]) => { const r = at(rate); return <tr key={label} className={cls}>
          <td>{label}</td><td>{pct(rate)}</td><td>{money(r.emi)}</td><td>{money(r.interest)}</td><td>{share(r.share)}</td></tr>; })}
      </tbody>
    </table>
    <p className="rv-note">EMI on {money(loan)} over {months} months. “All EMIs” adds the customer’s other EMIs of {money(other)} a month to this loan’s EMI.</p>
  </section>;
}

// The area manager's review of one lower-rate request: what was asked and what it
// costs, why the system priced it so, and the decision. Everything else is a tab.
const APP_SUBTABS = [['details', 'Details'], ['upselling', 'Upselling'], ['competitors', 'Competitors'], ['notes', 'Notes']];
const BRANCH_SUBTABS = [['summary', 'Summary'], ['upselling', 'Upselling'], ['strength', 'Competitor strength'], ['notes', 'Sales notes']];

export function RequestReview({ a, request, final, busy, failed, onRetry, tab, setTab, rate, setRate, reasons, upliftReasons, onDecided, notify }) {
  const [appSub, setAppSub] = useState('details');
  const [preset, setPreset] = useState(null);
  const [branchSub, setBranchSub] = useState('summary');
  const history = useHistory(request.request_id, (tab === 'application' && appSub !== 'details') || (tab === 'branch' && branchSub !== 'summary'));
  // Describe the file as it was priced for this request, not as it may have been edited since.
  const f = { ...a, ...(final?.application_snapshot || {}) };
  const loan = request.loan_amount || f.sanctioned_loan_amount || f.requested_loan_amount;
  const value = f.appraised_property_value || f.estimated_property_value;
  const meta = [a.loan_id, `RM ${request.requested_by_name || request.requested_by}`, SHORT[f.segment], f.city,
    `${f.tenure_years || f.requested_tenure_years} years`, loan && value ? `LTV ${Math.round((loan / value) * 100)}%` : null,
    f.new_to_credit || !(f.bureau_score > 0) ? 'New to credit' : `Credit score ${f.bureau_score}`].filter(Boolean);
  return <div className="rv">
    <div className="rv-head">
      <h1>{a.name} · {lakh(loan)} home loan</h1>
      <div className="rv-meta"><span>{meta.join(' · ')}</span><Button onClick={() => setTab('application')}>View application</Button></div>
    </div>
    <ApplicationStatus loanId={a.loan_id} refresh={`${request.request_id}|${final?.decision_id}`}/>
    <nav className="tabs" aria-label="Review">{REVIEW_TABS.map(([key, label]) =>
      <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{label}</button>)}</nav>
    {tab === 'decision' && <>
      <RequestSummary request={request}/>
      <LoanEmi a={f} request={request}/>
      <div className="rv-grid">
        <PricingRationale a={a} final={final} busy={busy} failed={failed} onRetry={onRetry}/>
        <ManagerDecision side preset={preset} request={request} rate={rate} setRate={setRate} reasons={reasons} upliftReasons={upliftReasons} onDecided={onDecided} notify={notify}/>
      </div>
    </>}
    {tab === 'application' && <><SubTabs tabs={APP_SUBTABS} value={appSub} onChange={setAppSub}/>
      {appSub === 'details' ? <ApplicationTab a={a} request={request} final={final} notify={notify}/>
        : <HistoryGate state={history}>{h => appSub === 'upselling' ? <FileUpselling h={h} request={request}/>
          : appSub === 'competitors' ? <FileCompetitors h={h} a={a}/> : <FileNotes h={h}/>}</HistoryGate>}</>}
    {tab === 'branch' && <><SubTabs tabs={BRANCH_SUBTABS} value={branchSub} onChange={setBranchSub}/>
      {branchSub === 'summary' ? <ApprovalBriefing requestId={request.request_id}/>
        : <HistoryGate state={history}>{h => branchSub === 'upselling' ? <BranchUpselling h={h}/>
          : branchSub === 'strength' ? <CompetitorStrength h={h}/> : <SalesNotes h={h}/>}</HistoryGate>}</>}
    {tab === 'support' && <DecisionSupport request={request} a={a} final={final} onUse={x => { setPreset({ ...x, at: Date.now() }); setTab('decision'); setTimeout(() => document.getElementById('mgr-decision')?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 80); }} go={(t, sub) => { setTab(t); if (t === 'application' && sub) setAppSub(sub); if (t === 'branch' && sub) setBranchSub(sub); }}/>}
    {tab === 'price' && <PriceScenarios requestId={request.request_id}/>}
  </div>;
}

const given = v => v !== '' && v != null;
const shown = (v, fmt = x => x) => given(v) ? fmt(v) : 'Not verified yet';
const flag = v => !given(v) ? 'Not checked yet' : Number(v) ? 'Issue flagged' : 'Clear';

function Section({ title, children, note }) {
  return <section className="rv-card"><h2>{title}</h2>{children}{note && <p className="rv-note">{note}</p>}</section>;
}



function Facts({ groups }) {
  return <div className="rv-details">{groups.map(([heading, rows]) => <div key={heading}>
    <h3>{heading}</h3>
    <dl>{rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v ?? '—'}</dd></div>)}</dl></div>)}</div>;
}

// Everything on the file, one section under another: the customer and the loan,
// what was verified, the risk, the price and acceptance, the competitor quote and
// the documents.
function ApplicationTab({ a, request, final, notify }) {
  const [stored, setStored] = useState(undefined);
  useEffect(() => {
    api('/applications').then(({ applications }) => setStored(applications.some(x => x.loan_id === a.loan_id))).catch(() => setStored(false));
  }, [a.loan_id]);
  const sc = a.sales_context || {}, r = final?.risk || {}, p = final?.policy || {}, env = final?.envelope || {};
  const loanNow = a.sanctioned_loan_amount || a.requested_loan_amount;
  const value = a.appraised_property_value || a.estimated_property_value;
  const changed = request.loan_amount && loanNow && Math.abs(loanNow - request.loan_amount) > 1;
  return <div className="rv-stack">
    {changed && <InfoBox warning>The file was changed after this request was raised: the loan is now {money(loanNow)}; the request was priced on {money(request.loan_amount)}.</InfoBox>}
    <Fold title="Applicant and loan" summary={[a.name, SEGMENT[a.segment], a.city, `loan ${money(loanNow)}`, loanNow && value ? `LTV ${Math.round((loanNow / value) * 100)}%` : null,
      a.new_to_credit || !(a.bureau_score > 0) ? 'new to credit' : `credit score ${a.bureau_score}`].filter(Boolean).join(' · ')}>
      <Facts groups={[
        ['Borrower', [['Name', a.name], ['Mobile', a.mobile || '—'], ['Email', a.email || '—'], ['Age', a.applicant_age ? `${a.applicant_age} years` : '—'],
          ['Employment', SEGMENT[a.segment]], ['City', a.city]]],
        ['Income and credit', [['Declared monthly income', money(a.declared_monthly_income)], ['Verified monthly income', shown(a.verified_monthly_income, money)],
          ['Monthly obligations', money(a.declared_obligations)], ['Credit score', a.new_to_credit || !(a.bureau_score > 0) ? 'New to credit' : a.bureau_score],
          ['Credit history', shown(a.bureau_vintage_months, v => `${v} months`)], ['Banking behaviour', shown(a.banking_behaviour_score, v => `${v} / 100`)],
          ['Missed payments, last 24 months', shown(a.delinquency_count_24m)], ['Credit deviations', shown(a.deviation_count)]]],
        ['Loan and property', [['Requested loan', money(a.requested_loan_amount)], ['Assessed loan', shown(a.sanctioned_loan_amount, money)],
          ['Tenure', `${a.tenure_years || a.requested_tenure_years} years`], ['Property type', PROPERTY[a.property_type] || titleCase(a.property_type)],
          ['Property value', money(a.estimated_property_value)], ['Appraised value', shown(a.appraised_property_value, money)],
          ['Loan to value', loanNow && value ? `${Math.round((loanNow / value) * 100)}%` : '—'], ['Legal check', flag(a.legal_flag)], ['Technical check', flag(a.technical_flag)]]],
        ['Application', [['Application ID', a.loan_id], ['Channel', CHANNEL[a.channel] || titleCase(a.channel)],
          ['Competitor quote', sc.competitor_rate ? `${pct(sc.competitor_rate)}${sc.competitor_name ? ` · ${sc.competitor_name}` : ''}` : 'None'],
          ['Sanction letter', sc.competitor_rate ? (sc.competitor_evidence_verified ? 'Seen' : 'Not yet seen') : '—'],
          ['Customer negotiating', sc.customer_negotiating == null ? '—' : sc.customer_negotiating ? 'Yes' : 'No'],
          ['Existing customer', sc.existing_relationship == null ? '—' : sc.existing_relationship ? 'Yes' : 'No']]],
      ]}/>
    </Fold>
    <Fold title="Risk and underwriting" summary={final ? [`Risk grade ${r.risk_grade}`, r.pd != null ? `about ${(r.pd * 100).toFixed(1)}% default a year` : null,
      `break-even ${pct(r.breakeven_rate)}`, p.allowed === false ? 'outside policy' : 'inside policy'].filter(Boolean).join(' · ') : 'The file is being priced.'}>
      {final && <Facts groups={[
        ['Credit risk', [['Risk grade', r.risk_grade], ['Default each year', r.pd != null ? `About ${(r.pd * 100).toFixed(1)}%` : '—'],
          ['Loss if default', r.lgd != null ? `${Math.round(r.lgd * 100)}%` : '—'], ['Expected loss a year', r.expected_loss_pct != null ? pct(r.expected_loss_pct) : '—'],
          ['Break-even rate', pct(r.breakeven_rate)]]],
        ['Policy', [['Permissible range', env.min_rate != null ? `${pct(env.min_rate)}–${pct(env.max_rate)}` : '—'], ['Card rate', pct(env.card_rate)],
          ['Required authority', titleCase(p.required_approval) || '—'], ['Policy check', p.allowed === false ? 'Outside policy' : 'Inside policy'],
          ['Fair-pricing review', final.explanation?.fair_pricing_review?.flagged ? 'Flagged' : 'Passed'], ['Policy version', <PolicyLink version={p.policy_version}/>]]],
      ]}/>}
      {(p.violations || []).map(v => <InfoBox warning key={v}>{v}</InfoBox>)}
      <p className="rv-note">From the pricing of this file; the policy limits come from the pricing policy.</p>
    </Fold>
    <Fold title="Documents" summary={stored === false ? 'No uploaded documents' : 'What the customer uploaded'}>{stored ? <ApplicationDocuments loanId={a.loan_id} notify={notify}/>
      : <p className="rv-muted">{stored === false ? 'This file was priced without an online application, so it has no uploaded documents.' : 'Loading…'}</p>}</Fold>
  </div>;
}
