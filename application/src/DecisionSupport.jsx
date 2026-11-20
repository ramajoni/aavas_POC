import React, { useEffect, useRef, useState } from 'react';
import { CaretDown, CaretRight, CheckCircle, CircleNotch, Info, MagnifyingGlass, WarningCircle, XCircle } from '@phosphor-icons/react';
import { apiOnce, money } from './api';
import { Button, InfoBox } from './components';
import { PolicyLink } from './PolicyViewer';

const pct = r => r == null || Number.isNaN(r) ? '—' : `${Number(r).toFixed(2)}%`;
const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
// Each factor's signal, read against the RM's ask. "Positive" supports giving the
// lower rate; "Negative" argues against it; "OK" means only that nothing on that
// count stands in the way - a file can be OK on every count and still not need the
// discount. "Verify" is a fact to check before approving; "Watchful" needs a second look.
const SIGNAL = { for: 'Positive', clear: 'OK', verify: 'Verify', caution: 'Watchful', against: 'Negative', neutral: 'Info' };
const SIGNAL_ICON = { for: CheckCircle, clear: CheckCircle, verify: MagnifyingGlass, caution: WarningCircle, against: XCircle, neutral: Info };
const FROM = { good: 'clear', warn: 'caution', bad: 'against', ask: 'neutral', info: 'neutral', for: 'for', verify: 'verify' };
const ORDER = ['for', 'clear', 'verify', 'caution', 'against', 'neutral'];
// Where each Details link leads, named so the manager knows before clicking.
const WHERE = { price: 'Pricing intelligence', 'application/competitors': 'Application · Competitors', 'application/notes': 'Application · Notes',
  'application/details': 'Application', 'branch/summary': 'Branch details' };
// The filter over the rows: everything, or only what needs a look.
const FILTERS = [['all', 'All', () => true], ['attention', 'Needs attention', i => ['against', 'caution', 'verify'].includes(i.signal)],
  ['against', 'Negative for the ask', i => i.signal === 'against'], ['verify', 'To verify', i => i.signal === 'verify'],
  ['fine', 'Positive or OK', i => i.signal === 'for' || i.signal === 'clear']];

function emi(amount, rate, years) { const r = rate / 1200, n = years * 12; return r ? amount * r * (1 + r) ** n / ((1 + r) ** n - 1) : amount / n; }
// The annual rate a borrower really pays once upfront charges are counted: the rate at
// which the EMIs repay only what they actually received.
function effectiveRate(amount, rate, years, upfront) {
  const pay = emi(amount, rate, years), n = years * 12, net = amount - upfront;
  if (!(net > 0)) return null;
  let lo = 0, hi = 0.05;
  for (let i = 0; i < 80; i++) { const mid = (lo + hi) / 2; const pv = mid ? pay * (1 - (1 + mid) ** -n) / mid : pay * n; if (pv > net) lo = mid; else hi = mid; }
  return ((lo + hi) / 2) * 1200;
}

// The competitor's offer against ours on the same terms: headline rate, fees and
// charges together, as the yearly cost the customer really bears.
function LikeForLike({ c, a, ask, rec }) {
  const sc = a.sales_context || {};
  const [their, setTheir] = useState({ rate: sc.competitor_rate || '', fee: '', charges: '', years: c.tenure_years });
  const loan = c.loan_amount, ourFee = loan * (c.fees.our_processing_fee_pct || 0) / 100;
  const num = v => (v === '' || v == null ? 0 : Number(v));
  const theirFee = loan * num(their.fee) / 100 + num(their.charges);
  const rows = their.rate ? [
    [`${sc.competitor_name || 'Competitor'}`, num(their.rate), num(their.years) || c.tenure_years, theirFee],
    ['Our ask', ask, c.tenure_years, ourFee], ['Our system recommendation', rec, c.tenure_years, ourFee],
    ['Our system recommendation, fee waived', rec, c.tenure_years, 0],
  ].map(([label, rate, years, upfront]) => ({ label, rate, upfront, emi: emi(loan, rate, years), eff: effectiveRate(loan, rate, years, upfront) })) : [];
  const set = k => e => setTheir(t => ({ ...t, [k]: e.target.value }));
  return <div className="ck-calc">
    <div className="ck-inputs">
      <label><span>Their rate (%)</span><input type="number" step="0.01" value={their.rate} onChange={set('rate')}/></label>
      <label><span>Their processing fee (% of loan)</span><input type="number" step="0.05" placeholder="e.g. 0.50" value={their.fee} onChange={set('fee')}/></label>
      <label><span>Their other charges, insurance (₹)</span><input type="number" step="1000" placeholder="e.g. 25000" value={their.charges} onChange={set('charges')}/></label>
      <label><span>Their tenure (years)</span><input type="number" value={their.years} onChange={set('years')}/></label>
    </div>
    {rows.length > 0 && <table className="sc-table ck-table"><thead><tr><th>Offer</th><th>Headline rate</th><th>Paid upfront</th><th>EMI</th><th>Real yearly cost</th></tr></thead>
      <tbody>{rows.map(r => <tr key={r.label}><td><b>{r.label}</b></td><td>{pct(r.rate)}</td><td>{money(Math.round(r.upfront))}</td><td>{money(Math.round(r.emi))}</td><td><b>{pct(r.eff)}</b></td></tr>)}</tbody></table>}
    <p className="rv-note">Real yearly cost counts the fee and charges paid upfront. Enter what the customer was actually offered; if their fees are higher, their headline rate is not the whole story.</p>
  </div>;
}

function Factor({ id, signal, label, value, reading, target, go, flash, toggle, children }) {
  const Icon = SIGNAL_ICON[signal];
  const [open, setOpen] = useState(false);
  const where = target && (WHERE[target.join('/')] || WHERE[target[0]]);
  return <div id={id} className={`an-row ${signal}${flash ? ' flash' : ''}`}>
    <div className="an-label">{label}</div>
    <div className="an-value">{toggle ? <button type="button" className="an-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? toggle.hide : toggle.show}<CaretDown size={13}/></button> : value}</div>
    <div className="an-reading">{reading}{where && <button type="button" className="link-button an-go" onClick={() => go(...target)}>See in {where} <CaretRight size={12}/></button>}</div>
    <div className="an-signal"><span><Icon size={14} weight="fill"/>{SIGNAL[signal]}</span></div>
    {children && (!toggle || open) && <div className="an-extra">{children}</div>}
  </div>;
}

function Section({ id, title, items, children }) {
  const tally = items.reduce((t, i) => ({ ...t, [i.signal]: (t[i.signal] || 0) + 1 }), {});
  return <section id={id} className="rv-card an-section">
    <div className="an-section-head"><h2>{title}</h2>
      <div className="an-chips">{ORDER.filter(k => tally[k] && k !== 'neutral').map(k => <span key={k} className={k}>{tally[k]} {SIGNAL[k].toLowerCase()}</span>)}</div></div>
    <div className="an-table">
      <div className="an-head"><span>Factor</span><span>Figure</span><span>What it means</span><span>Signal</span></div>{children}</div></section>;
}

// Decision memory: how this manager decided requests like this one, and what the
// customers did. It sits beside the suggestion and never changes it.
const DID = { approve: 'Approved the ask', counter: 'Approved a smaller cut', hold: 'Held the rate', uplift: 'Priced higher' };
const SUGGESTED = { approve: 'Approve', counter: 'Counter-offer', hold: 'Hold', uplift: 'Uplift' };
const OUTCOME = { accepted: 'good', not_accepted: 'bad' };
const day = s => s ? new Date(s).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : '—';

function Recalled({ cases, others }) {
  return <div className="mem-scroll"><table className="sc-table ck-table mem-table">
    <thead><tr><th>Decided</th>{others && <th>By</th>}<th>The ask</th><th>Suggested</th><th>Decision</th><th>What happened</th><th>Why it is like this one</th></tr></thead>
    <tbody>{cases.map(c => <tr key={c.episode_id}>
      <td>{day(c.decided_at)}</td>{others && <td>{c.manager}</td>}
      <td>{pct(c.requested_rate)}<small>against {pct(c.recommended_rate)}</small></td>
      <td>{c.suggested?.action ? `${SUGGESTED[c.suggested.action] || c.suggested.action}${c.suggested.rate != null && c.suggested.action !== 'hold' ? ` ${pct(c.suggested.rate)}` : ''}` : <small>Not recorded</small>}</td>
      <td><b>{DID[c.decided.action] || c.decided.action}</b>{c.decided.action !== 'hold' && c.decided.rate != null ? ` ${pct(c.decided.rate)}` : ''}{c.note && <small>“{c.note}”</small>}</td>
      <td className={`mem-out ${OUTCOME[c.outcome?.state] || ''}`}>{c.outcome?.text || '—'}</td>
      <td><small>{(c.matched || []).join(', ') || '—'}</small></td></tr>)}</tbody></table></div>;
}

function Memory({ m }) {
  if (!m) return null;
  if (!m.available) return <section className="rv-card an-memory"><h2>Your decision memory</h2><p className="rv-muted">{m.reason}</p></section>;
  const mine = m.yours?.cases || [], theirs = m.others?.cases || [], lines = m.profile?.lines || [];
  return <section className="rv-card an-memory" aria-labelledby="mem-title">
    <div className="an-section-head"><h2 id="mem-title">Your decision memory</h2>
      <small className="mem-count">{m.your_episodes} of your decisions remembered · {m.episodes} in all</small></div>
    {m.consistency && <p className={`mem-consistency ${m.consistency.tone}`}>{m.consistency.text}</p>}
    {lines.length > 0 && <ul className="mem-lines">{lines.map(l => <li key={l}>{l}</li>)}</ul>}
    <h3>Your decisions most like this one</h3>
    {mine.length ? <Recalled cases={mine}/> : <p className="rv-muted">{m.yours?.empty}</p>}
    {theirs.length > 0 && <details className="mem-others"><summary>Other managers’ decisions like this one ({theirs.length})</summary><Recalled cases={theirs} others/></details>}
    <p className="rv-note">{m.method}</p>
  </section>;
}

export function DecisionSupport({ request, a, final, go, onUse }) {
  const id = encodeURIComponent(request.request_id);
  const [d, setD] = useState({});
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('all');
  const [flash, setFlash] = useState(null);
  const top = useRef(null);
  const [summaryInView, setSummaryInView] = useState(true);
  useEffect(() => {
    if (!top.current || typeof IntersectionObserver === 'undefined') return undefined;
    const seen = new IntersectionObserver(([e]) => setSummaryInView(e.isIntersecting), { threshold: 0 });
    seen.observe(top.current);
    return () => seen.disconnect();
  }, [d.c]);
  useEffect(() => {
    let live = true;
    const put = key => v => live && setD(x => ({ ...x, [key]: v }));
    apiOnce(`/pricing/rate-requests/${id}/checklist`).then(put('c')).catch(e => live && setError(e.message));
    apiOnce(`/pricing/rate-requests/${id}/suggestion`).then(put('s')).catch(() => put('s')(null));
    apiOnce(`/pricing/rate-requests/${id}/briefing`).then(put('b')).catch(() => put('b')(null));
    apiOnce(`/pricing/rate-requests/${id}/history`).then(put('h')).catch(() => put('h')(null));
    apiOnce(`/pricing/rate-requests/${id}/memory`).then(put('m')).catch(() => put('m')(null));
    return () => { live = false; };
  }, [id]);
  if (error) return <InfoBox warning><span role="alert">{error}</span></InfoBox>;
  const { c, b, h } = d;
  if (!c) return <section className="rv-card"><p className="rv-muted"><CircleNotch className="spin" size={16}/> Analysing the request…</p></section>;
  const suggestion = d.s || c.suggestion;
  // The signal on each row is decided once, on the server, so the screen shows what memory records.
  const server = Object.fromEntries((d.m?.factors || []).map(f => [f.key, f.signal]));

  const x = request.analysis || {}, ask = request.requested_rate, rec = request.recommended_rate;
  const sc = a.sales_context || {}, r = final?.risk || {}, pol = final?.policy || {};
  const evidenced = request.evidence_verified || sc.competitor_evidence_verified;
  const gain = x.acceptance_at_requested != null && x.acceptance_at_recommended != null
    ? Math.round(x.acceptance_at_requested * 100) - Math.round(x.acceptance_at_recommended * 100) : null;
  const hints = c.reason_hints || {}, other = (hints.found || []).filter(k => k !== 'rate');
  const deepest = b?.overrides?.similar_here?.deepest_bps, depthBps = Math.round((rec - ask) * 100);
  const breakeven = r.breakeven_rate ?? x.breakeven_rate;
  const margin = breakeven != null ? ask - breakeven : null;
  const pay = b?.signals?.find(t => t.topic === 'did_it_pay');
  const rm = b?.performance?.rm, month = c.this_month || {}, mine = month.rm_disbursed;
  const prec = c.precedent || {}, lender = h?.competitors?.this_lender;
  const checksClear = [a.legal_flag, a.technical_flag].every(v => v === 0 || v === '0');
  const ltv = a.estimated_property_value ? Math.round(((a.sanctioned_loan_amount || a.requested_loan_amount) / (a.appraised_property_value || a.estimated_property_value)) * 100) : null;

  // "What it means" in plain words: what the number says, then what to do about it.
  const GRADE = { 'A+': 'Very strong credit', A: 'Strong credit', B: 'Average credit', C: 'Weak credit', D: 'High-risk credit: no offer under policy' };
  const cutWord = g => g >= 5 ? 'That is a big gain.' : g >= 2 ? 'That is a small gain.' : 'That is a very small gain.';
  const items = [
    // Need for the discount
    ['need', 'competitor', sc.competitor_rate ? (evidenced ? 'for' : 'verify') : 'info', 'Other lender’s offer',
      sc.competitor_rate ? `${sc.competitor_name || 'A lender'} ${pct(sc.competitor_rate)}` : 'None',
      sc.competitor_rate
        ? <>{evidenced ? `The customer has a real offer from ${sc.competitor_name || 'another lender'} (letter seen). A lower rate may be needed to keep them.`
            : `The customer says ${sc.competitor_name || 'another lender'} offered ${pct(sc.competitor_rate)}, but we have not seen the letter. Ask for it before approving.`}
          {lender ? ` We still win ${share(lender.we_won)} of customers here who name ${lender.lender}.` : ''}</>
        : 'No other lender’s offer. The customer is only asking for a lower rate.', ['application', 'competitors']],
    ['need', 'response', gain == null ? 'info' : gain >= 5 ? 'for' : 'bad', 'Effect of the cut',
      gain == null ? '—' : gain <= 0 ? 'No gain' : `+${gain} pt${gain === 1 ? '' : 's'}`,
      gain == null ? 'Not available.' : <>If you approve, the chance the customer accepts goes from {share(x.acceptance_at_recommended)} to {share(x.acceptance_at_requested)}. <b>{cutWord(gain)}</b></>, ['price']],
    ['need', 'negotiation', sc.customer_negotiating ? 'warn' : 'info', 'Bargaining',
      sc.customer_negotiating ? `Yes${sc.prior_negotiation_rounds ? `, ${sc.prior_negotiation_rounds} round${sc.prior_negotiation_rounds === 1 ? '' : 's'}` : ''}` : sc.customer_negotiating === false ? 'No' : 'Not recorded',
      sc.customer_negotiating ? 'The customer is bargaining. Check if a lower EMI or a fee waiver would close the deal instead of a rate cut.'
        : sc.customer_negotiating === false ? 'The customer is not bargaining.' : 'The RM has not recorded this.'],
    ['need', 'reason', other.length ? 'warn' : 'info', 'What the customer cares about',
      other.length ? hints.labels.filter(l => l !== 'the rate itself').join(', ') : hints.text ? 'The rate' : 'Not recorded',
      other.length ? 'The notes show the customer cares about more than the rate. One of the options below may work without a big cut.'
        : hints.text ? 'The notes show the customer cares only about the rate.' : 'The RM has not added notes from the conversation.', ['application', 'notes']],
    ['need', 'likeforlike', 'info', 'Compare full offers', 'Calculator', 'Compare their total cost (rate plus fees) with ours.', null,
      <LikeForLike c={c} a={a} ask={ask} rec={rec}/>],
    // Cost to the bank
    ['cost', 'interest', (x.cost_full_term || 0) > 500000 ? 'bad' : 'warn', 'Interest the bank loses', money(x.cost_full_term ?? x.cost_over_life),
      <>If you approve, the bank earns this much less over {x.tenure_years || c.tenure_years} years{x.emi_recommended != null && x.emi_requested != null ? <>. The customer’s EMI drops by {money(x.emi_recommended - x.emi_requested)} a month</> : null}.</>],
    ['cost', 'depth', deepest != null && depthBps > deepest ? 'bad' : 'good', 'Size of the cut', `${pct(depthBps / 100)} lower`,
      deepest == null ? 'Lower than the system recommendation.'
        : depthBps > deepest ? <>Bigger than any cut given here before for this reason (largest so far {pct(deepest / 100)}). <b>Approving sets a new record.</b></>
        : `Cuts this size have been given here before (largest so far ${pct(deepest / 100)}).`, ['branch', 'summary']],
    ['cost', 'margin', margin == null ? 'info' : margin >= 1 ? 'good' : margin >= 0.5 ? 'warn' : 'bad', 'Profit margin left',
      margin == null ? '—' : pct(margin), margin == null ? 'Not available.' : <>The bank breaks even at {pct(breakeven)}. At {pct(ask)} it still makes {pct(margin)} a year. <b>{margin >= 1 ? 'Healthy.' : margin >= 0.5 ? 'Thin.' : 'Close to a loss.'}</b></>, ['price']],
    ['cost', 'credit', ['C', 'D'].includes(r.risk_grade) ? 'bad' : !checksClear ? 'verify' : ['A+', 'A'].includes(r.risk_grade) ? 'good' : 'warn', 'Credit quality', r.risk_grade ? `Grade ${r.risk_grade}` : '—',
      <>{GRADE[r.risk_grade] || 'Credit grade not known'}. {a.new_to_credit || !(a.bureau_score > 0) ? 'New to credit' : `Credit score ${a.bureau_score}`}{ltv ? `, loan is ${ltv}% of the property value` : ''}. {checksClear ? 'Legal and technical checks are done.' : <b>Legal or technical check not done yet: verify before approving.</b>}</>, ['application', 'details']],
    ['cost', 'alternatives', 'info', 'Other ways to close the deal', `${(c.alternatives?.options || []).length} options`, 'Options that may work instead of a big rate cut, and what each costs the bank.', null,
      <table className="sc-table ck-table"><thead><tr><th>Option</th><th>Rate</th><th>Tenure</th><th>EMI</th><th>EMI against the ask</th><th>For the bank, over the loan</th></tr></thead>
        <tbody>{(c.alternatives?.options || []).map(o => <tr key={o.key}><td><b>{o.label}</b><small>{o.note}{o.key === 'tenure' && o.age_at_end ? ` The customer would be about ${o.age_at_end}.` : ''}</small></td>
          <td>{pct(o.rate)}</td><td>{o.tenure_years} years</td><td>{money(o.emi)}</td><td>{o.emi_vs_ask === 0 ? 'Same' : `${o.emi_vs_ask > 0 ? '+' : '−'}${money(Math.abs(o.emi_vs_ask))}`}</td>
          <td className={o.bank_change_inr < 0 ? 'neg' : 'pos'}>{o.bank_change_inr < 0 ? `Gives up ${money(-o.bank_change_inr)}` : `Earns ${money(o.bank_change_inr)} more`}</td></tr>)}</tbody></table>],
    // Branch context and precedent
    ['branch', 'policy', x.below_policy_floor || pol.allowed === false ? 'bad' : 'good', 'Policy',
      x.below_policy_floor ? 'Below the lowest rate' : pol.allowed === false ? 'Outside policy' : 'Within policy',
      <>{x.below_policy_floor ? <b>{pct(ask)} is below {pct(x.policy_floor)}, the lowest rate policy allows. </b> : pol.allowed === false ? 'This file is outside policy. ' : 'This rate is allowed by policy. '}
        Fair pricing check {final?.explanation?.fair_pricing_review?.flagged ? 'needs a review' : 'passed'}. Policy <PolicyLink version={pol.policy_version}/>.{!evidenced && sc.competitor_rate ? ' If you approve, add the condition: subject to the sanction letter.' : ''}</>],
    ['branch', 'precedent', prec.files ? (prec.share_at_or_below < 0.1 ? 'bad' : prec.share_at_or_below < 0.25 ? 'warn' : 'good') : 'info', 'Rates on similar loans',
      prec.files ? `${share(prec.share_at_or_below)} got this rate or lower` : '—',
      prec.files ? <>Of {prec.files} similar loans {prec.scope.includes('every branch') ? 'across all branches' : 'in this branch'}, {share(prec.share_at_or_below)} got {pct(ask)} or lower. The usual rate is {pct(prec.median_rate)}.{prec.share_at_or_below < 0.1 ? <b> Approving sets a new low that others will ask for.</b> : null}</> : 'No similar loans to compare.'],
    ['branch', 'track', pay ? ({ good: 'for', bad: 'bad' }[pay.tone] || 'warn') : 'info', 'Past cuts in this branch', pay ? pay.verdict : 'Too few to judge',
      pay ? pay.text : 'Too few similar cuts in this branch to tell whether they work.', ['branch', 'summary']],
    ['branch', 'month', (month.branch?.approved || 0) >= 3 ? 'warn' : 'good', 'Cuts approved this month', month.branch ? `${month.branch.approved} of ${month.branch.asked}` : '—',
      month.branch ? `${month.branch.approved} of ${month.branch.asked} cut requests were approved in ${c.city || 'the branch'} this month${month.branch.avg_approved_cut_bps != null ? `, ${pct(month.branch.avg_approved_cut_bps / 100)} on average` : ''}. ${month.branch.pending} still waiting.` : ''],
    ['branch', 'target', mine ? (mine.on_pace ? 'good' : 'warn') : 'info', 'RM’s monthly target', mine ? `${share(mine.pct_of_target)} done` : '—',
      mine ? <>{mine.name} has done {share(mine.pct_of_target)} of this month’s target ({money(mine.disbursed_inr)} of {money(mine.target_inr)}). {mine.on_pace ? 'On track.' : <b>Behind target: the RM may be pushing for cuts to close deals.</b>}</> : 'No target set.'],
    ['branch', 'rm', rm && rm.requests > 3 && rm.approval_rate > 0.8 ? 'warn' : 'info', 'RM’s past requests', rm ? `${rm.requests} asked · ${share(rm.approval_rate)} approved` : '—',
      rm ? `The RM usually asks for ${pct((rm.avg_requested_discount_bps || 0) / 100)} below the system recommendation. ${rm.offers_accepted} of ${rm.offers_sent} offers were taken by customers.` : 'No past requests.', ['branch', 'summary']],
    ['branch', 'relationship', (c.relationship || []).includes('Existing customer') ? 'for' : 'info', 'Relationship', (c.relationship || [])[0] || '—',
      (c.relationship || []).includes('Existing customer') ? 'Already an Aavas customer: worth keeping.' : (c.relationship || []).slice(1).join(' · ') || 'A new customer with no other link to Aavas.'],
  ].map(([group, key, status, label, value, reading, target, extra]) => ({ group, key, signal: server[key] || FROM[status], label, value, reading, target, extra }));

  const tally = items.reduce((t, i) => ({ ...t, [i.signal]: (t[i.signal] || 0) + 1 }), {});
  const of = k => items.filter(i => i.signal === k);
  const against = of('against'), favour = of('for'), caution = of('caution'), toVerify = of('verify');
  const lean = against.length > favour.length ? 'against' : favour.length > against.length && caution.length <= favour.length ? 'for' : 'balanced';
  const headline = lean === 'for' ? 'A good case for the discount.' : lean === 'balanced' ? 'A mixed case for the discount.' : 'Not a strong case for the discount.';
  const sections = [['need', 'Need for the discount'], ['cost', 'Cost to the bank'], ['branch', 'Branch context and precedent']];
  const keep = (FILTERS.find(f => f[0] === filter) || FILTERS[0])[2];
  // A factor named in the summary takes the manager straight to its row.
  const jump = key => { setFilter('all'); setFlash(key); setTimeout(() => document.getElementById(`an-${key}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 30); setTimeout(() => setFlash(null), 1800); };
  const names = list => list.length ? list.map((i, n) => <React.Fragment key={i.key}>{n ? ', ' : ''}<button type="button" className="an-name" onClick={() => jump(i.key)}>{i.label}</button></React.Fragment>) : 'nothing';
  const figures = [
    ['The ask', `${pct(ask)}`, `against the system recommendation of ${pct(rec)} (${pct(rec - ask)} lower)`],
    ['Acceptance', gain == null ? '—' : `${gain > 0 ? '+' : gain < 0 ? '−' : ''}${Math.abs(gain)} pt${Math.abs(gain) === 1 ? '' : 's'}`, `${share(x.acceptance_at_recommended)} → ${share(x.acceptance_at_requested)} if approved`],
    ['Interest given up', money(x.cost_full_term ?? x.cost_over_life), `over the full ${x.tenure_years || c.tenure_years}-year loan`],
    ['Margin left', margin == null ? '—' : pct(margin), `above break-even of ${pct(breakeven)} at the ask`],
  ];
  return <div className="rv-stack an">
    <section ref={top} className={`rv-card an-summary ${lean}`}>
      <div className="an-summary-head">
        <div><h2>Decision support</h2><p className="an-headline">{headline}</p></div>
        {suggestion && <div className="an-suggest"><small>Suggested decision</small><b>{suggestion.headline}</b>
          {onUse && <Button primary onClick={() => onUse(suggestion)}>Use this decision</Button>}</div>}
      </div>
      <div className="an-figures">{figures.map(([label, value, note]) => <div key={label}><small>{label}</small><b>{value}</b><span>{note}</span></div>)}</div>
      <ul className="an-points">
        <li className="for"><i/><b>Positive for the RM’s ask:</b> <span>{names(favour)}</span></li>
        <li className="against"><i/><b>Negative for the RM’s ask:</b> <span>{names(against)}</span></li>
        <li className="caution"><i/><b>Be watchful:</b> <span>{names(caution)}</span></li>
        {toVerify.length > 0 && <li className="verify"><i/><b>Verify before approving:</b> <span>{names(toVerify)}</span></li>}
      </ul>
      <div className="an-balance" role="img" aria-label={ORDER.map(k => `${tally[k] || 0} ${SIGNAL[k].toLowerCase()}`).join(', ')}>
        {ORDER.map(k => tally[k] ? <i key={k} className={k} style={{ flex: tally[k] }}/> : null)}</div>
      <div className="an-tally">{ORDER.map(k => <span key={k} className={k}><i/>{tally[k] || 0} {SIGNAL[k].toLowerCase()}</span>)}</div>
    </section>

    <Memory m={d.m?.memory}/>

    <div className="an-toolbar">
      <div className="an-filters" role="tablist" aria-label="Show factors">{FILTERS.map(([k, label, test]) =>
        <button key={k} type="button" role="tab" aria-selected={filter === k} className={filter === k ? 'active' : ''} onClick={() => setFilter(k)}>
          {label} <small>{items.filter(test).length}</small></button>)}</div>
      <nav className="an-jump" aria-label="Go to a section">{sections.map(([key, title]) =>
        <button key={key} type="button" className="link-button" onClick={() => document.getElementById(`an-sec-${key}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>{title}</button>)}</nav>
    </div>

    {sections.map(([key, title]) => { const rows = items.filter(i => i.group === key && keep(i)); return rows.length ? <Section key={key} id={`an-sec-${key}`} title={title} items={rows}>
      {rows.map(i => <Factor key={i.key} id={`an-${i.key}`} signal={i.signal} label={i.label} value={i.value} reading={i.reading} target={i.target} go={go}
        flash={flash === i.key} toggle={i.key === 'likeforlike' ? { show: 'Open calculator', hide: 'Hide calculator' }
          : i.key === 'alternatives' ? { show: `Show ${(c.alternatives?.options || []).length} options`, hide: 'Hide options' } : null}>{i.extra}</Factor>)}
    </Section> : null; })}
    {!sections.some(([key]) => items.some(i => i.group === key && keep(i))) && <section className="rv-card"><p className="rv-muted">No factor in this view.</p></section>}

    {/* The suggestion stays in reach once the summary has scrolled away. */}
    {suggestion && onUse && !summaryInView && <div className="an-dock" role="region" aria-label="Suggested decision">
      <span><small>Suggested decision</small><b>{suggestion.headline}</b></span>
      <Button primary onClick={() => onUse(suggestion)}>Use this decision</Button></div>}
  </div>;
}
