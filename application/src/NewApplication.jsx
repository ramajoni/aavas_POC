import React, { useState } from 'react';
import { ArrowRight, FloppyDisk, Sparkle } from '@phosphor-icons/react';
import { api, emi, emptyApplicant, money, newId, payload, titleCase } from './api';
import { Button, InfoBox } from './components';
import { useLenders } from './CompetitorCheck';
import { CustomerLoginNotice } from './CustomerLogin';
import './price-offer.css';
import './new-application.css';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const lakh = n => n == null ? '—' : `₹${(n / 100000).toFixed(1)} lakh`;
const SEGMENTS = [['salaried_formal', 'Salaried (formal)'], ['salaried_informal', 'Salaried (informal)'],
  ['self_employed_professional', 'Self-employed professional'], ['self_employed_non_professional', 'Self-employed']];
const PROPERTY = [['resale_flat', 'Resale flat'], ['builder_flat', 'Builder flat'], ['self_construction', 'Self construction'],
  ['plot_plus_construction', 'Plot + construction']];
const CHANNELS = [['branch_walkin', 'Branch walk-in'], ['dsa', 'DSA'], ['connector', 'Connector'], ['digital', 'Digital']];
const CITIES = ['Jaipur', 'Udaipur', 'Ahmedabad', 'Indore', 'Lucknow', 'Kota', 'Pune', 'Bhopal'];

// Every field starts empty: only what the customer tells the RM goes on the file.
const blank = () => ({ name: '', mobile: '', email: '', segment: '', city: '', channel: '',
  applicant_age: '', declared_monthly_income: '', declared_obligations: '', bureau_score: '', new_to_credit: false,
  requested_loan_amount: '', estimated_property_value: '', requested_tenure_years: '', property_type: '',
  competitor_rate: '', competitor_name: '', competitor_evidence: 'no', notes: '' });

// The application the pricing service and the file store expect, built from the form.
function toApplicant(f, loanId) {
  const base = emptyApplicant();
  const num = v => (v === '' || v == null ? null : Number(v));
  const loan = num(f.requested_loan_amount), value = num(f.estimated_property_value), income = num(f.declared_monthly_income);
  return {
    ...base, loan_id: loanId, name: f.name.trim(), mobile: f.mobile.replace(/\D/g, ''), email: f.email.trim(),
    segment: f.segment, city: f.city, channel: f.channel, property_type: f.property_type,
    applicant_age: num(f.applicant_age), declared_monthly_income: income, declared_obligations: num(f.declared_obligations),
    requested_loan_amount: loan, estimated_property_value: value, requested_tenure_years: num(f.requested_tenure_years),
    new_to_credit: f.new_to_credit, bureau_score: f.new_to_credit ? -1 : num(f.bureau_score),
    // Verified figures are entered after the documents are checked (Edit details);
    // until then pricing uses the declared ones and the file says they are unverified.
    sales_context: { ...base.sales_context, competitor_rate: num(f.competitor_rate),
      competitor_name: num(f.competitor_rate) && f.competitor_name ? f.competitor_name : null,
      competitor_evidence_verified: f.competitor_evidence === 'yes',
      ...(/negotiat|bargain|lower/i.test(f.notes) ? { customer_negotiating: true } : {}) },
    conversation_notes: f.notes.trim(),
  };
}

// Wireframe 02, "RM · Assisted application": enter the customer's details, see the
// indicative range and what the notes say, then save it to My applications.
export function NewApplication({ user, notify, navigate }) {
  const [f, setF] = useState(blank);
  const [loanId, setLoanId] = useState(newId);
  const [quote, setQuote] = useState(null);
  const [read, setRead] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [issued, setIssued] = useState(null);     // the customer's new login, shown once
  const lenders = useLenders();
  const set = key => e => {
    const value = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
    setF(v => ({ ...v, [key]: value })); setQuote(null); setRead(null); setError('');
  };

  // Nothing is assumed, so everything the price depends on has to be entered.
  const needed = [['name', 'the name'], ['segment', 'employment'], ['city', 'the city'], ['channel', 'the source'],
    ['applicant_age', 'age'], ['declared_monthly_income', 'income'], ['declared_obligations', 'monthly obligations (0 if none)'],
    ['requested_loan_amount', 'loan amount'], ['estimated_property_value', 'property value'],
    ['requested_tenure_years', 'tenure'], ['property_type', 'property type']];
  const gaps = [...needed.filter(([k]) => f[k] === '' || f[k] == null || String(f[k]).trim() === '').map(([, l]) => l),
    ...(/^[6-9]\d{9}$/.test(f.mobile.replace(/\D/g, '')) ? [] : ['a 10-digit mobile']),
    ...(f.new_to_credit || f.bureau_score ? [] : ['credit score (or tick new to credit)'])];
  const missing = gaps.length > 0;

  async function getRange(e) {
    e.preventDefault();
    if (missing) { setError(`Still needed: ${gaps.join(', ')}.`); return; }
    setBusy(true); setError('');
    const applicant = toApplicant(f, loanId);
    try {
      const [pre, notes] = await Promise.all([
        api('/pricing/pre-assessment', payload(applicant)),
        f.notes.trim() ? api('/notes/extract', { loan_id: loanId, notes: f.notes.split(/\n+/).filter(Boolean), sales_context: { competitor_rate: applicant.sales_context.competitor_rate, competitor_evidence_verified: applicant.sales_context.competitor_evidence_verified } }).catch(() => null) : Promise.resolve(null),
      ]);
      setQuote(pre); setRead(notes);
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  function done() {
    setIssued(null); setF(blank()); setLoanId(newId()); setQuote(null); setRead(null);
    navigate('applications');
  }

  async function save() {
    setBusy(true); setError('');
    try {
      const saved = await api('/applications', { loan_id: loanId, applicant: toApplicant(f, loanId) });
      notify(`${f.name.trim()}'s application ${loanId} is saved in My applications.`);
      if (saved.customer_login) { setIssued({ login: saved.customer_login, name: f.name.trim() }); return; }
      done();
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  const bullets = read ? [
    ...read.competitors.map(c => `${c.competitor} offer: ${pct(c.rate)} (${c.evidence === 'documented' ? 'documented' : 'unverified'})`),
    ...read.unattributed_rates.map(r => `Rate mentioned: ${pct(r.rate)}, lender not named (“${r.sentence}”)`),
    ...read.customer_asks.map(a => `Customer asks for ${pct(a.rate)}`),
    ...Object.entries(read.customer_signals).map(([k, v]) => `${titleCase(k.replace(/^customer_/, ''))}: ${typeof v.value === 'number' ? v.value : titleCase(String(v.value))}${v.quote ? ` (“${v.quote}”)` : ''}`),
    ...read.conflicts.map(c => c.message),
  ] : [];

  return <div className="price-offer new-app">
    {issued && <CustomerLoginNotice login={issued.login} name={issued.name} onClose={done}/>}
    <div className="po-head"><span className="po-crumbs">{user.name} · {user.title}</span><h1>New application</h1></div>
    <div className="po-layout">
      <form className="po-card" onSubmit={getRange} autoComplete="off">
        <fieldset><legend>Borrower</legend><div className="po-form">
          <label><span>Full name</span><input value={f.name} required autoComplete="off" onChange={set('name')}/></label>
          <label><span>Mobile</span><input value={f.mobile} inputMode="tel" placeholder="10 digits" required onChange={set('mobile')}/></label>
          <label><span>Email (optional)</span><input type="email" value={f.email} onChange={set('email')}/></label>
          <label><span>Employment</span><select value={f.segment} onChange={set('segment')}><option value="" disabled>Choose</option>{SEGMENTS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>
          <label><span>Monthly income (₹)</span><input type="number" min="1" value={f.declared_monthly_income} required onChange={set('declared_monthly_income')}/></label>
          <label><span>Monthly obligations (₹)</span><input type="number" min="0" value={f.declared_obligations} onChange={set('declared_obligations')}/></label>
          <label><span>Credit score</span><input type="number" min="300" max="900" value={f.bureau_score} disabled={f.new_to_credit} placeholder={f.new_to_credit ? 'New to credit' : '300–900'} onChange={set('bureau_score')}/></label>
          <label className="na-check"><input type="checkbox" checked={f.new_to_credit} onChange={set('new_to_credit')}/> New to credit (no score)</label>
          <label><span>Age</span><input type="number" min="18" max="75" value={f.applicant_age} onChange={set('applicant_age')}/></label>
          <label><span>City</span><select value={f.city} onChange={set('city')}><option value="" disabled>Choose a city</option>{CITIES.map(c => <option key={c}>{c}</option>)}</select></label>
        </div></fieldset>
        <fieldset><legend>Loan and property</legend><div className="po-form">
          <label><span>Loan amount (₹)</span><input type="number" min="1" value={f.requested_loan_amount} required onChange={set('requested_loan_amount')}/></label>
          <label><span>Property value (₹)</span><input type="number" min="1" value={f.estimated_property_value} required onChange={set('estimated_property_value')}/></label>
          <label><span>Tenure (years)</span><input type="number" min="1" max="30" value={f.requested_tenure_years} onChange={set('requested_tenure_years')}/></label>
          <label><span>Property type</span><select value={f.property_type} onChange={set('property_type')}><option value="" disabled>Choose</option>{PROPERTY.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>
          <label><span>Source</span><select value={f.channel} onChange={set('channel')}><option value="" disabled>Choose</option>{CHANNELS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>
        </div></fieldset>
        <fieldset><legend>Market context</legend><div className="po-form">
          <label><span>Competitor lender</span><select value={f.competitor_name} onChange={set('competitor_name')}>
            <option value="">Not named</option>{lenders.map(l => <option key={l.name} value={l.name}>{l.name}</option>)}</select></label>
          <label><span>Competitor quote (%)</span><input type="number" step="0.01" min="1" max="30" value={f.competitor_rate} placeholder="If any" onChange={set('competitor_rate')}/></label>
          <label><span>Evidence</span><select value={f.competitor_evidence} onChange={set('competitor_evidence')}><option value="no">Not yet seen</option><option value="yes">Sanction letter seen</option></select></label>
          <label className="na-wide"><span>Notes from the conversation</span><textarea rows="3" value={f.notes} placeholder="e.g. Customer has an offer at 11.5% from another HFC. Keen to close this month." onChange={set('notes')}/></label>
        </div></fieldset>
        {error && <p className="po-error" role="alert">{error}</p>}
        <div className="po-actions"><Button primary type="submit" busy={busy}>Get indicative range <ArrowRight size={17}/></Button></div>
      </form>

      <aside className="na-side">
        <section className="po-card">
          <span className="po-label">Indicative range</span>
          {quote ? <>
            <strong className="na-range">{pct(quote.band.min_rate)}–{pct(quote.band.max_rate)}</strong>
            <span>Loans up to {lakh(quote.eligible_amount?.max)} · EMI from {money(emi(Number(f.requested_loan_amount), quote.band.min_rate, Number(f.requested_tenure_years)))}</span>
            <small className="muted">Confidence: {quote.confidence_label || '—'}, documents not yet verified</small>
          </> : <small className="muted">Fill in the details and press “Get indicative range”.</small>}
        </section>
        {quote && f.notes.trim() && <section className="po-card">
          <span className="po-label strong">Read from your notes</span>
          {read ? (bullets.length ? <ul className="na-list">{bullets.map(b => <li key={b}>{b}</li>)}</ul> : <small className="muted">Nothing picked up from these notes.</small>)
            : <small className="muted">The notes could not be read just now.</small>}
          <small className="muted">Please confirm before saving.</small>
        </section>}
        <Button primary={Boolean(quote)} busy={busy} disabled={!quote} onClick={save}><FloppyDisk size={17}/> Save application</Button>
        {!quote && <small className="muted na-hint"><Sparkle size={13}/> Get the indicative range first; it is saved with the application.</small>}
      </aside>
    </div>
  </div>;
}
