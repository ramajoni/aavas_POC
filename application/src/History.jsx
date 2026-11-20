import React, { useEffect, useState } from 'react';
import { CircleNotch } from '@phosphor-icons/react';
import { api, money, number, titleCase, apiOnce } from './api';
import { InfoBox } from './components';
import { CompetitorCheck } from './CompetitorCheck';
import { noRoom } from './Status';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const bps = v => v == null ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${(Math.abs(v) / 100).toFixed(2)}%`;
const SEGMENT = { salaried_formal: 'Salaried', salaried_informal: 'Salaried (informal)',
  self_employed_professional: 'Self-employed professional', self_employed_non_professional: 'Self-employed' };
const NOTE_TYPES = { enquiry: 'Enquiry', negotiation: 'Negotiation', follow_up: 'Follow-up' };

// One fetch serves every history sub-tab of the review.
export function useHistory(requestId, wanted) {
  const [h, setH] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { setH(null); setError(''); }, [requestId]);
  useEffect(() => {
    if (!wanted || h || error) return;
    apiOnce(`/pricing/rate-requests/${encodeURIComponent(requestId)}/history`).then(setH).catch(e => setError(e.message));
  }, [requestId, wanted, h, error]);
  return { h, error };
}

export function HistoryGate({ state, children }) {
  if (state.error) return <InfoBox warning><span role="alert">{state.error}</span></InfoBox>;
  if (!state.h) return <section className="rv-card"><p className="rv-muted"><CircleNotch className="spin" size={16}/> Reading the branch’s history…</p></section>;
  return children(state.h);
}

export function SubTabs({ tabs, value, onChange }) {
  return <nav className="rv-subtabs" aria-label="Sections">{tabs.map(([k, l]) =>
    <button key={k} type="button" className={value === k ? 'active' : ''} aria-pressed={value === k} onClick={() => onChange(k)}>{l}</button>)}</nav>;
}

function Card({ title, sub, children }) {
  return <section className="rv-card"><div><h2>{title}</h2>{sub && <p className="rv-muted hs-sub">{sub}</p>}</div>{children}</section>;
}

function UpsellRow({ label, p }) {
  return <tr><td><b>{label}</b><small>{number(p.files)} files</small></td><td>{share(p.upsell_share)}<small>{p.upsold} upsold</small></td>
    <td>{p.median_premium_bps == null ? '—' : `+${(p.median_premium_bps / 100).toFixed(2)}%`}</td>
    <td>{share(p.took_when_upsold)}</td><td>{share(p.took_at_recommendation)}</td>
    <td>{p.value_upsold_inr == null ? '—' : money(p.value_upsold_inr)}</td><td>{p.value_at_recommendation_inr == null ? '—' : money(p.value_at_recommendation_inr)}</td></tr>;
}

function UpsellTable({ rows }) {
  return <div className="table-scroll"><table className="sc-table hs-table">
    <thead><tr><th rowSpan="2"></th><th rowSpan="2">Priced above the system recommendation</th><th rowSpan="2">Typical upsell</th><th colSpan="2">Took the loan</th><th colSpan="2">Value per loan</th></tr>
      <tr><th>Upsold</th><th>At the system recommendation</th><th>Upsold</th><th>At the system recommendation</th></tr></thead>
    <tbody>{rows.map(([label, p]) => <UpsellRow key={label} label={label} p={p}/>)}</tbody></table></div>;
}

// ------------------------------------------------------------ this file

export function FileUpselling({ h, request }) {
  const x = request.analysis || {}, s = h.upselling.similar, city = h.where.city;
  const up = Math.round((request.recommended_rate - request.requested_rate) * 100);
  const roomUp = x.headroom_rate != null && x.headroom_rate - request.recommended_rate > 0.005;
  return <div className="rv-stack">
    <Card title="Upselling on this file" sub="Holding the system recommendation is the upsell over the RM’s ask; the headroom is how far similar customers still accept.">
      <div className="hs-facts">
        <div><small>RM’s ask</small><b>{pct(request.requested_rate)}</b><em>{share(x.acceptance_at_requested)} likely to accept</em></div>
        <div className="good"><small>System recommendation (upsell)</small><b>{pct(request.recommended_rate)}</b><em>+{up} bps · {share(x.acceptance_at_recommended)} likely to accept</em></div>
        <div className={roomUp ? 'good' : ''}><small>Upsell room</small><b>{roomUp ? `Up to ${pct(x.headroom_rate)}` : 'No room'}</b><em>{roomUp ? `+${Math.round((x.headroom_rate - request.recommended_rate) * 100)} bps; 70% of customers like this still accept` : noRoom(x.headroom_capped_by)}</em></div>
        <div><small>Extra interest by holding</small><b>{money(x.cost_full_term)}</b><em>{x.tenure_years ? `over ${x.tenure_years} years` : ''}</em></div>
      </div>
    </Card>
    <Card title={`Files like this in ${city || 'the book'}`} sub={`${SEGMENT[s.segment] || titleCase(s.segment)}, risk grade ${s.risk_grade || '—'}: how often the branch priced them above the model, and what happened.`}>
      <UpsellTable rows={[['Files like this', s], [`All of ${city || 'the book'}`, h.upselling.branch], ['Every branch', h.upselling.all_branches]]}/>
      <p className="rv-note">{h.note}</p>
    </Card>
  </div>;
}

export function FileCompetitors({ h, a }) {
  const c = h.file_notes.competitor, row = h.competitors.this_lender, city = h.where.city;
  return <div className="rv-stack">
    <Card title="The competitor on this file" sub={c.name ? `${c.name} quoted ${pct(c.rate)} · sanction letter ${c.evidence ? 'seen' : 'not seen'}` : 'No lender named on this file.'}>
      {row ? <div className="hs-facts">
        <div><small>{row.lender} in {city}</small><b>{number(row.files)} files</b><em>{share(row.share)} of the branch’s files</em></div>
        <div><small>Usual quote</small><b>{pct(row.median_quote)}</b><em>{bps(row.median_gap_bps)} against our rate</em></div>
        <div><small>Quote documented</small><b>{share(row.documented)}</b><em>sanction letter or offer seen</em></div>
        <div className={row.we_won != null && row.we_won < 0.6 ? 'bad' : 'good'}><small>We still won</small><b>{share(row.we_won)}</b><em>of files naming {row.lender}</em></div>
      </div> : c.name && <p className="rv-muted">{c.name} has no history in {city || 'this branch'}.</p>}
    </Card>
    {a.sales_context?.competitor_rate ? <CompetitorCheck applicant={a}/> : null}
  </div>;
}

export function FileNotes({ h }) {
  const f = h.file_notes, lender = h.competitors.lender;
  const past = h.notes.notes.filter(n => n.names_lender).slice(0, 8);
  return <div className="rv-stack">
    <Card title="Notes on this file">
      <dl className="hs-notes-file">
        <div><dt>RM’s reason for the lower rate</dt><dd>{titleCase(f.rm_reason)}{f.rm_request ? ` · “${f.rm_request}”` : ''}</dd></div>
        <div><dt>Conversation notes</dt><dd>{f.conversation || 'None recorded when the application was entered.'}</dd></div>
      </dl>
    </Card>
    <Precedents p={h.similar_negotiations}/>
    {lender && h.notes.naming_lender > 0
      ? <Card title={`Past conversations naming ${lender} in ${h.where.city}`} sub={`${h.notes.naming_lender} notes in the branch’s history name ${lender}; the newest are below.`}><NoteList notes={past}/></Card>
      : <Card title={`Recent negotiations in ${h.where.city || 'the branch'}`} sub={lender ? `No past note in ${h.where.city || 'this branch'} names ${lender}; the branch’s newest negotiation notes are below.` : 'The branch’s newest negotiation notes.'}>
        <NoteList notes={h.notes.notes.filter(n => n.type === 'negotiation').slice(0, 8)}/></Card>}
  </div>;
}

// ------------------------------------------------------------ the branch

export function BranchUpselling({ h }) {
  const u = h.upselling;
  return <div className="rv-stack">
    <Card title={`Upselling in ${h.where.city || 'every branch'}`} sub="Files priced above the system’s recommendation, against files left at it.">
      <UpsellTable rows={[[h.where.level === 'branch' ? h.where.city : 'Every branch', u.branch], ['Every branch', u.all_branches]]}/>
    </Card>
    {u.by_reason.length > 0 && <Card title="Why the branch priced above the system recommendation">
      <div className="table-scroll"><table className="sc-table hs-table"><thead><tr><th>Reason given</th><th>Files</th><th>Typical upsell</th><th>Took the loan</th></tr></thead>
        <tbody>{u.by_reason.map(r => <tr key={r.reason}><td>{titleCase(r.reason)}</td><td>{r.files}</td><td>+{(r.median_premium_bps / 100).toFixed(2)}%</td><td>{share(r.took)}</td></tr>)}</tbody></table></div>
      <p className="rv-note">{h.note}</p></Card>}
  </div>;
}

export function CompetitorStrength({ h }) {
  const c = h.competitors;
  return <Card title={`Competitor strength in ${h.where.city || 'every branch'}`} sub={`${number(c.named_files)} of ${number(c.files)} files named a lender. Strongest first: the share of those files we did not win.`}>
    <div className="table-scroll"><table className="sc-table hs-table">
      <thead><tr><th>Lender</th><th>Files</th><th>Usual quote</th><th>Against our rate</th><th>Quote documented</th><th>We won</th><th>We lost</th></tr></thead>
      <tbody>{c.lenders.map(r => <tr key={r.lender} className={c.lender && r.lender.toLowerCase() === c.lender.toLowerCase() ? 'requested' : ''}>
        <td><b>{r.lender}</b>{c.lender && r.lender.toLowerCase() === c.lender.toLowerCase() && <small>On this file</small>}</td>
        <td>{r.files}<small>{share(r.share)}</small></td><td>{pct(r.median_quote)}</td><td>{bps(r.median_gap_bps)}</td>
        <td>{share(r.documented)}</td><td>{share(r.we_won)}</td><td className={r.they_won != null && r.they_won > 0.4 ? 'late' : ''}>{share(r.they_won)}</td></tr>)}</tbody></table></div>
    <p className="rv-note">“We lost” counts every file naming the lender that did not take our loan, whatever the reason. A minus means the lender quoted under the rate we sanctioned. {h.note}</p>
  </Card>;
}

const PRICED = { up: 'Priced above the system recommendation', down: 'Price cut below the system recommendation', at_recommendation: 'Held at the system recommendation' };

// The past notes closest in meaning to this file's conversation, and what happened on each.
// Context for the area manager; it never changes the recommendation.
function Precedents({ p }) {
  if (!p) return null;
  if (!p.available) return <Card title="Most similar past negotiations" sub={p.unavailable ? `Not available: ${p.unavailable}.` : 'Not available.'}/>;
  const where = p.scope === 'city' ? `in ${p.city}` : 'across every branch';
  return <Card title="Most similar past negotiations" sub={`The ${p.precedents.length} past notes closest in meaning to this file’s conversation, ${where}, most similar first.`}>
    {!p.precedents.length ? <p className="rv-muted">No similar past note was found.</p>
      : <ul className="hs-notelist">{p.precedents.map(n => <li key={n.note_id} className="lender">
        <div className="hs-note-meta"><span className="hs-sim">{Math.round(n.similarity * 100)}% similar</span>
          {n.date && <span>{n.date}</span>}{n.city && <span>{n.city}</span>}
          {n.lender && <span>{n.lender}{n.quote ? ` · quoted ${pct(n.quote)}` : ''}</span>}
          {n.took != null && <span className={n.took ? 'took' : 'lost'}>{n.took ? `Took our loan${n.rate ? ` at ${pct(n.rate)}` : ''}` : 'Did not take our loan'}</span>}
          {n.priced && <span>{PRICED[n.priced]}</span>}</div>
        <p>{n.text}</p></li>)}</ul>}
    <p className="rv-note">Found by {p.backend || 'text search'}{p.fallback_reason ? ' (the semantic index is unavailable, so this is keyword matching)' : ''}. {p.note}</p>
  </Card>;
}

function NoteList({ notes, first = 5 }) {
  const [all, setAll] = useState(false);
  if (!notes.length) return <p className="rv-muted">No notes.</p>;
  const shown = all ? notes : notes.slice(0, first);
  return <><ul className="hs-notelist">{shown.map((n, i) => <li key={i} className={n.names_lender ? 'lender' : ''}>
    <div className="hs-note-meta"><span>{n.date}</span><span>{NOTE_TYPES[n.type] || titleCase(n.type)}</span>
      {n.lender && <span>{n.lender}{n.quote ? ` · ${pct(n.quote)}` : ''}</span>}
      {n.took != null && <span className={n.took ? 'took' : 'lost'}>{n.took ? `Took our loan${n.rate ? ` at ${pct(n.rate)}` : ''}` : 'Did not take our loan'}</span>}
      {n.priced === 'premium' && <span className="took">Priced above the system recommendation</span>}</div>
    <p>{n.text}</p></li>)}</ul>
  {notes.length > first && <button type="button" className="link-button hs-more" onClick={() => setAll(!all)}>{all ? 'Show fewer' : `Show all ${notes.length}`}</button>}</>;
}

export function SalesNotes({ h }) {
  const [type, setType] = useState('');
  const n = h.notes;
  const shown = n.notes.filter(x => !type || x.type === type);
  return <Card title={`Sales and conversation notes · ${h.where.city || 'every branch'}`} sub={`${number(n.total)} notes on the branch’s past files${h.competitors.lender && n.naming_lender ? `; the ${number(n.naming_lender)} naming ${h.competitors.lender} first` : ''}, newest first.`}>
    <div className="hs-filter">{[['', 'All'], ...Object.entries(NOTE_TYPES)].map(([k, l]) =>
      <button key={k || 'all'} type="button" className={type === k ? 'active' : ''} onClick={() => setType(k)}>{l}{k && n.counts[k] ? ` (${number(n.counts[k])})` : ''}</button>)}</div>
    <NoteList notes={shown}/>
  </Card>;
}
