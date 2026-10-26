import React, { useEffect, useRef, useState } from 'react';
import { ArrowClockwise, ArrowSquareOut, CircleNotch, Globe } from '@phosphor-icons/react';
import { api } from './api';
import './competitor-check.css';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const day = iso => iso ? new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : '';
const host = url => { try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return url; } };
const signed = n => (n > 0 ? `+${n}` : n < 0 ? `−${Math.abs(n)}` : '0');
const POLL_MS = 4000, POLL_LIMIT = 60;      // asks again for up to four minutes
const MARKET = 'market';

// Lender names to pick a competitor from (the competitor registry).
export function useLenders() {
  const [lenders, setLenders] = useState([]);
  useEffect(() => { api('/competitors/lenders').then(r => setLenders(r.lenders || [])).catch(() => setLenders([])); }, []);
  return lenders;
}

// The competitor fact check: what the lender advertises today, found by the
// market research agent, and how believable the quote is against it, out of 100.
// Guidance only - it never moves the recommended rate.
export function CompetitorCheck({ applicant: a, onResult }) {
  const sc = a.sales_context || {};
  const quote = Number(sc.competitor_rate) || null;
  const body = { loan_id: a.loan_id, competitor_name: sc.competitor_name || null, competitor_rate: quote,
    evidence_seen: Boolean(sc.competitor_evidence_verified), bureau_score: a.bureau_score > 0 ? a.bureau_score : null,
    new_to_credit: Boolean(a.new_to_credit), segment: a.segment || null };
  const key = JSON.stringify(body);
  const [check, setCheck] = useState(null);
  const [error, setError] = useState('');
  const [round, setRound] = useState(0);
  const refresh = useRef(false);

  useEffect(() => {
    if (!quote) { setCheck(null); onResult?.(null); return undefined; }
    let live = true, timer, polls = 0;
    const tick = async () => {
      const again = refresh.current; refresh.current = false;
      try {
        const r = await api('/competitors/check', { ...body, refresh: again });
        if (!live) return;
        setCheck(r); setError(''); onResult?.(r);
        if (r.state === 'researching' && polls++ < POLL_LIMIT) timer = setTimeout(tick, POLL_MS);
      } catch (e) { if (live) setError(e.message); }
    };
    tick();
    return () => { live = false; clearTimeout(timer); };
  }, [key, round]);

  if (!quote) return null;
  const s = check?.score, res = check?.research;
  const market = check?.lender === MARKET;
  const who = market ? 'Lenders in the market' : (check?.lender_label || sc.competitor_name || 'The lender');
  const sources = [...new Map((res?.rates || []).map(r => [r.source_url, r])).values()].slice(0, 4);
  const searching = check?.state === 'researching';

  return <section className={`cc cc-${s?.band || 'pending'}`} aria-live="polite">
    <div className="cc-head">
      <span className="cc-title"><Globe size={17}/> Competitor fact check</span>
      {s && <span className="cc-score"><b>{s.points}</b>/100 · {s.label}</span>}
    </div>
    <p className="cc-line">
      {res?.published_min != null
        ? <>{who} {market ? 'advertise' : 'advertises'} <b>{pct(res.published_min)}{res.published_max > res.published_min ? `–${pct(res.published_max)}` : ''}</b> ({res.source_quality === 'lender_website' ? (market ? "lenders' own websites" : 'its own website') : 'news and comparison sites'}, checked {day(res.researched_at)}). </>
        : null}
      Quote {pct(quote)}{sc.competitor_name ? ` from ${sc.competitor_name}` : ', lender not named'} · sanction letter {body.evidence_seen ? 'seen' : 'not seen'}.
    </p>
    {searching && <p className="cc-status"><CircleNotch className="spin" size={15}/> Fact checking against {market ? 'lenders’' : `${who}’s`} published rates. This takes a minute or two; the score below rests on the evidence until then.</p>}
    {check?.state === 'unavailable' && <p className="cc-status">{check.reason} The quote is scored on its evidence alone.</p>}
    {(check?.state === 'failed' || check?.state === 'not_found') && <p className="cc-status">{check.state === 'failed' ? (check.reason || 'The fact check did not complete.') : 'No published rate was found to check against.'} The quote is scored on its evidence alone.</p>}
    {error && <p className="cc-status" role="alert">{error}</p>}
    {s && <p className="cc-guidance">{s.guidance}</p>}
    {s && <details className="cc-detail">
      <summary>How this was scored</summary>
      <ul className="cc-points">{s.components.map(c => <li key={c.label}><b>{signed(c.points)}</b> {c.label}{c.detail ? ` — ${c.detail}` : ''}</li>)}</ul>
      {sources.length > 0 && <div className="cc-sources"><small>Sources</small>
        {sources.map(r => <a key={r.source_url} href={r.source_url} target="_blank" rel="noopener noreferrer"><ArrowSquareOut size={13}/> {host(r.source_url)}{r.as_of ? ` · ${r.as_of}` : ''}</a>)}</div>}
      <small className="muted">Guidance only: fact checking never changes the recommended rate.</small>
    </details>}
    {!searching && check?.state !== 'unavailable' && <button type="button" className="link-button cc-again" onClick={() => { refresh.current = true; setRound(n => n + 1); }}><ArrowClockwise size={14}/> Fact check again</button>}
  </section>;
}
