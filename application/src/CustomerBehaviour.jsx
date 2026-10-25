import React, { useEffect, useState } from 'react';
import { CalendarBlank, ChatsCircle, Database, Gauge, Handshake, Lightbulb, Storefront, UsersThree } from '@phosphor-icons/react';
import { api, percent } from './api';
import { Badge, Empty, InfoBox, Panel } from './components';
import './customer-behaviour.css';

const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const RESPONSES = [['accepted', 'Accepted'], ['negotiated', 'Negotiated'], ['deferred', 'Deferred'], ['rejected', 'Rejected']];
const when = iso => new Date(iso).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });

// Which price-gap band a gap (our rate minus their quote, in bps) falls in: (lo, hi].
const bandFor = (bands, gap) => bands.findIndex(b => (b.lo_bps == null || gap > b.lo_bps) && (b.hi_bps == null || gap <= b.hi_bps));

// One customer's behaviour, from the database: what they have told us, what they
// have done, and what customers like them did at the rate on the slider.
export function CustomerBehaviourPanel({ loanId, decisionId, name, proposedRate }) {
  const [body, setBody] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!decisionId) return;
    let live = true; setError('');
    api(`/customers/${encodeURIComponent(loanId)}/behaviour`).then(b => live && setBody(b)).catch(e => live && setError(e.message));
    return () => { live = false; };
  }, [loanId, decisionId]);

  const first = (name || 'this customer').split(' ')[0];
  const title = `Customer behaviour · ${name || loanId}`;
  if (!decisionId) return <Panel title={title} icon={UsersThree}><Empty icon={UsersThree} title="Analysed after pricing">Appears as soon as the file is priced, which happens automatically when it opens.</Empty></Panel>;
  if (error) return <Panel title={title} icon={UsersThree}><InfoBox warning><span role="alert">{error}</span></InfoBox></Panel>;
  if (!body) return <Panel title={title} icon={UsersThree}><p className="muted">Reading the loan book…</p></Panel>;

  const s = body.signals, c = body.cohort;
  const rate = Number(proposedRate) || s.recommended_rate;
  const gap = s.competitor_rate && rate ? Math.round((rate - s.competitor_rate) * 100) : null;
  const at = gap == null ? -1 : bandFor(c.by_gap, gap);
  const here = at >= 0 ? c.by_gap[at] : null;

  return <Panel className="cb" title={title} icon={UsersThree}
    action={<span className="cb-source"><Database size={15}/> {body.loan_book.rows.toLocaleString('en-IN')} loans · {body.loan_book.table} table</span>}>
    <div className="cb-grid">
      <section>
        <h3>What {first} is telling us</h3>
        <dl className="cb-signals">
          <div><dt><Storefront size={17}/> Competitor quote</dt><dd>{s.competitor_rate ? <>{percent(s.competitor_rate)} <small>{s.competitor_name}</small> <Badge tone={s.competitor_evidence_verified ? 'green' : 'amber'}>{s.competitor_evidence_verified ? 'Documented' : 'Verbal only'}</Badge></> : 'None on file'}</dd></div>
          <div><dt><Handshake size={17}/> Negotiation</dt><dd>{s.customer_negotiating ? 'Negotiating' : 'Not negotiating'} · {s.prior_negotiation_rounds} round{s.prior_negotiation_rounds === 1 ? '' : 's'} so far</dd></div>
          <div><dt><Gauge size={17}/> Price sensitivity</dt><dd>{s.price_sensitivity ? <>{s.price_sensitivity.sensitivity_label} <small>{s.price_sensitivity.points_lost_per_25bps?.toFixed(1)} pts per 25 bps</small></> : '—'}</dd></div>
          <div><dt><ChatsCircle size={17}/> Relationship</dt><dd>{s.existing_relationship ? 'Existing customer' : 'New to us'} · {s.channel_label}</dd></div>
          <div><dt><CalendarBlank size={17}/> Timing</dt><dd>{s.days_to_month_end ?? '—'} days to month end{s.sales_target_pct != null && <small> · branch at {Math.round(s.sales_target_pct)}% of target</small>}</dd></div>
        </dl>
      </section>

      <section>
        <h3>What {first} has done</h3>
        <ol className="cb-timeline">{body.timeline.map((e, i) => <li key={i} className={e.kind}>
          <span>{when(e.at)}{e.count > 1 && <> · {e.count} times, last {when(e.last_at)}</>}</span><p>{e.text}</p></li>)}</ol>
      </section>

      <section>
        <h3>Customers like {first}</h3>
        <p className="cb-match">{c.n.toLocaleString('en-IN')} loans: {c.matched_on.join(' · ') || 'the whole book'}{c.relaxed.length > 0 && <small> (not matched on {c.relaxed.join(', ')}: too few loans)</small>}</p>
        {gap == null
          ? <p className="muted">No competitor quote on file, so no price gap to compare.</p>
          : <div className="cb-now">
              <p>At <b>{percent(rate)}</b>, {first} is <b>{Math.abs(gap)} bps {gap < 0 ? 'below' : gap > 0 ? 'above' : 'level with'}</b> the quote.</p>
              {here && !here.suppressed ? <>
                <div className="cb-figures">
                  <div><b>{share(here.win_rate)}</b><small>took the loan</small></div>
                  <div><b>{share(here.moved_away_24m)}</b><small>of those left within 24 months</small></div>
                  <div><b>{here.avg_discount_bps.toFixed(1)}</b><small>bps average discount</small></div>
                </div>
                <div className="cb-mix" role="img" aria-label={RESPONSES.map(([k, l]) => `${l} ${share(here.responses[k])}`).join(', ')}>
                  {RESPONSES.map(([k, l]) => <span key={k} className={k} style={{ width: `${(here.responses[k] || 0) * 100}%` }} title={`${l} ${share(here.responses[k])}`}/>)}
                </div>
                <div className="cb-legend">{RESPONSES.map(([k, l]) => <span key={k}><i className={k}/>{l} {share(here.responses[k])}</span>)}</div>
                <small className="muted">{here.label.toLowerCase()} · n = {here.n.toLocaleString('en-IN')}</small>
              </> : <p className="muted">Too few similar loans at this gap to say (n = {here?.n ?? 0}).</p>}
            </div>}
        <div className="cb-bands">{c.by_gap.map((b, i) => <div key={b.label} className={i === at ? 'here' : ''}>
          <span>{b.label}</span>
          <div className="cb-bar"><i style={{ width: b.suppressed ? 0 : `${b.win_rate * 100}%` }}/></div>
          <b>{b.suppressed ? `n=${b.n}` : share(b.win_rate)}</b>
        </div>)}</div>
      </section>
    </div>
    {body.readouts.length > 0 && <InfoBox><b className="cb-read"><Lightbulb size={16}/> Reading {first}</b><ul>{body.readouts.map(r => <li key={r}>{r}</li>)}</ul></InfoBox>}
    <p className="small-copy">Context for the conversation, never a price input. Similar customers are read from the loan book table in the database (loaded {when(body.loan_book.loaded_at)}); gaps with fewer than {body.settings.min_cell} loans are not shown. Moves with the rate on the slider.</p>
  </Panel>;
}
