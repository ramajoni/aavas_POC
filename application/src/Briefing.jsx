import React, { useEffect, useState } from 'react';
import { Binoculars, Buildings, ChartLineUp, CircleNotch } from '@phosphor-icons/react';
import { api, money, titleCase } from './api';
import { InfoBox } from './components';
import './briefing.css';

const share = r => r == null ? '—' : `${Math.round(r * 100)}%`;
const bps = n => n == null ? '—' : `${Math.round(n)} bps`;
const rate = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const rupees = n => n == null ? '—' : money(n);
const signedPts = n => n == null ? '—' : `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n).toFixed(1)} pts`;
const signedInr = n => n == null ? '—' : `${n > 0 ? '+' : n < 0 ? '−' : ''}${money(Math.abs(n))}`;
const VERDICT = { 'value creating': ['Paid for themselves', 'good'], 'value destroying': ['Lost value', 'bad'],
  'no measurable effect': ['No measurable difference', 'flat'], 'insufficient evidence': ['Too few files to tell', 'flat'] };
const TABS = [['overrides', 'Branch overrides', Buildings], ['competitors', 'Competitor intelligence', Binoculars],
  ['performance', 'Historical performance', ChartLineUp]];

function Tile({ label, value, compare, note, tone }) {
  return <div className={`bf-tile${tone ? ` ${tone}` : ''}`}><small>{label}</small><b>{value}</b>
    {compare && <em>{compare}</em>}{note && <em>{note}</em>}</div>;
}

// Before an area manager approves a lower rate: how this branch overrides, who
// competes here, and how its book has performed, with the all-branch figure for reference.
export function ApprovalBriefing({ requestId }) {
  const [b, setB] = useState(null);
  const [error, setError] = useState('');
  const [tab, setTab] = useState('overrides');
  useEffect(() => {
    setB(null); setError('');
    api(`/pricing/rate-requests/${encodeURIComponent(requestId)}/briefing`).then(setB).catch(e => setError(e.message));
  }, [requestId]);

  if (error) return <InfoBox warning><span role="alert">{error}</span></InfoBox>;
  if (!b) return <section className="bf"><p className="bf-loading"><CircleNotch className="spin" size={16}/> Reading this branch’s history…</p></section>;
  const where = b.where.level === 'branch' ? `${b.where.city} branch` : 'all branches';
  return <section className="bf">
    <div className="bf-head">
      <div><h2>Before you approve · {where}</h2>
        <small>{b.where.level === 'branch' ? `${b.where.branch_id} · ${b.where.state} · ` : ''}{b.where.files.toLocaleString('en-IN')} files, {b.window.from} to {b.window.to}</small></div>
    </div>
    {b.where.note && <InfoBox>{b.where.note}</InfoBox>}
    {b.signals?.length > 0 && <ul className="bf-signals">{b.signals.map(s => <li key={s}>{s}</li>)}</ul>}
    <div className="bf-tabs" role="tablist">{TABS.map(([key, label, Icon]) =>
      <button key={key} type="button" role="tab" aria-selected={tab === key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>
        <Icon size={16}/> {label}</button>)}</div>
    {tab === 'overrides' && <Overrides b={b}/>}
    {tab === 'competitors' && <Competitors b={b}/>}
    {tab === 'performance' && <Performance b={b}/>}
    <details className="bf-caveats"><summary>About these figures</summary><ul>{b.caveats.map(c => <li key={c}>{c}</li>)}</ul></details>
  </section>;
}

function Overrides({ b }) {
  const o = b.overrides, br = o.branch, all = o.all_branches;
  const effect = o.effect_this_reason || o.effect_all_discounts;
  const [verdict, tone] = VERDICT[effect?.verdict] || ['—', 'flat'];
  const like = o.similar_here;
  return <div className="bf-body">
    <div className="bf-tiles">
      <Tile label="Files discounted below the recommendation" value={share(br.discount_rate.rate)}
        compare={`All branches ${share(all.discount_rate.rate)}`}/>
      <Tile label="Median discount" value={bps(br.median_discount_bps)} compare={`All branches ${bps(all.median_discount_bps)}`}/>
      <Tile label="Discounts citing a competitor" value={share(br.competitor_share_of_discounts.rate)}
        compare={`All branches ${share(all.competitor_share_of_discounts.rate)}`}/>
      <Tile tone={tone} label={`Did ${effect?.label === 'all discounts' || !effect ? 'discounts' : titleCase(effect.label).toLowerCase() + ' discounts'} pay?`}
        value={verdict} note={effect && effect.verdict !== 'insufficient evidence' ? `${signedPts(effect.s2d_lift_pts)} conversion · ${signedInr(effect.value_lift_inr)} a file · ${effect.n_override} files` : null}/>
    </div>
    <div className="bf-grid">
      <div><h3>Why this branch discounts</h3>
        <table className="bf-table"><thead><tr><th>Reason</th><th>Files</th><th>Share</th><th>Median</th><th>Converted</th></tr></thead>
          <tbody>{o.by_reason.map(r => <tr key={r.reason} className={r.reason === b.reason ? 'mine' : ''}>
            <td>{titleCase(r.reason)}</td><td>{r.files}</td><td>{share(r.share)}</td><td>{bps(r.median_discount_bps)}</td><td>{share(r.conversion)}</td></tr>)}</tbody></table></div>
      <div><h3>Past discounts like this one</h3>
        <p className="bf-note">{titleCase(like.reason || 'any reason')}, {Math.round(like.window_bps[0])}–{Math.round(like.window_bps[1])} bps, this branch: <b>{like.files} files</b></p>
        {like.files === 0 ? <InfoBox warning>No past discount went this deep: {Math.round(b.requested_discount_bps)} bps asked; the deepest here was {bps(like.deepest_bps)}.</InfoBox> :
        <div className="bf-mini">
          <Tile label="Converted" value={share(like.conversion)}/>
          <Tile label="Moved away within 2 years" value={share(like.bt_out_24m)}/>
          <Tile label="Defaulted" value={share(like.default_rate)}/>
          <Tile label="Value per file" value={rupees(like.value_per_file_inr)}/>
        </div>}</div>
    </div>
  </div>;
}

function Competitors({ b }) {
  const c = b.competitors;
  if (!c) return <div className="bf-body"><InfoBox>Competitor history is not available on this server.</InfoBox></div>;
  const mine = c.this_file;
  return <div className="bf-body">
    {mine ? <div className="bf-tiles">
      <Tile label={`${mine.competitor} usually quotes here`} value={rate(mine.benchmark.median)}
        compare={`${mine.benchmark.n} quotes · ${mine.benchmark.level || '—'} · 5th percentile ${rate(mine.benchmark.p5)}`}/>
      <Tile tone={mine.unusually_low ? 'bad' : null} label="This file’s quote" value={rate(mine.quote)}
        compare={`${mine.unusually_low ? 'Unusually low for this lender · ' : ''}sanction letter ${mine.evidence_seen ? 'seen' : 'not seen'}`}/>
      <Tile label="Gap to the rate asked" value={mine.gap_to_requested_bps == null ? '—' : `${mine.gap_to_requested_bps > 0 ? '+' : ''}${Math.round(mine.gap_to_requested_bps)} bps`}
        compare={mine.win_rate_at_gap.bucket ? `Bucket ${mine.win_rate_at_gap.bucket} bps` : null}/>
      <Tile label="We won at this gap" value={share(mine.win_rate_at_gap.city ?? mine.win_rate_at_gap.all_cities)}
        compare={mine.win_rate_at_gap.city != null ? `In ${b.where.city} · ${mine.win_rate_at_gap.city_n} files` : `All branches · ${mine.win_rate_at_gap.all_n ?? 0} files`}
        note={mine.win_rate_at_gap.beyond_history ? 'No past file had a gap this large' : null}/>
    </div> : <p className="bf-note">The customer did not name a lender we track, so there is no lender-specific history; the city’s competitors are below.</p>}
    <h3>Who quotes against us {b.where.level === 'branch' ? `in ${b.where.city}` : 'across branches'}</h3>
    <div className="table-scroll"><table className="bf-table">
      <thead><tr><th>Lender</th><th>Files</th><th>Median quote</th><th>Gap to our rate</th><th>Documented</th><th>We still won</th></tr></thead>
      <tbody>{c.landscape.map(r => <tr key={r.competitor} className={mine?.competitor === r.competitor ? 'mine' : ''}>
        <td>{r.competitor} <small>{r.lender_type}</small></td><td>{r.files} <small>{share(r.share_of_files)}</small></td>
        <td>{rate(r.median_quote)}</td><td>{r.median_gap_bps == null ? '—' : `${r.median_gap_bps > 0 ? '+' : ''}${Math.round(r.median_gap_bps)} bps`}</td>
        <td>{share(r.documented_share)}</td><td>{share(r.win_rate)}</td></tr>)}</tbody></table></div>
    <p className="bf-note">From the sales notes on past files, joined to what happened. A negative gap means the lender quoted under our sanctioned rate. The fact check above shows what the lender advertises today.</p>
  </div>;
}

function Performance({ b }) {
  const p = b.performance, br = p.branch, all = p.all_branches, rm = p.rm;
  const vs = (fmt, key) => `All branches ${fmt(all[key])}`;
  const peak = Math.max(1, ...p.monthly.map(m => m.files));
  return <div className="bf-body">
    <div className="bf-tiles six">
      <Tile label="Converted" value={share(br.conversion)} compare={vs(share, 'conversion')}/>
      <Tile label="Average sanctioned rate" value={rate(br.avg_sanctioned_rate)} compare={vs(rate, 'avg_sanctioned_rate')}/>
      <Tile label="Median yield" value={rate(br.median_yield_pct)} compare={vs(rate, 'median_yield_pct')}/>
      <Tile label="Moved away within 2 years" value={share(br.bt_out_24m)} compare={vs(share, 'bt_out_24m')}/>
      <Tile label="Delinquent in 12 months" value={share(br.delinquent_12m)} compare={vs(share, 'delinquent_12m')}/>
      <Tile label="Value per file" value={rupees(br.value_per_file_inr)} compare={vs(rupees, 'value_per_file_inr')}/>
    </div>
    <div className="bf-grid">
      <div><h3>Last {p.monthly.length} months</h3>
        <table className="bf-table"><thead><tr><th>Month</th><th>Files</th><th>Discounted</th><th>Converted</th></tr></thead>
          <tbody>{p.monthly.map(m => <tr key={m.month}><td>{m.month}</td>
            <td><span className="bf-bar"><i style={{ width: `${(m.files / peak) * 100}%` }}/></span>{m.files}</td>
            <td>{share(m.discount_rate)}</td><td>{share(m.conversion)}</td></tr>)}</tbody></table></div>
      <div><h3>{b.requested_by_name}’s record since go-live</h3>
        <div className="bf-mini">
          <Tile label="Lower rates asked" value={rm.requests} compare={`${rm.approved} approved · ${rm.declined} declined · ${rm.pending} waiting`}/>
          <Tile label="Approved when decided" value={share(rm.approval_rate)}/>
          <Tile label="Average discount asked" value={bps(rm.avg_requested_discount_bps)}/>
          <Tile label="Offers accepted" value={share(rm.acceptance_rate)} compare={`${rm.offers_accepted} of ${rm.offers_sent} sent`}/>
        </div></div>
    </div>
  </div>;
}
