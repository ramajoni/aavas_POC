import React, { useEffect, useRef, useState } from 'react';
import { CircleNotch } from '@phosphor-icons/react';
import { money, apiOnce } from './api';
import { InfoBox } from './components';
import { Fold } from './Fold';
import './best-offer.css';

const pct = r => r == null ? '—' : `${Number(r).toFixed(2)}%`;
const share = v => v == null ? '—' : `${Math.round(v * 100)}%`;
const pts = v => v == null ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}`;
const rupees = v => v == null ? '—' : `${v < 0 ? '−' : v > 0 ? '+' : ''}${money(Math.abs(v))}`;
const years = m => m % 12 === 0 ? `${m / 12} years` : `${m} months`;
// "Highest acceptance while earning at least what the system recommendation earns", from the objective's own settings.
const goal = b => {
  const ref = b.at_least.reference === 'ask' ? 'the RM’s ask' : 'the system recommendation';
  const head = b.maximise === 'acceptance' ? 'Highest acceptance' : 'Most risk-adjusted value';
  return b.at_least.measure === 'value' ? `${head} while earning at least what ${ref} earns` : `${head} while at least as likely to accept as ${ref}`;
};
const cautions = o => [o.amount_reduced && 'smaller loan than asked', o.tenure_changed && 'different tenure'].filter(Boolean);
const MARK = '#eb6834', REC = '#2a78d6', BEST = '#15803d';

const lakh = v => v == null ? '—' : v >= 1e7 ? `Rs ${+(v / 1e7).toFixed(2)} crore` : v >= 1e5 ? `Rs ${+(v / 1e5).toFixed(2)} lakh` : `Rs ${Math.round(v).toLocaleString('en-IN')}`;
const tickRupees = v => v === 0 ? '0' : v >= 1e7 ? `${+(v / 1e7).toFixed(2)} crore` : v >= 1e5 ? `${+(v / 1e5).toFixed(2)} lakh` : `${Math.round(v / 1000)}k`;
const sameLevers = (a, b) => a && b && Math.abs(a.rate - b.rate) < 0.004 && Math.abs(a.loan_amount - b.loan_amount) < 0.5 && a.tenure_months === b.tenure_months && Math.abs(a.fee_pct - b.fee_pct) < 1e-9;
// A round axis maximum, with four equal steps that are round rupee amounts.
const niceMax = v => { const raw = Math.max(v, 1) / 4, mag = 10 ** Math.floor(Math.log10(raw)); const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw); return step * 4; };

// What differs from the file as it stands, in words: "fee waived · 5 years · Rs 22.4 lakh".
function changed(o, rec, own) {
  const out = [];
  if (Math.abs(o.rate - rec.rate) > 0.004) out.push(`rate ${pct(o.rate)}`);
  if (o.tenure_months !== own.tenure_months) out.push(years(o.tenure_months));
  if (Math.abs(o.fee_pct - own.fee_pct) > 1e-9) out.push(o.fee_pct <= 1e-9 ? 'fee waived' : `fee ${pct(o.fee_pct)}`);
  if (Math.abs(o.loan_amount - own.loan_amount) > 0.5) out.push(lakh(o.loan_amount));
  return out.length ? out.join(' · ') : 'as the file stands';
}

const GRID = [0, 0.25, 0.5, 0.75, 1];

// "Compare the offers": the ask, the recommendation and the three best offers as two bar
// charts that share their row labels, one axis each. At narrow widths the charts stack.
function Compare({ rows, own, rec }) {
  const [hi, setHi] = useState(null);
  const [top, setTop] = useState(0);
  const offers = rows.filter(r => r.offer);
  const vmax = niceMax(Math.max(...offers.map(r => r.offer.value_inr || 0), 1));
  const show = (i, el) => { setHi(i); setTop(el.offsetTop + el.offsetHeight + 4); };
  const track = (r, i, kind) => {
    const o = r.offer;
    const frac = !o ? null : kind === 'acc' ? Math.min(Math.max(o.acceptance ?? 0, 0), 1) : Math.min(Math.max((o.value_inr ?? 0) / vmax, 0), 1);
    const text = !o ? '' : kind === 'acc' ? share(o.acceptance) : lakh(o.value_inr);
    return <div key={`${kind}${i}`} className={`bo-track ${kind}`} style={{ '--i': i, '--o': (kind === 'acc' ? 10 : 100) + i * 2 + 1 }}
      onPointerEnter={e => show(i, e.currentTarget)} onPointerLeave={() => setHi(null)}>
      <div className="bo-plotarea" aria-hidden="true">{GRID.map(g => <u key={g} style={{ left: `${g * 100}%` }}/>)}
        {o && <><i className={`bo-bar ${r.kind}`} style={{ width: `${frac * 100}%` }}/><em className="bo-end" style={{ left: `calc(${frac * 100}% + 6px)` }}>{text}</em></>}</div>
      {!o && <span className="bo-none">no bar</span>}
    </div>;
  };
  const label = (r, i, second) => <div key={`l${second ? 2 : 1}${i}`} className={`bo-lab ${second ? 'second' : 'first'}`} style={{ '--i': i, '--o': (second ? 100 : 10) + i * 2 }}
    tabIndex={second ? -1 : 0} onPointerEnter={e => show(i, e.currentTarget)} onPointerLeave={() => setHi(null)} onFocus={e => show(i, e.currentTarget)} onBlur={() => setHi(null)}>
    <b>{r.name}</b>{r.same && <span className="bo-same"> (same offer as {r.same})</span>}
    {r.outside && <span className="bo-tag">outside policy</span>}
    {r.offer ? <span className="bo-lev">{changed(r.offer, rec, own)}{cautions(r.offer).map(c => <span key={c} className="bo-caut">{c}</span>)}</span>
      : <span className="bo-lev">{r.reason}</span>}
  </div>;
  const axis = kind => <div key={`a${kind}`} className={`bo-axisrow ${kind}`} style={{ '--i': rows.length, '--o': (kind === 'acc' ? 10 : 100) + rows.length * 2 + 1 }} aria-hidden="true">
    <div className="bo-axisplot">{GRID.map(g => <span key={g} style={{ left: `${g * 100}%` }}>{kind === 'acc' ? `${g * 100}%` : tickRupees(g * vmax)}</span>)}</div></div>;
  const tip = hi != null ? rows[hi] : null, t = tip?.offer;
  const summary = rows.map(r => r.offer ? `${r.name}: ${share(r.offer.acceptance)} likely to accept, ${lakh(r.offer.value_inr)} risk-adjusted value` : `${r.name}: ${r.reason}`).join('. ');
  return <div className="bo-compare" role="group" aria-label={`Compare the offers. ${summary}`}>
    <div className="bo-cmp">
      <h4 className="bo-h acc" style={{ '--o': 1 }}>Likely to accept</h4>
      <h4 className="bo-h val" style={{ '--o': 99 }}>Risk-adjusted value</h4>
      {rows.map((r, i) => [label(r, i, false), track(r, i, 'acc'), label(r, i, true), track(r, i, 'val')])}
      {axis('acc')}{axis('val')}
    </div>
    {t && <div className="bo-rowtip" role="tooltip" style={{ top }}><b>{tip.name}</b>
      <span>{pct(t.rate)} · {money(t.loan_amount)} · {years(t.tenure_months)} · fee {pct(t.fee_pct)}</span>
      <span>EMI {money(t.emi)} · APR {pct(t.apr_pct)} · grade {t.risk_grade}, default risk {(t.pd_annual * 100).toFixed(2)}%</span>
      <span>{share(t.acceptance)} likely to accept · {money(t.value_inr)} risk-adjusted value</span>
      {!t.eligible && <span className="bad">Outside policy for this file</span>}
      {cautions(t).length > 0 && <span>Caution: {cautions(t).join(', ')}</span>}</div>}
  </div>;
}

// Every eligible offer as a scatter. Each marker is a lettered pill; pills that would overlap are
// moved clear of each other with a thin leader line back to their true point, and pills share
// one marker only when their points coincide.
function Scatter({ d, marks }) {
  const [hover, setHover] = useState(null);
  const [hm, setHm] = useState(null);
  const svg = useRef(null);
  const pts_ = d.points;
  if (!pts_.length) return null;
  const W = 760, H = 340, L = 74, R = 30, T = 30, B = 42;
  const all = [...pts_, ...marks.map(m => ({ acceptance: m.acceptance, value_inr: m.value_inr }))].filter(p => p.acceptance != null && p.value_inr != null);
  const xs = all.map(p => p.acceptance), ys = all.map(p => p.value_inr);
  const x0 = Math.max(0, Math.floor((Math.min(...xs) - 0.02) * 20) / 20), x1 = Math.min(1, Math.ceil((Math.max(...xs) + 0.02) * 20) / 20);
  const span = Math.max(...ys) - Math.min(...ys) || 1;
  const y0 = Math.min(...ys) - span * 0.06, y1 = Math.max(...ys) + span * 0.1;
  const x = a => L + ((a - x0) / (x1 - x0 || 1)) * (W - L - R), y = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const tx = []; for (let a = x0; a <= x1 + 1e-9; a += 0.05) tx.push(+a.toFixed(2));
  const ty = []; for (let i = 0; i <= 4; i++) ty.push(y0 + ((y1 - y0) * i) / 4);
  const line = d.frontier.map((p, i) => `${i ? 'L' : 'M'}${x(p.acceptance).toFixed(1)},${y(p.value_inr).toFixed(1)}`).join(' ');
  function move(e) {
    const bx = svg.current.getBoundingClientRect();
    const px = ((e.clientX - bx.left) / bx.width) * W, py = ((e.clientY - bx.top) / bx.height) * H;
    const near = pts_.reduce((b, p) => { const dd = (x(p.acceptance) - px) ** 2 + (y(p.value_inr) - py) ** 2; return dd < b[0] ? [dd, p] : b; }, [Infinity, null]);
    setHover(near[0] < 30 ** 2 ? near[1] : null);
  }
  // Merge only coincident points, then lay the pills out so no two boxes overlap.
  const clusters = [];
  for (const m of marks) {
    const same = clusters.find(c => Math.abs(x(c.acceptance) - x(m.acceptance)) < 0.5 && Math.abs(y(c.value_inr) - y(m.value_inr)) < 0.5);
    if (same) { same.keys.push(m.key); same.names.push(m); } else clusters.push({ acceptance: m.acceptance, value_inr: m.value_inr, keys: [m.key], names: [m] });
  }
  const PH = 20, GAP = 3, placed = [];
  const hit = (a, b) => a.x < b.x + b.w + GAP && b.x < a.x + a.w + GAP && a.y < b.y + b.h + GAP && b.y < a.y + a.h + GAP;
  for (const c of clusters) {
    const t = c.keys.join('·'), w = 12 + 8 * t.length, px = x(c.acceptance), py = y(c.value_inr);
    const tries = [[0, 0]];
    for (let k = 1; k <= 8; k++) { tries.push([0, -k * (PH + GAP)], [0, k * (PH + GAP)]); if (k <= 3) tries.push([-(w + GAP) * k, 0], [(w + GAP) * k, 0]); }
    let box = null;
    for (const [dx, dy] of tries) {
      const cx = Math.min(Math.max(px + dx, L + w / 2), W - R - w / 2), cy = Math.min(Math.max(py + dy, T + PH / 2), H - B - PH / 2);
      const b = { x: cx - w / 2, y: cy - PH / 2, w, h: PH };
      if (!placed.some(o => hit(b, o))) { box = b; break; }
    }
    box = box || { x: px - w / 2, y: py - PH / 2, w, h: PH };
    placed.push(box);
    Object.assign(c, { t, box, px, py, moved: Math.hypot(box.x + box.w / 2 - px, box.y + box.h / 2 - py) > 1 });
  }
  const label = `${pts_.length} eligible offers by likelihood to accept and risk-adjusted value. ${marks.map(m => `${m.key}, ${m.label}: ${share(m.acceptance)} likely to accept, ${money(m.value_inr)}`).join('. ')}.`;
  return <figure className="bo-chart">
    <ul className="bo-legend"><li><i className="dot"/>Eligible offer</li><li><i className="dot tenure"/>Different tenure</li><li><i className="dot reduced"/>Smaller loan</li><li><i className="ln"/>Best trade-offs</li></ul>
    <ul className="bo-key" aria-label="Key to the lettered markers">{marks.map(m => <li key={m.key}><b className="bo-chip" style={{ background: m.color }}>{m.key}</b>{m.label}</li>)}</ul>
    <div className="bo-plot">
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} onPointerMove={move} onPointerLeave={() => setHover(null)}>
        {ty.map(v => <g key={v}><line className="bo-grid" x1={L} x2={W - R} y1={y(v)} y2={y(v)}/><text className="bo-axis" x={L - 8} y={y(v) + 4} textAnchor="end">{money(Math.round(v / 1000) * 1000)}</text></g>)}
        {tx.map(a => <text key={a} className="bo-axis" x={x(a)} y={H - B + 18} textAnchor="middle">{share(a)}</text>)}
        <text className="bo-axis" x={(L + W - R) / 2} y={H - 6} textAnchor="middle">Likely to accept</text>
        {pts_.map((p, i) => <circle key={i} className={`bo-pt${p.amount_reduced ? ' reduced' : ''}${p.tenure_changed ? ' tenure' : ''}`} cx={x(p.acceptance)} cy={y(p.value_inr)} r="2.6"/>)}
        {line && <path className="bo-front" d={line}/>}
        {clusters.map(c => <g key={c.t} className="bo-m" tabIndex="0" onPointerEnter={() => setHm(c)} onPointerLeave={() => setHm(null)} onFocus={() => setHm(c)} onBlur={() => setHm(null)}>
          {c.moved && <><line className="bo-leader" x1={c.px} y1={c.py} x2={c.box.x + c.box.w / 2} y2={c.box.y + c.box.h / 2}/><circle className="bo-true" cx={c.px} cy={c.py} r="3.5" fill={c.names[0].color}/></>}
          <rect x={c.box.x} y={c.box.y} width={c.box.w} height={c.box.h} rx="10" fill={c.names[0].color}/>
          <text x={c.box.x + c.box.w / 2} y={c.box.y + 14} textAnchor="middle" className="bo-letter">{c.t}</text></g>)}
        {hover && <circle className="bo-hover" cx={x(hover.acceptance)} cy={y(hover.value_inr)} r="5" pointerEvents="none"/>}
      </svg>
      {hm && <div className="bo-tip" style={{ left: `${(x(hm.acceptance) / W) * 100}%` }}>{hm.names.map(m => <span key={m.key}><b>{m.key} · {m.label}</b> {share(m.acceptance)} likely to accept · {money(m.value_inr)}{m.detail ? ` · ${m.detail}` : ''}</span>)}</div>}
      {!hm && hover && <div className="bo-tip" style={{ left: `${(x(hover.acceptance) / W) * 100}%` }}><b>{pct(hover.rate)} · {years(hover.tenure_months)}</b>
        <span>{money(hover.loan_amount)}, fee {pct(hover.fee_pct)}</span>{cautions(hover).length > 0 && <span>Caution: {cautions(hover).join(', ')}</span>}<span>{share(hover.acceptance)} likely to accept · {money(hover.value_inr)}</span></div>}
    </div>
    <p className="rv-note">Hover or tab to a lettered marker for the offer behind it. A thin line joins a moved marker to its true point.</p>
  </figure>;
}

// The conclusion of the Pricing intelligence tab: the best offers the four levers allow, set
// beside the RM's ask and the recommendation. Decision support: the engine sets the price.
export function BestOffer({ requestId }) {
  const [d, setD] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { setD(null); setError(''); apiOnce(`/pricing/rate-requests/${encodeURIComponent(requestId)}/best-offer`).then(setD).catch(e => setError(e.message)); }, [requestId]);
  if (error) return <InfoBox warning><span role="alert">{error}</span></InfoBox>;
  if (!d) return <section className="rv-card"><p className="rv-muted"><CircleNotch className="spin" size={16}/> Searching rates, tenures, amounts and fees for the best offers…</p></section>;
  const rec = d.references.recommendation, ask = d.references.ask, own = d.own;
  const cols = [{ key: 'ask', letter: 'A', title: 'RM’s ask', offer: ask, note: ask.eligible ? null : 'Outside policy for this file' },
    { key: 'rec', letter: 'R', title: 'Recommendation', offer: rec },
    ...d.best.map((b, i) => ({ key: b.key, letter: String(i + 1), color: BEST, title: b.title, goal: goal(b), offer: b.offer, reason: b.reason, sentence: b.offer?.sentence }))];
  const diff = {
    rate: o => Math.abs(o.rate - rec.rate) > 0.004, loan_amount: o => Math.abs(o.loan_amount - own.loan_amount) > 0.5,
    cautions: o => cautions(o).length > 0, tenure_months: o => o.tenure_months !== own.tenure_months, fee_pct: o => Math.abs(o.fee_pct - own.fee_pct) > 1e-9 };
  const rows = [['Rate', 'rate', o => pct(o.rate)], ['Loan amount', 'loan_amount', o => money(o.loan_amount)], ['Tenure', 'tenure_months', o => years(o.tenure_months)],
    ['Processing fee', 'fee_pct', o => pct(o.fee_pct)], ['EMI', null, o => money(o.emi)], ['APR, fee included', null, o => pct(o.apr_pct)],
    ['Grade and default risk', null, o => `${o.risk_grade} · ${(o.pd_annual * 100).toFixed(2)}%`], ['Likely to accept', null, o => share(o.acceptance)],
    ['Risk-adjusted value', null, o => money(o.value_inr)],
    ['Cautions', 'cautions', o => cautions(o).join(', ') || '—']];
  const detail = o => `${pct(o.rate)}, ${years(o.tenure_months)}, ${money(o.loan_amount)}, fee ${pct(o.fee_pct)}`;
  const marks = [{ key: 'A', label: 'RM’s ask', acceptance: ask.acceptance, value_inr: ask.value_inr, color: MARK, kind: 'ref', detail: detail(ask) },
    { key: 'R', label: 'Recommendation', acceptance: rec.acceptance, value_inr: rec.value_inr, color: REC, kind: 'ref', detail: detail(rec) },
    ...d.best.map((b, i) => b.offer && ({ key: String(i + 1), label: b.title, acceptance: b.offer.acceptance, value_inr: b.offer.value_inr, color: BEST, kind: 'best', detail: detail(b.offer) })).filter(Boolean)]
    .filter(m => m.acceptance != null && m.value_inr != null);
  const bests = d.best.map(b => ({ key: b.key, name: b.title, kind: 'best', offer: b.offer, reason: b.reason }));
  bests.forEach((r, i) => { if (r.offer) { const j = bests.slice(0, i).find(o => sameLevers(o.offer, r.offer)); if (j) r.same = j.name; } });
  const compareRows = [{ key: 'ask', name: 'RM’s ask', kind: 'ref', offer: ask, outside: !ask.eligible },
    { key: 'rec', name: 'Recommendation', kind: 'ref', offer: rec }, ...bests];
  return <section className="rv-card bo" aria-labelledby="bo-title">
    <h2 id="bo-title">Best offer</h2>
    <p className="rv-note">The ask and the recommendation, and the best offers the four levers allow. Only offers inside policy can be best.</p>
    <h3>Compare the offers</h3>
    <Compare rows={compareRows} own={own} rec={rec}/>
    <p className="rv-note">Each column of the table below is one offer for this file. Shaded cells are levers that differ from the file as it stands.</p>
    <div className="table-scroll"><table className="sc-table bo-table">
      <caption className="sr-only">The ask, the recommendation and the best offers by objective</caption>
      <thead><tr><th scope="col"><span className="sr-only">Measure</span></th>{cols.map(c => <th scope="col" key={c.key} className={c.key === 'rec' ? 'rec' : ''}><b className="bo-chip" style={{ background: c.color || (c.key === 'ask' ? MARK : REC) }}>{c.letter}</b> {c.title}{c.goal && <small>{c.goal}</small>}{c.note && <small className="bad">{c.note}</small>}</th>)}</tr></thead>
      <tbody>{rows.map(([label, lever, f]) => <tr key={label}><th scope="row">{label}</th>
        {cols.map(c => c.offer ? <td key={c.key} className={lever && diff[lever](c.offer) ? 'chg' : ''}>{f(c.offer)}</td>
          : lever === 'rate' ? <td key={c.key} rowSpan={rows.length} className="none">None qualifies<small>{c.reason}</small></td> : null)}</tr>)}</tbody>
      <tfoot><tr><th scope="row">In plain words</th>{cols.map(c => <td key={c.key} className="say">{c.sentence || (c.key === 'ask' ? 'What the relationship manager asked for.' : c.key === 'rec' ? 'The system’s recommended rate on the file as it stands.' : '')}</td>)}</tr></tfoot>
    </table></div>
    <Fold title={`See every offer (${d.counts.eligible.toLocaleString('en-IN')} eligible)`} summary="Every eligible offer by likelihood to accept and value, with the best trade-offs"><Scatter d={d} marks={marks}/></Fold>
    <h3>What each lever buys on its own</h3>
    <p className="rv-note">One change at a time from the recommendation: {pct(rec.rate)}, {money(own.loan_amount)}, {years(own.tenure_months)}, {pct(own.fee_pct)} fee.</p>
    <div className="table-scroll"><table className="sc-table bo-table">
      <caption className="sr-only">Each single lever change against the recommendation</caption>
      <thead><tr><th scope="col">Change</th><th scope="col">Rate</th><th scope="col">Likely to accept</th><th scope="col">Value</th><th scope="col">Inside policy</th></tr></thead>
      <tbody>{d.levers.map(l => <tr key={`${l.lever}-${l.label}`} className={l.eligible ? '' : 'out'}>
        <th scope="row">{l.label}<small>{{ rate: 'Rate', tenure: 'Tenure', fee: 'Processing fee', amount: 'Loan amount' }[l.lever]}{l.amount_reduced ? ' · less than asked' : ''}</small></th>
        <td>{pct(l.rate)}</td><td>{share(l.acceptance)} <small>{pts(l.acceptance_change_pts)} pts</small></td><td>{money(l.value_inr)} <small>{rupees(l.value_change_inr)}</small></td>
        <td>{l.eligible ? 'Yes' : <span className="bad">No<small>{l.failures.join(', ').replaceAll('_', ' ')}</small></span>}</td></tr>)}</tbody></table></div>
    <h3>Outside policy</h3>
    {(() => {
      const list = kind => d.ineligible.filter(r => r.kind === kind);
      const say = rs => rs.map(r => `${r.offers.toLocaleString('en-IN')} ${r.words}`).join('; ');
      return <><p className="rv-note">{d.counts.scored.toLocaleString('en-IN')} offers were tried and {d.counts.eligible.toLocaleString('en-IN')} can be offered. {list('policy').length ? `Outside policy: ${say(list('policy'))}. An offer can fail for more than one reason.` : 'None were outside policy.'}</p>
        {list('comfort').length > 0 && <><h3>Outside the comfort limit</h3><p className="rv-note">{say(list('comfort'))}. This is a limit set for this analysis, not a policy rule: it keeps the search from proposing an offer that is much heavier on the customer than the one they asked for.</p></>}</>;
    })()}
    {d.limitations?.length > 0 && <ul className="rk-limits">{d.limitations.map(t => <li key={t}>{t}</li>)}</ul>}
    <p className="rv-note">{d.note}</p>
  </section>;
}
