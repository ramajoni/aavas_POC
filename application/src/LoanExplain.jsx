import React, {useEffect, useMemo, useState} from 'react';
import {Chart as ChartJS, CategoryScale, LinearScale, BarElement, Tooltip} from 'chart.js';
import {Bar} from 'react-chartjs-2';
import {Sparkle, ArrowClockwise, CheckCircle, Warning, X, ChartBar, ListNumbers, Lightbulb, Scales} from '@phosphor-icons/react';
import {Panel, Button, Badge, InfoBox, Empty} from './components';
import {api, money, percent, titleCase} from './api';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

// Opened in its own window from the System Recommendation Rationale panel:
// #/explain/<decision_id>?f=<feature> - the feature is highlighted on arrival.
export function parseExplainHash(hash) {
  const match = /^#\/explain\/([^?]+)(?:\?f=(.*))?$/.exec(hash || '');
  return match ? {decisionId: decodeURIComponent(match[1]), feature: match[2] ? decodeURIComponent(match[2]) : null} : null;
}

export function openExplain(decisionId, feature) {
  const url = `${location.pathname}#/explain/${encodeURIComponent(decisionId)}${feature ? `?f=${encodeURIComponent(feature)}` : ''}`;
  // One window per decision: selecting another driver re-targets it rather than stacking windows.
  const win = window.open(url, `aavas-explain-${decisionId}`, 'popup=yes,width=1320,height=900');
  win?.focus();
  return win;
}

// A reused window only sees its hash change, so the route has to follow it.
export function ExplainRoute({initial}) {
  const [route, setRoute] = useState(initial);
  useEffect(() => {
    const listener = () => { const next = parseExplainHash(location.hash); if (next) setRoute(next); };
    window.addEventListener('hashchange', listener);
    return () => window.removeEventListener('hashchange', listener);
  }, []);
  return <LoanExplain decisionId={route.decisionId} feature={route.feature}/>;
}

export const methodName = m => m === 'shap' ? 'SHAP' : titleCase(m);

const TABS = [['stage2', 'Rate model'], ['risk', 'Risk model'], ['conversion', 'Conversion model']];
const GROUPS = {customer_risk: 'Customer risk', commercial: 'Commercial pressure'};
const MONEY = new Set(['verified_monthly_income', 'declared_monthly_income', 'sanctioned_loan_amount', 'requested_loan_amount', 'appraised_property_value', 'estimated_property_value', 'declared_obligations']);
const RATIO = new Set(['ltv', 'estimated_ltv', 'foir', 'income_verification_haircut']);
const RATE = new Set(['competitor_rate', 'offered_rate', 'ai_recommended_rate']);
const PCT = new Set(['sales_target_pct', 'branch_target_pct']);
const BOTTOM_STEPS = 10;

export function featureValue(feature, value) {
  if (value == null) return '—';
  // Categorical flags come back decoded as the strings "True" / "False".
  if (value === 'True' || value === 'False') value = value === 'True';
  if (feature === 'new_to_credit') return value ? 'New to credit' : 'Established';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string') return titleCase(value);
  if (MONEY.has(feature)) return money(value);
  if (RATIO.has(feature)) return percent(value * 100, 1);
  if (RATE.has(feature)) return percent(value);
  if (PCT.has(feature)) return percent(value, 0);
  if (feature === 'competitor_gap_bps') return `${value > 0 ? '+' : ''}${Number(value).toFixed(0)} bps`;
  if (['legal_flag', 'technical_flag', 'month_end', 'new_to_credit', 'customer_negotiating', 'existing_relationship', 'competitor_evidence_verified'].includes(feature)) return value ? 'Yes' : 'No';
  return Number(value).toLocaleString('en-IN', {maximumFractionDigits: 2});
}

const bps = n => n == null ? '—' : `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n).toFixed(1)} bps`;
const signed = (n, d = 3) => n == null ? '—' : `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n).toFixed(d)}`;
const adverse = (model, f) => f.direction === 'none' ? null : (f.direction === 'increase') === model.positive_is_adverse;

export function LoanExplain({decisionId, feature}) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState('');
  const [tab, setTab] = useState('stage2');
  const [group, setGroup] = useState('all');

  useEffect(() => {
    document.title = `Loan explanation · ${decisionId}`;
    api(`/pricing/explain/${encodeURIComponent(decisionId)}`).then(setDetail).catch(e => setError(e.message));
  }, [decisionId]);

  useEffect(() => {
    if (!feature) return;
    // Driver links come from the rate model's reason codes.
    setTab('stage2');
    setGroup('all');
  }, [feature]);

  useEffect(() => {
    if (!detail || !feature) return;
    const row = document.getElementById(`lx-feature-${feature}`);
    row?.scrollIntoView({block: 'center', behavior: 'smooth'});
  }, [detail, feature, tab]);

  if (error) return <Shell><Empty icon={Warning} title="This explanation could not be loaded">{error}</Empty></Shell>;
  if (!detail) return <Shell><div className="lx-loading" role="status">Computing SHAP attributions for every model…</div></Shell>;

  const {rate, outcomes, applicant} = detail;
  const model = detail.models[tab];
  return <Shell detail={detail}>
    <div className="lx-kpis">
      <Kpi label="Recommended rate" value={percent(rate.recommended)} strong/>
      <Kpi label="Card rate" value={percent(rate.card)}/>
      <Kpi label="Model anchor" value={percent(rate.model_anchor)} hint="Rate model output"/>
      <Kpi label="Typical customer" value={percent(rate.typical_customer)} hint="SHAP baseline"/>
      <Kpi label="Risk" value={outcomes.risk ? `Grade ${outcomes.risk.risk_grade}` : '—'} hint={outcomes.risk?.pd != null ? `PD ${percent(outcomes.risk.pd * 100)} p.a.` : null}/>
      <Kpi label="Conversion" value={outcomes.conversion_probability != null ? percent(outcomes.conversion_probability * 100, 1) : '—'} hint="At the recommended rate"/>
      <Kpi label="Approval needed" value={titleCase(rate.required_approval)}/>
    </div>
    <p className="lx-applicant">{titleCase(applicant.segment)} · {applicant.city} · {titleCase(applicant.property_type)} · {money(applicant.sanctioned_loan_amount)} over {applicant.tenure_years} years · LTV {featureValue('ltv', applicant.ltv)} · bureau {applicant.bureau_score}</p>
    {detail.warnings.map(w => <InfoBox warning key={w}>{w}</InfoBox>)}

    <Panel title="How the rate was built" icon={ChartBar}>
      <Waterfall bridge={detail.bridge}/>
      <p className="tiny-note">SHAP explains the rate model's anchor, starting from the typical customer. The last step is yield optimisation and policy, which uses conversion, expected loss and loan life rather than the customer's features.</p>
    </Panel>

    <div className="lx-main">
      <Panel title="Feature weightage (SHAP)" icon={ListNumbers} className="lx-weights">
        <div className="segmented" role="tablist">{TABS.filter(([k]) => detail.models[k]).map(([k, label]) =>
          <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{label}</button>)}</div>
        <p className="lx-model-note"><b>{model.title}.</b> {model.features.length} features, explained with {methodName(model.method)}. A positive value {model.positive_means}.{model.units === 'log_odds' && ' Contributions are in log-odds; the weight share is each feature\'s part of the total movement.'}</p>
        <GroupSplit model={model} group={group} setGroup={setGroup}/>
        <FeatureTable model={model} group={group} highlight={tab === 'stage2' ? feature : null}/>
      </Panel>
      <Narrative decisionId={detail.decision_id} status={detail.narrative}/>
    </div>

    <div className="lx-bottom">
      <Panel title="What the customer could change" icon={Lightbulb}>
        {detail.context.actionable_for_customer.length
          ? <ul className="lx-list">{detail.context.actionable_for_customer.map(a => <li key={a.text}>{a.text}</li>)}</ul>
          : <p className="muted">No realistic change to the file would materially improve the rate.</p>}
      </Panel>
      <Panel title="Fair-pricing review" icon={Scales}>
        <Badge tone={detail.context.fair_pricing_review?.flagged ? 'amber' : 'green'}>{detail.context.fair_pricing_review?.flagged ? 'Review required' : 'No review needed'}</Badge>
        <p className="small-copy">{detail.context.fair_pricing_review?.reason}</p>
      </Panel>
      <Panel title="Caveats for the customer">
        <ul className="lx-list">{detail.context.customer_caveats.map(c => <li key={c}>{c}</li>)}</ul>
      </Panel>
    </div>
    <p className="lx-disclaimer">{detail.context.disclaimer}</p>
  </Shell>;
}

function Shell({detail, children}) {
  return <div className="lx-page">
    <header className="lx-header">
      <img src="/assets/aavas-logo.png" alt="Aavas Financiers"/>
      <div className="lx-title">
        <h1>Loan explanation</h1>
        {detail && <p>Loan <b>{detail.loan_id}</b> · Decision {detail.decision_id} · {detail.created_at ? new Date(detail.created_at).toLocaleString('en-IN') : ''}</p>}
      </div>
      <Button onClick={() => window.close()}><X/> Close window</Button>
    </header>
    <main className="lx-body">{children}</main>
  </div>;
}

function Kpi({label, value, hint, strong}) {
  return <div className={`lx-kpi ${strong ? 'strong' : ''}`}><span>{label}</span><b>{value}</b>{hint && <small>{hint}</small>}</div>;
}

function Waterfall({bridge}) {
  const data = useMemo(() => {
    const steps = bridge.steps.slice(0, BOTTOM_STEPS);
    const rest = bridge.steps.slice(BOTTOM_STEPS);
    const rows = [{label: 'Typical customer', start: 0, end: bridge.baseline_rate, kind: 'total'}];
    let level = bridge.baseline_rate;
    const push = (label, deltaBps) => { const end = level + deltaBps / 100; rows.push({label, start: level, end, kind: deltaBps > 0 ? 'up' : 'down', delta: deltaBps}); level = end; };
    steps.forEach(s => push(s.label, s.delta_bps));
    if (rest.length) push(`Other ${rest.length} features`, rest.reduce((t, s) => t + s.delta_bps, 0));
    rows.push({label: 'Model anchor', start: 0, end: bridge.model_rate, kind: 'subtotal'});
    rows.push({label: 'Optimisation & policy', start: bridge.model_rate, end: bridge.recommended_rate, kind: bridge.optimisation_bps > 0 ? 'up' : 'down', delta: bridge.optimisation_bps});
    rows.push({label: 'Recommended', start: 0, end: bridge.recommended_rate, kind: 'final'});
    return rows;
  }, [bridge]);
  const values = data.flatMap(r => r.kind === 'up' || r.kind === 'down' ? [r.start, r.end] : [r.end]);
  const min = Math.floor((Math.min(...values) - 0.25) * 4) / 4;
  const max = Math.ceil((Math.max(...values) + 0.25) * 4) / 4;
  const color = {total: '#7d93c4', subtotal: '#08277f', final: '#0756ff', up: '#f34854', down: '#29a96c'};
  return <div className="lx-waterfall">
    <Bar
      aria-label="Waterfall from the typical customer's rate through each feature to the recommended rate"
      data={{labels: data.map(r => r.label), datasets: [{data: data.map(r => [r.kind === 'up' || r.kind === 'down' ? r.start : min, r.end]), backgroundColor: data.map(r => color[r.kind]), borderRadius: 3, barPercentage: 0.75}]}}
      options={{maintainAspectRatio: false, plugins: {legend: {display: false}, tooltip: {callbacks: {label: ctx => { const r = data[ctx.dataIndex]; return r.delta != null ? `${bps(r.delta)} → ${r.end.toFixed(2)}%` : `${r.end.toFixed(2)}%`; }}}},
        scales: {y: {min, max, ticks: {callback: v => `${Number(v).toFixed(2)}%`, font: {size: 10}}, grid: {color: '#edf1f8'}}, x: {ticks: {font: {size: 10}, maxRotation: 50, minRotation: 30}, grid: {display: false}}}}}/>
  </div>;
}

function GroupSplit({model, group, setGroup}) {
  return <div className="lx-groups">
    {[['all', 'All features', null], ...Object.entries(GROUPS).map(([k, label]) => [k, label, model.groups[k]])].map(([k, label, g]) =>
      <button key={k} className={`lx-group ${group === k ? 'active' : ''}`} onClick={() => setGroup(k)} aria-pressed={group === k}>
        <span>{label}</span>
        {g ? <b>{g.share_pct.toFixed(1)}% of weight</b> : <b>{model.features.length} features</b>}
        {g && <small>{g.count} features · net {model.units === 'rate_pct' ? bps(g.net_bps) : signed(g.net)}</small>}
      </button>)}
  </div>;
}

function FeatureTable({model, group, highlight}) {
  const rows = model.features.filter(f => group === 'all' || f.group === group);
  const maxShare = Math.max(...model.features.map(f => f.share_pct), 1);
  const isRate = model.units === 'rate_pct';
  return <div className="table-scroll lx-table">
    <table>
      <thead><tr><th>#</th><th>Feature</th><th>Value</th><th>Group</th>{isRate && <th>Effect</th>}<th className="num">SHAP {isRate ? '(bps)' : '(log-odds)'}</th><th>Weight share</th></tr></thead>
      <tbody>{rows.map(f => {
        const bad = adverse(model, f);
        return <tr key={f.feature} id={`lx-feature-${f.feature}`} className={highlight === f.feature ? 'lx-highlight' : ''}>
          <td className="muted">{f.rank}</td>
          <td><b>{f.label}</b></td>
          <td>{featureValue(f.feature, f.value)}</td>
          <td><Badge tone={f.group === 'commercial' ? 'amber' : 'blue'}>{GROUPS[f.group]}</Badge></td>
          {isRate && <td className="small-copy">{f.effect}</td>}
          <td className={`num ${bad == null ? '' : bad ? 'red' : 'green-text'}`}>{isRate ? bps(f.impact_bps) : signed(f.contribution)}</td>
          <td><div className="lx-share"><i className={bad ? 'bad' : 'good'} style={{width: `${(f.share_pct / maxShare) * 100}%`}}/><span>{f.share_pct.toFixed(1)}%</span></div></td>
        </tr>;
      })}</tbody>
    </table>
  </div>;
}

function Narrative({decisionId, status}) {
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function generate(refresh = false) {
    setBusy(true); setError('');
    try { setResult(await api(`/pricing/explain/${encodeURIComponent(decisionId)}/narrative${refresh ? '?refresh=true' : ''}`, {})); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  const s = result?.sections;
  const flagged = result?.unverified_figures || [];
  const mark = text => highlightFigures(text, flagged);
  return <Panel title="Written explanation" icon={Sparkle} className="lx-narrative"
    action={result && <button className="text-button" disabled={busy} onClick={() => generate(true)}><ArrowClockwise size={15}/> Regenerate</button>}>
    {!status?.available
      ? <InfoBox warning>The written explanation is not configured. {status?.reason} Add <code>ANTHROPIC_API_KEY</code> to <code>.env</code> and restart the API. Everything else on this page works without it.</InfoBox>
      : !result
        ? <div className="lx-narrative-empty">
            <p>Claude explains this rate in plain language for the credit reviewer, using only the figures on this page. Any figure it did not take from them is flagged.</p>
            <Button primary busy={busy} onClick={() => generate()}><Sparkle/> {busy ? 'Writing… about 20 seconds' : 'Generate explanation'}</Button>
          </div>
        : <div className="lx-narrative-body">
            {flagged.length > 0 && <InfoBox warning>Not found among this page's figures: <b>{flagged.join(', ')}</b>. They are highlighted below. Check them before relying on the text.</InfoBox>}
            <p className="lx-summary">{mark(s.summary)}</p>
            <h3>Why this rate</h3><ul className="lx-list">{s.why_this_rate.map((t, i) => <li key={i}>{mark(t)}</li>)}</ul>
            <h3>Risk view</h3><p>{mark(s.risk_view)}</p>
            <h3>Commercial pressure</h3><p>{mark(s.commercial_view)}</p>
            <h3>What the customer can do</h3><ul className="lx-list">{s.customer_options.map((t, i) => <li key={i}>{mark(t)}</li>)}</ul>
            <h3>Checks for the reviewer</h3><ul className="lx-checks">{s.reviewer_checks.map((t, i) => <li key={i}><CheckCircle size={16}/><span>{mark(t)}</span></li>)}</ul>
            <p className="lx-meta">{result.model} · {new Date(result.generated_at).toLocaleString('en-IN')}{result.cached ? ' · saved copy' : ''}</p>
            <p className="lx-disclaimer">{result.disclaimer}</p>
          </div>}
    {error && <InfoBox warning>{error}</InfoBox>}
  </Panel>;
}

function highlightFigures(text, figures) {
  if (!figures.length || !text) return text;
  const escaped = figures.map(f => f.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const pattern = new RegExp(`(?<![\\w.,])(${escaped.join('|')})(?![\\w]|[.,]\\d)`, 'g');
  return text.split(pattern).map((part, i) => i % 2 ? <mark key={i} title="Not among this page's figures">{part}</mark> : part);
}
