import React, { useEffect, useState } from 'react';
import { ArrowClockwise, ArrowRight, ArrowSquareOut, Scales, WarningCircle } from '@phosphor-icons/react';
import { api } from './api';
import { useCanOpen } from './access';
import { Badge, Button, Empty, InfoBox, Panel } from './components';
import { PolicyLink } from './PolicyViewer';
import './regulatory.css';

// The RBI directions behind home-loan pricing and model governance, each ask set
// beside what the platform does and a live number that shows it. Gaps and failing
// evidence lead; the full register follows, filterable by the instrument's status.
const COVERAGE_TONE = { in_place: 'green', partial: 'amber', gap: 'red', outside: 'grey' };
const STATUS_TONE = { in_force: 'green', draft: 'amber', report: 'grey' };
const PAGE_NAMES = { record: 'System of Record', xai: 'Model Explainability', config: 'Configuration', models: 'Model Performance',
  governance: 'Approvals & Gates', runtime: 'Runtime & Drift', audit: 'Governance & Audit' };
const day = d => d ? new Date(`${d}T00:00:00`).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : '—';

function OpenPage({ page, navigate }) {
  const canOpen = useCanOpen();
  if (!page || !PAGE_NAMES[page] || !canOpen(page)) return null;
  return <button type="button" className="text-button rg-open" onClick={() => navigate(page)}>{PAGE_NAMES[page]}<ArrowRight size={14}/></button>;
}

function Artefact({ a, labels, statusLabels, policyVersion, navigate }) {
  const counts = a.requirements.reduce((m, r) => ({ ...m, [r.coverage]: (m[r.coverage] || 0) + 1 }), {});
  return <section className="panel rg-art" id={`rbi-${a.id}`}>
    <div className="rg-art-head">
      <div><h2>{a.title}</h2>
        <p className="rg-instrument">{a.instrument}</p>
        <p className="rg-ref"><span>{a.reference}</span><span>Issued {day(a.issued)}</span>
          {a.source && <a href={a.source} target="_blank" rel="noopener noreferrer">Source <ArrowSquareOut size={12}/></a>}</p></div>
      <div className="rg-art-badges"><Badge tone={STATUS_TONE[a.status]}>{statusLabels[a.status] || a.status}</Badge>
        {Object.keys(labels).filter(k => counts[k]).map(k => <span key={k} className={`rg-count ${COVERAGE_TONE[k]}`}>{counts[k]} {labels[k].toLowerCase()}</span>)}</div>
    </div>
    {a.status_note && <p className="rg-note">{a.status_note}</p>}
    {a.why && <p className="rg-why">{a.why}</p>}
    <div className="table-scroll"><table className="rg-table">
      <thead><tr><th>What RBI asks</th><th>What the platform does</th><th>Coverage</th></tr></thead>
      <tbody>{a.requirements.map((r, i) => <tr key={i} className={r.attention ? 'attention' : ''}>
        <td>{r.ask}</td>
        <td>{r.platform && <p>{r.platform}</p>}
          {r.evidence_text && <p className="rg-evidence"><b>Evidence now</b> {r.evidence_text}</p>}
          {!r.platform && !r.evidence_text && <p className="muted">Handled outside a pricing platform.</p>}
          <div className="rg-links"><OpenPage page={r.page} navigate={navigate}/>
            {r.policy && policyVersion && <PolicyLink version={policyVersion}>Read policy {policyVersion}</PolicyLink>}</div></td>
        <td><Badge tone={COVERAGE_TONE[r.coverage]}>{labels[r.coverage] || r.coverage}</Badge></td>
      </tr>)}</tbody></table></div>
  </section>;
}

export function Regulatory({ navigate }) {
  const [d, setD] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('all');
  const load = () => { setLoading(true); setError(''); api('/governance/regulatory').then(setD).catch(e => setError(e.message)).finally(() => setLoading(false)); };
  useEffect(load, []);

  if (error && !d) return <div className="operations"><InfoBox warning><span role="alert">{error}</span></InfoBox></div>;
  if (!d) return <div className="operations"><Empty icon={Scales} title="Reading the RBI register">Working out the evidence…</Empty></div>;

  const labels = d.coverage_labels || {};
  const statusLabels = d.status_labels || {};
  const by = d.summary.by_coverage;
  const shown = d.artefacts.filter(a => status === 'all' || a.status === status);
  const filters = [['all', 'All'], ...Object.entries(statusLabels).filter(([k]) => d.artefacts.some(a => a.status === k))];
  const jump = id => document.getElementById(`rbi-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  return <div className="operations rg">
    <div className="page-intro"><h1>RBI home loan artefacts</h1>
      <p>The RBI directions, circulars and drafts that bear on how a home loan is priced and how the pricing
        models are governed: what each asks for, what this platform does about it, and the evidence as it stands now.</p></div>
    <p className="rg-meta">{d.entity} · References checked {day(d.checked_on)} · {d.version}</p>

    <div className="model-metrics">
      <div><small>Artefacts</small><b>{d.summary.artefacts}</b></div>
      {Object.entries(labels).map(([k, label]) => <div key={k} className={`rg-tile ${COVERAGE_TONE[k]}`}><small>{label}</small><b>{by[k] || 0}</b></div>)}
    </div>

    {d.attention.length > 0 && <Panel title="Needs attention" icon={WarningCircle} className="rg-attention">
      <ul>{d.attention.map((g, i) => <li key={i}>
        <Badge tone={COVERAGE_TONE[g.coverage]}>{labels[g.coverage]}</Badge>
        <div><b>{g.artefact}</b><span>{g.ask}</span><p>{g.evidence}</p>
          <div className="rg-links"><button type="button" className="text-button" onClick={() => jump(g.id)}>See the artefact<ArrowRight size={14}/></button>
            <OpenPage page={g.page} navigate={navigate}/></div></div>
      </li>)}</ul>
    </Panel>}

    <div className="rg-bar">
      <div className="tabs" role="tablist" aria-label="Filter by status">{filters.map(([k, label]) =>
        <button key={k} role="tab" aria-selected={k === status} className={k === status ? 'active' : ''} onClick={() => setStatus(k)}>
          {label} <small>{k === 'all' ? d.artefacts.length : d.artefacts.filter(a => a.status === k).length}</small></button>)}</div>
      <Button busy={loading} onClick={load}><ArrowClockwise/> Refresh evidence</Button>
    </div>

    {shown.map(a => <Artefact key={a.id} a={a} labels={labels} statusLabels={statusLabels} policyVersion={d.policy_version} navigate={navigate}/>)}

    <p className="rg-foot">This register maps the platform to RBI instruments as summarised on the date above; it is not legal advice.
      Directions are amended and drafts change before they are final, so read the current text on rbi.org.in before relying on a row.</p>
  </div>;
}
