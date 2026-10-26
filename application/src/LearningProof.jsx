import React, { useEffect, useState } from 'react';
import { ArrowClockwise, CheckCircle, CircleDashed, CircleNotch, Play } from '@phosphor-icons/react';
import { api, titleCase } from './api';
import { Button, InfoBox } from './components';
import './learning-proof.css';

const share = r => r == null ? '—' : `${Math.round(r * 100)}%`;
const day = iso => iso ? new Date(iso).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—';
const fixed = (n, d = 2) => n == null ? '—' : Number(n).toFixed(d);
const STRATEGY = { fedavg: 'FedAvg', fedprox: 'FedProx' };
const MODELS = { conversion: 'Conversion', lgd: 'Loss given default', override: 'Override prediction',
  prepayment: 'Prepayment', risk: 'Risk (probability of default)', stage1: 'Indicative band', stage2: 'Rate anchor', uplift: 'Price uplift' };

function Claims({ claims }) {
  return <ol className="lp-claims">{claims.map(c => <li key={c.claim} className={c.proven ? 'proven' : 'open'}>
    {c.proven ? <CheckCircle size={20} weight="fill"/> : <CircleDashed size={20}/>}
    <div><b>{c.claim}</b><span>{c.evidence}</span></div>
    <em>{c.proven ? 'Proven' : 'Not yet'}</em></li>)}</ol>;
}

// A row of values drawn as bars against a common scale: lower or higher is better.
function Bar({ value, min, max, good = 'high' }) {
  const span = Math.max(max - min, 1e-9);
  const width = 12 + ((value - min) / span) * 88;
  return <span className={`lp-bar ${good}`}><i style={{ width: `${width}%` }}/></span>;
}

// How we prove the engine learns: each claim, the evidence on file, and how to check it.
export function LearningProof({ notify }) {
  const [tab, setTab] = useState('continuous');
  return <div className="operations lp">
    <div className="page-intro"><h1>Learning Proof</h1>
      <p>Each claim about how the engine learns, the evidence on file for it, and how to check it yourself. A claim
        without evidence is marked “Not yet” rather than assumed.</p></div>
    <div className="lp-tabs" role="tablist">
      {[['continuous', 'Continuous learning'], ['federated', 'Federated learning']].map(([k, l]) =>
        <button key={k} type="button" role="tab" aria-selected={tab === k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{l}</button>)}
    </div>
    {tab === 'continuous' ? <Continuous notify={notify}/> : <Federated notify={notify}/>}
  </div>;
}

function Continuous({ notify }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [running, setRunning] = useState(false);
  const load = () => api('/learning/continuous').then(d => { setData(d); setError(''); }).catch(e => setError(e.message));
  useEffect(() => { load(); }, []);

  async function runChecks() {
    setRunning(true);
    try { await api('/governance/monitors/run', { monitor: null }); notify?.('Drift and retraining checks ran and were recorded.'); await load(); }
    catch (e) { notify?.(e.message, 'error'); } finally { setRunning(false); }
  }

  if (error) return <InfoBox warning><span role="alert">{error}</span></InfoBox>;
  if (!data) return <p className="lp-loading"><CircleNotch className="spin" size={16}/> Gathering the evidence and re-running the backtest…</p>;
  const { claims, loop, checks, curve, promotion } = data;
  const aucs = curve.rows.map(r => r.auc), losses = curve.rows.map(r => r.log_loss);
  return <div className="lp-body">
    <Claims claims={claims}/>

    <section className="lp-card">
      <h2>More outcomes, better predictions</h2>
      <p className="lp-note">The conversion model is refitted on a growing window of history, from the first 3 months to everything
        before {curve.holdout_from}. Every fit is scored on the same {curve.holdout_files.toLocaleString('en-IN')} files from {curve.holdout_from} to {curve.holdout_to},
        which none of them saw. The only thing that changes is how much the model has learned from.</p>
      <table className="lp-table"><thead><tr><th>History used</th><th>Files</th><th>Trained to</th><th>AUC (ranking who takes the loan)</th><th>Log loss (lower is better)</th><th>Brier</th></tr></thead>
        <tbody>{curve.rows.map(r => <tr key={r.months_of_history}>
          <td>{r.months_of_history} months</td><td>{r.files.toLocaleString('en-IN')}</td><td>{r.trained_to}</td>
          <td><Bar value={r.auc} min={Math.min(...aucs)} max={Math.max(...aucs)}/>{fixed(r.auc, 3)}</td>
          <td><Bar value={-r.log_loss} min={-Math.max(...losses)} max={-Math.min(...losses)}/>{fixed(r.log_loss, 4)}</td>
          <td>{fixed(r.brier, 4)}</td></tr>)}</tbody></table>
      <p className="lp-result">From the first to the latest window: AUC <b>+{fixed(curve.gain.auc_pts)} points</b>, log loss <b>−{fixed(curve.gain.log_loss_pct)}%</b>, Brier <b>−{fixed(curve.gain.brier_pct)}%</b>{curve.improves_every_step ? ', better at every step.' : '.'}</p>
    </section>

    <section className="lp-card">
      <div className="lp-card-head"><h2>The loop, on file</h2>
        <Button busy={running} onClick={runChecks}><Play size={16}/> Run checks now</Button></div>
      <div className="lp-steps">
        <div><small>1 · Decisions recorded</small><b>{loop.decisions}</b><em>{loop.decisions_with_versions} carry model and policy versions · latest {day(loop.latest_decision?.created_at)}</em></div>
        <div><small>2 · Overrides captured</small><b>{loop.overrides}</b><em>each with its reason and approver</em></div>
        <div><small>3 · Outcomes joined back</small><b>{loop.outcomes}</b><em>{loop.outcomes_joined_to_decision} joined to the decision they judge</em></div>
        <div><small>4 · Checks run</small><b>{checks.history.length ? checks.history.length : 0}</b><em>{checks.never_run.length ? `Never run: ${checks.never_run.map(titleCase).join(', ')}` : 'every check has run'}</em></div>
      </div>
      {loop.recent_outcomes.length > 0 && <><h3>Outcomes against what was expected</h3>
        <table className="lp-table"><thead><tr><th>File</th><th>Recorded</th><th>Customer</th><th>Expected to take the loan</th><th>Took it</th><th>Rate vs recommendation</th></tr></thead>
          <tbody>{loop.recent_outcomes.map(o => <tr key={o.loan_id + o.updated_at}>
            <td>{o.loan_id}</td><td>{day(o.updated_at)}</td><td>{titleCase(o.customer_response || '—')}</td>
            <td>{share(o.expected_conversion)}</td><td>{o.converted ? 'Yes' : 'No'}</td>
            <td>{o.rate_delta_bps == null ? '—' : `${o.rate_delta_bps > 0 ? '+' : ''}${o.rate_delta_bps} bps`}</td></tr>)}</tbody></table></>}
      {checks.history.length > 0 && <><h3>Latest checks</h3>
        <table className="lp-table"><thead><tr><th>Check</th><th>Ran</th><th>By</th><th>Result</th><th>Retrain?</th></tr></thead>
          <tbody>{checks.history.slice(0, 6).map(h => <tr key={h.run_id}>
            <td>{titleCase(h.monitor)}</td><td>{day(h.started_at)}</td><td>{h.actor}</td><td>{titleCase(h.status)}</td>
            <td>{h.monitor === 'retraining' ? (h.triggered ? 'Yes' : 'No') : '—'}</td></tr>)}</tbody></table></>}
    </section>

    <section className="lp-card">
      <h2>Retrain, compare, promote</h2>
      <p className="lp-note">{promotion.status}</p>
      <ul className="lp-gates">
        <li>A challenger is compared in shadow on at least <b>{promotion.gates.shadow_min_decisions}</b> decisions.</li>
        <li>It must beat the champion’s expected value by <b>{promotion.gates.promote_if_better_by_pct}%</b>…</li>
        <li>…without lowering conversion by more than <b>{promotion.gates.max_conversion_drop_pct} {promotion.gates.max_conversion_drop_pct === 1 ? 'point' : 'points'}</b>.</li>
        <li>{promotion.gates.require_human_signoff ? 'A named person from model risk signs it off; nothing promotes itself.' : 'Promotion does not need a sign-off.'}</li>
      </ul>
      {promotion.artefacts.length > 0 && <table className="lp-table"><thead><tr><th>Model in service</th><th>Version</th><th>Trained</th><th>Headline</th></tr></thead>
        <tbody>{promotion.artefacts.map(m => <tr key={m.name}><td>{MODELS[m.name] || titleCase(m.name)}</td><td>{m.version}</td><td>{day(m.trained_at)}</td>
          <td>{m.metric ? `${m.metric} ${fixed(m.headline, 3)}` : '—'}</td></tr>)}</tbody></table>}
    </section>

    <details className="lp-how" open><summary>How to check it yourself</summary><ol>
      <li>Price a file, record an override and then the customer’s decision: the counts in “The loop, on file” go up, and the outcome appears against what was expected.</li>
      <li>Press “Run checks now”: drift and retraining checks run on the loan book and their results are recorded with your name.</li>
      <li>The backtest is recomputed from the loan book when the service starts; the held-out months are never used for training.</li>
      <li>For a technical check: <code>make monitors</code> runs the same checks from the command line, and <code>pytest tests/test_learning_proof.py</code> asserts every claim above.</li>
    </ol></details>
  </div>;
}

function Federated({ notify }) {
  const [strategy, setStrategy] = useState('fedavg');
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [running, setRunning] = useState(false);
  useEffect(() => {
    setData(null);
    api(`/learning/federated?strategy=${strategy}`).then(d => { setData(d); setError(''); }).catch(e => setError(e.message));
  }, [strategy]);

  async function record() {
    setRunning(true);
    try { const d = await api('/learning/federated', { strategy }); setData(d); notify?.(`Federated round recorded as ${d.run_id}.`); }
    catch (e) { notify?.(e.message, 'error'); } finally { setRunning(false); }
  }

  if (error) return <InfoBox warning><span role="alert">{error}</span></InfoBox>;
  if (!data) return <p className="lp-loading"><CircleNotch className="spin" size={16}/> Running the federated rounds…</p>;
  const errs = data.history.map(h => h.global_mae_bps);
  const shown = data.history.filter(h => [0, 1, 2, 3, 5, 10, 20].includes(h.round) || h.round === data.rounds);
  return <div className="lp-body">
    <Claims claims={data.claims}/>

    <section className="lp-card">
      <div className="lp-card-head"><h2>Five states, one shared model, no pooled data</h2>
        <div className="lp-actions">
          <select value={strategy} onChange={e => setStrategy(e.target.value)} aria-label="Strategy">
            <option value="fedavg">FedAvg</option><option value="fedprox">FedProx</option></select>
          <Button primary busy={running} onClick={record}><ArrowClockwise size={16}/> Run and record</Button></div></div>
      <p className="lp-note">Each state learns the regional discount ({data.target}) from its own files. Every round it receives the
        shared model, trains on its files, and sends back only {data.parameters_per_state} numbers. Errors are on each state’s own
        held-out files, in basis points.</p>
      <div className="lp-steps">
        <div><small>Before any round</small><b>{fixed(data.mae_before_bps)} bps</b><em>shared model untrained</em></div>
        <div><small>After {data.rounds} rounds</small><b>{fixed(data.mae_after_bps)} bps</b><em>{STRATEGY[data.strategy] || data.strategy}{data.proximal_mu ? ` · μ ${data.proximal_mu}` : ''}</em></div>
        <div><small>All data pooled centrally</small><b>{fixed(data.mae_centralised_bps)} bps</b><em>what federation must match, without pooling</em></div>
        <div><small>Run</small><b className="lp-id">{data.run_id}</b><em>{data.recorded ? 'on file' : 'not recorded yet'}</em></div>
      </div>
      <table className="lp-table"><thead><tr><th>State</th><th>Files kept in the state</th><th>Fingerprint</th><th>Before</th><th>Federated</th><th>Pooled centrally</th><th>Own data only</th><th>Personalised</th></tr></thead>
        <tbody>{data.states.map(s => <tr key={s.state}><td>{s.state}</td><td>{s.files.toLocaleString('en-IN')}</td><td><code>{s.fingerprint}</code></td>
          <td>{fixed(s.before_mae_bps)}</td><td><b>{fixed(s.federated_mae_bps)}</b></td><td>{fixed(s.centralised_mae_bps)}</td>
          <td>{fixed(s.local_mae_bps)}</td><td className={s.personalised_mae_bps < s.federated_mae_bps ? 'better' : ''}>{fixed(s.personalised_mae_bps)}</td></tr>)}</tbody></table>
      <h3>Error by round</h3>
      <table className="lp-table lp-rounds"><tbody>{shown.map(h => <tr key={h.round}><td>Round {h.round}</td>
        <td><Bar value={-h.global_mae_bps} min={-Math.max(...errs)} max={-Math.min(...errs)}/>{fixed(h.global_mae_bps, 3)} bps</td></tr>)}</tbody></table>
      <p className="lp-note">Most of the learning happens in the first round. After that the shared model settles, and the movement is a few hundredths of a basis point.</p>
      <p className="lp-note">{data.note}</p>
    </section>

    {data.recent_runs.length > 0 && <section className="lp-card"><h2>Recorded rounds</h2>
      <table className="lp-table"><thead><tr><th>Run</th><th>Recorded</th><th>Strategy</th><th>States</th><th>Rounds</th><th>Shared model</th><th>Error</th></tr></thead>
        <tbody>{data.recent_runs.map(r => <tr key={r.run_id}><td><code>{r.run_id}</code></td><td>{day(r.created_at)}</td><td>{STRATEGY[r.strategy] || r.strategy}</td>
          <td>{r.participants}</td><td>{r.rounds}</td><td><code>{r.global_version}</code></td><td>{fixed(r.final_mae_bps)} bps</td></tr>)}</tbody></table></section>}

    <details className="lp-how" open><summary>How to check it yourself</summary><ol>
      <li>Compare “Federated” with “Pooled centrally”: the shared model reaches the accuracy of having every state’s files in one place, though no file left its state.</li>
      <li>The fingerprint is a hash of each state’s own files, computed where they are; only the {data.parameters_per_state} numbers per round cross between states.</li>
      <li>Press “Run and record” twice: the same data and settings give the same run id, because the id is a hash of what was trained.</li>
      <li>For a technical check: <code>pytest tests/test_learning_proof.py</code>, and <code>make verify-algorithms</code> runs every aggregation strategy.</li>
    </ol></details>
  </div>;
}
