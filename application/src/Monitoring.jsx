import React,{useEffect,useState} from 'react';
import {ChartLineUp,ArrowClockwise,Waveform,Trophy,Timer} from '@phosphor-icons/react';
import {api,number,percent,titleCase} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable} from './components';

// Runtime operations and drift, from analytics/performance.py. Those functions
// were written for a governance dashboard and, until the control tower, nothing
// imported them - the analysis existed and was unreachable.
const tone=s=>s==='ok'?'green':s==='warn'?'amber':s==='alert'?'red':'blue';
const TABS=[['ops','Runtime operations',Timer],['drift','Drift',Waveform],['champions','Champions',Trophy]];

export function Monitoring(){
 const [tab,setTab]=useState('ops');
 const [ops,setOps]=useState(null);const [drift,setDrift]=useState(null);const [gov,setGov]=useState(null);
 const [error,setError]=useState('');const [loading,setLoading]=useState(false);

 async function load(){setLoading(true);setError('');
  try{const [o,d,g]=await Promise.all([
    api('/control-tower/operations'),
    api('/control-tower/drift').catch(e=>({unavailable:e.message})),
    api('/control-tower/governance')]);
   setOps(o);setDrift(d);setGov(g);}
  catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);

 if(error)return <div className="operations"><InfoBox warning><span role="alert">{error}</span></InfoBox></div>;
 if(!ops)return <div className="operations"><Empty icon={ChartLineUp} title="Reading the store">One moment…</Empty></div>;

 const lat=ops.latency_ms||{};
 const byStage=ops.latency_by_stage||{};
 const worst=Object.entries(byStage).sort((a,b)=>b[1].max-a[1].max)[0];
 const cd=drift?.conversion_drift||{};

 return <div className="operations">
  <div className="page-intro"><h1>Runtime & learning</h1>
   <p>What the engine actually did, whether the world it was fitted on still exists, and what is
    queued to replace what is in production.</p></div>

  <div className="tabs" role="tablist">{TABS.map(([k,label,Icon])=>
    <button key={k} role="tab" aria-selected={k===tab} className={k===tab?'active':''}
      onClick={()=>setTab(k)}><Icon size={15}/> {label}</button>)}</div>
  <div className="actions end"><Button busy={loading} onClick={load}><ArrowClockwise/> Refresh</Button></div>

  {tab==='ops'&&<>
   <div className="model-metrics">
    <div><small>Decisions</small><b>{number(ops.decisions_total)}</b></div>
    <div><small>Overrides</small><b>{number(ops.overrides_total)}</b></div>
    <div><small>Outcomes</small><b>{number(ops.outcomes_total)}</b></div>
    <div><small>Audit entries</small><b>{number(ops.audit_entries)}</b></div>
    <div><small>Latency p50</small><b>{lat.p50==null?'—':`${lat.p50.toFixed(0)} ms`}</b></div>
    <div><small>Latency p95</small><b>{lat.p95==null?'—':`${lat.p95.toFixed(0)} ms`}</b></div>
   </div>

   {Object.keys(byStage).length>1&&<Panel title="Latency by stage" icon={Timer}>
    <p className="small-copy">Pooled across both stages the median is {lat.p50?.toFixed(0)} ms, which is a
     figure no request has actually taken. Pre-assessment scores a band from a loaded model; final
     assessment runs the whole agent pipeline. They are different workloads and are reported separately.</p>
    <DataTable columns={[
      {label:'Stage',render:r=><b>{titleCase(r[0])}</b>},
      {label:'Decisions',render:r=>number(r[1].n)},
      {label:'p50',render:r=>`${r[1].p50.toFixed(0)} ms`},
      {label:'p95',render:r=><Badge tone={r[1].p95>2000?'amber':'green'}>{r[1].p95.toFixed(0)} ms</Badge>},
      {label:'Slowest',render:r=><Badge tone={r[1].max>10000?'red':r[1].max>2000?'amber':'blue'}>{r[1].max.toFixed(0)} ms</Badge>},
     ]} rows={Object.entries(byStage).sort((a,b)=>b[1].p50-a[1].p50)}/>
    {worst&&worst[1].max>10000&&<InfoBox warning>
     The slowest {titleCase(worst[0])} on record took {(worst[1].max/1000).toFixed(1)} s — roughly
     {' '}{Math.round(worst[1].max/worst[1].p50)}× its own median. One request is not a trend, and a mean
     latency would have hidden it entirely. Cold model load on the first request after a restart is the
     obvious candidate; it is worth confirming against the API log rather than assuming.
    </InfoBox>}
   </Panel>}

   <Panel title="Approval routing" icon={Timer}>
    <p className="small-copy">Which authority each recommendation was routed to. The engine recommends
     and routes; it does not approve.</p>
    <DataTable columns={[
      {label:'Authority',render:r=><b>{titleCase(r[0])}</b>},
      {label:'Decisions',render:r=>number(r[1])},
      {label:'Share',render:r=>percent(100*r[1]/Math.max(1,Object.values(ops.approval_mix||{}).reduce((a,b)=>a+b,0)),1)},
     ]} rows={Object.entries(ops.approval_mix||{}).sort((a,b)=>b[1]-a[1])}/>
   </Panel>

   <Panel title="Overrides">
    <div className="model-metrics">
     <div><small>Override rate</small><b>{ops.override_rate_on_recorded_decisions==null?'—':
       percent(100*ops.override_rate_on_recorded_decisions,1)}</b></div>
     <div><small>Outside authority</small><b>{number(ops.overrides_outside_authority)}</b></div>
     <div><small>Policy versions in use</small><b>{(ops.policy_versions_in_use||[]).join(', ')||'—'}</b></div>
    </div>
    <DataTable columns={[
      {label:'Reason',render:r=><b>{titleCase(r[0])}</b>},
      {label:'Count',render:r=>number(r[1])}]}
      rows={Object.entries(ops.override_reason_mix||{}).sort((a,b)=>b[1]-a[1])}
      empty="No override has been recorded."/>
    {ops.overrides_outside_authority>0&&<InfoBox warning>
     {ops.overrides_outside_authority} override(s) were recorded outside the approver&apos;s authority.
     They are stored and flagged rather than blocked — recording an override is not approval, and an
     engine that silently refused one would push the negotiation off-system where nothing can audit it.
    </InfoBox>}
   </Panel>

   <Panel title="Recent decisions">
    <DataTable columns={[
      {label:'Decision',render:d=><span><code>{d.decision_id}</code><br/><small className="muted">{d.loan_id}</small></span>},
      {label:'Stage',render:d=><Badge>{titleCase(d.stage)}</Badge>},
      {label:'Rate',render:d=>d.recommended_rate==null?'—':<span>{percent(d.recommended_rate)}
        {d.card_rate!=null&&<><br/><small className="muted">card {percent(d.card_rate)}</small></>}</span>},
      {label:'Authority',render:d=><small>{titleCase(d.required_approval||'—')}</small>},
      {label:'Took',render:d=>d.elapsed_ms==null?'—':`${d.elapsed_ms.toFixed(0)} ms`},
      {label:'When',render:d=><small>{String(d.created_at).replace('T',' ').slice(0,16)}</small>},
     ]} rows={(ops.recent_decisions||[]).slice(0,25)}/>
    <p className="small-copy">{ops.note}</p>
   </Panel>
  </>}

  {tab==='drift'&&(drift?.unavailable
   ?<InfoBox warning>Drift needs the training dataset, which is not readable here: {drift.unavailable}</InfoBox>
   :drift&&<>
   <div className="model-metrics">
    <div><small>Overall</small><b>{titleCase(drift.overall_status)}</b></div>
    <div><small>Reference window</small><b>{drift.reference_window.from} → {drift.reference_window.to}</b></div>
    <div><small>Current window</small><b>{drift.current_window.from} → {drift.current_window.to}</b></div>
    <div><small>PSI warn / alert</small><b>{drift.thresholds.psi_warn} / {drift.thresholds.psi_alert}</b></div>
   </div>

   <InfoBox>
    Both windows are drawn from the same synthetic generator, so a green board here demonstrates that
    the monitor runs — not that the world is stable. On real data the reference window is the book the
    models were fitted on, and this page is the first thing that would turn amber.
   </InfoBox>

   <Panel title="Input drift (PSI)" icon={Waveform}>
    <DataTable columns={[
      {label:'Feature',render:r=><b>{titleCase(r.feature)}</b>},
      {label:'PSI',render:r=>r.psi?.toFixed(4)},
      {label:'Status',render:r=><Badge tone={tone(r.status)}>{r.status}</Badge>},
     ]} rows={drift.input_drift||[]}/>
   </Panel>

   <Panel title="Conversion drift">
    <p className="small-copy">The one that matters: the same application converting differently than
     it used to, which no input-distribution monitor catches. Bucketed by how far the offer sits from
     the competitor rate.</p>
    <div className="model-metrics">
     <div><small>Status</small><b>{titleCase(cd.status||'unknown')}</b></div>
     <div><small>Largest shift</small><b>{cd.max_shift_pts==null?'—':`${cd.max_shift_pts} pts`}</b></div>
     <div><small>S2D reference</small><b>{cd.overall_s2d_reference==null?'—':percent(100*cd.overall_s2d_reference,1)}</b></div>
     <div><small>S2D current</small><b>{cd.overall_s2d_current==null?'—':percent(100*cd.overall_s2d_current,1)}</b></div>
    </div>
    <DataTable columns={[
      {label:'Competitor gap (bps)',render:r=><b>{r._bucket}</b>},
      {label:'Reference S2D',render:r=>percent(100*r.reference_s2d,1)},
      {label:'Current S2D',render:r=>percent(100*r.current_s2d,1)},
      {label:'Change',render:r=><Badge tone={Math.abs(r.change_pts)>=3?'amber':'blue'}>{r.change_pts>0?'+':''}{r.change_pts} pts</Badge>},
     ]} rows={cd.by_competitor_gap||[]}/>
    <p className="small-copy">{drift.note}</p>
   </Panel>
  </>)}

  {tab==='champions'&&gov&&<>
   <Panel title="In production" icon={Trophy}>
    <DataTable columns={[
      {label:'Model',render:r=><b>{titleCase(r.name)}</b>},
      {label:'Version',render:r=><code>{r.version}</code>},
      {label:'Registered',render:r=><small>{String(r.registered_at).replace('T',' ').slice(0,16)}</small>},
     ]} rows={gov.champions||[]}/>
   </Panel>
   <Panel title="Waiting to be promoted">
    <DataTable columns={[
      {label:'Model',render:r=><b>{titleCase(r.name)}</b>},
      {label:'Version',render:r=><code>{r.version}</code>},
      {label:'Registered',render:r=><small>{String(r.registered_at).replace('T',' ').slice(0,16)}</small>},
     ]} rows={gov.challengers||[]}
     empty="No challenger is registered. AutoML searches produce a leaderboard, not a registered challenger — promoting one is a human decision that nothing here performs."/>
   </Panel>
   <Panel title="Promotion gates">
    <div className="model-metrics">
     <div><small>Min shadow decisions</small><b>{number(gov.promotion_gates.shadow_min_decisions)}</b></div>
     <div><small>Must beat champion by</small><b>{gov.promotion_gates.promote_if_better_by_pct}%</b></div>
     <div><small>Max conversion drop</small><b>{gov.promotion_gates.max_conversion_drop_pct}%</b></div>
     <div><small>Human sign-off</small><b>{gov.promotion_gates.require_human_signoff?'required':'not required'}</b></div>
    </div>
    <p className="small-copy">{gov.note} These gates are declared in <code>model_config.yaml</code> and
     enforced by <code>learning/champion_challenger.py</code>, which no route calls — a promotion today
     happens by re-running training, not by passing this gate.</p>
   </Panel>
  </>}
 </div>;
}
