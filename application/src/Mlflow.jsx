import React,{useEffect,useState} from 'react';
import {Flask,Database,ArrowClockwise,ArrowSquareOut} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable} from './components';
import {useCanOpen} from './access';
import {Embed,StatusBadge,useServices} from './dashboards';

const num=(n,d=4)=>n==null?'—':Number(n).toFixed(d);
const secs=n=>n==null?'—':n<60?`${n.toFixed(1)}s`:`${(n/60).toFixed(1)}m`;
const when=s=>s?new Date(s).toLocaleString('en-IN'):'—';
// The metrics a search writes depend on the contract it searched, so the table
// reads the first name that is present rather than assuming one schema.
const pick=(metrics,names)=>{for(const n of names){if(metrics?.[n]!=null)return metrics[n];}return null;};

export function Mlflow({navigate:go}){
 const can=useCanOpen();const navigate=go&&can('xai-dash')?go:null;
 const [index,setIndex]=useState(null);const [runs,setRuns]=useState(null);
 const [open,setOpen]=useState(null);const [tab,setTab]=useState('searches');
 const [loading,setLoading]=useState(false);const [error,setError]=useState('');
 const {services,refresh:refreshServices}=useServices();

 async function load(){setLoading(true);setError('');
  try{const [i,r]=await Promise.all([api('/tracking'),api('/tracking/runs')]);setIndex(i);setRuns(r);}
  catch(e){setError(e.message);}finally{setLoading(false);refreshServices();}}
 useEffect(()=>{load();},[]);

 const mlflow=(services||[]).find(s=>s.kind==='mlflow');
 const store=index?.mlflow;

 return <div className="operations">
  <div className="page-intro"><h1>MLflow Tracking</h1>
   <p>Every AutoML search that has been run, and the MLflow server that holds them. The history below is
    read straight off the run store, so it is here whether or not the MLflow server is up.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  {store&&<>
   <div className="model-metrics">
    <div><small>Searches</small><b>{number(store.searches)}</b></div>
    <div><small>Candidates</small><b>{number(store.candidates)}</b></div>
    <div><small>Runs in total</small><b>{number(store.runs)}</b></div>
    <div><small>Last run</small><b>{when(store.last_run_at)}</b></div>
    <div><small>Store</small><b>{store.store_kind}</b></div>
    <div><small>Writes</small><b>{store.writes_enabled?'enabled':'off'}</b></div>
   </div>

   <div className="actions end">
    {mlflow&&<span className="server-pill">MLflow <code>{mlflow.url}</code> <StatusBadge up={mlflow.reachable}/></span>}
    <Button busy={loading} onClick={load}><ArrowClockwise/> Refresh</Button>
   </div>

   <div className="tabs" role="tablist">
    {[['searches','AutoML searches'],['server','MLflow UI']].map(([k,label])=>
     <button key={k} role="tab" aria-selected={tab===k} className={tab===k?'active':''} onClick={()=>setTab(k)}>{label}</button>)}
   </div>

   {/* -------------------------------------------------------------- searches */}
   {tab==='searches'&&(runs?.available
     ? <>
       <InfoBox>One panel per search. Each carries its candidates — the models that were tried and not
        promoted, which is the half of the record a champion artefact cannot show. Nothing here is promoted:
        the artefacts under <code>models/</code> still come from <code>make train</code>.</InfoBox>
       {runs.searches.map(s=><Panel key={s.run_id} title={s.name} icon={Database}
         action={<span><Badge tone={s.status==='finished'?'green':s.status==='failed'?'red':'amber'}>{s.status}</Badge>
          {' '}<Badge>{s.child_count} candidates</Badge></span>}>
        <div className="model-metrics">
         <div><small>Started</small><b>{when(s.started_at)}</b></div>
         <div><small>Duration</small><b>{secs(s.duration_sec)}</b></div>
         <div><small>Champion AUC</small><b>{num(pick(s.metrics,['champion_auc','auc']))}</b></div>
         <div><small>Champion ECE</small><b>{num(pick(s.metrics,['champion_ece','ece']))}</b></div>
         <div><small>Experiment</small><b>{s.experiment}</b></div>
        </div>
        <Button onClick={()=>setOpen(open===s.run_id?null:s.run_id)}>
         {open===s.run_id?'Hide candidates':`Show ${s.child_count} candidates`}</Button>
        {open===s.run_id&&<><DataTable columns={[
          {label:'Candidate',render:r=><b>{r.name.replace(/^candidate::/,'')}</b>},
          {label:'Family',render:r=>titleCase(r.tags?.family||'—')},
          {label:'Monotone',render:r=>r.tags?.monotone==='True'?<Badge tone="green">yes</Badge>:<Badge>no</Badge>},
          {label:'Fairness gate',render:r=>r.tags?.fairness_passed==null?'—'
            :r.tags.fairness_passed==='True'?<Badge tone="green">passed</Badge>:<Badge tone="red">failed</Badge>},
          {label:'AUC',render:r=>num(pick(r.metrics,['auc']))},
          {label:'KS',render:r=>num(pick(r.metrics,['ks']))},
          {label:'ECE (out-of-time)',render:r=>num(pick(r.metrics,['ece']))},
          {label:'PSI',render:r=>num(pick(r.metrics,['psi']))},
          {label:'Time',render:r=>secs(r.duration_sec)},
        ]} rows={[...s.children].sort((a,b)=>(pick(b.metrics,['auc'])??-1)-(pick(a.metrics,['auc'])??-1))}
          empty="This search logged no candidates."/>
        <p className="small-copy">Sorted by AUC, which is not how a champion is chosen — selection is
         multi-objective and gated on fairness and stability. <b>ECE here is the out-of-time <code>ece</code></b>:
         the run also logs <code>calib_ece_after</code>, which is measured on the rows the calibrator was fitted
         on and is therefore ~0 for every candidate. Do not read that one as generalisation.</p></>}
       </Panel>)}
      </>
     : <Empty icon={Database} title="No search history">{runs?.reason||'Loading…'}</Empty>)}

   {/* ------------------------------------------------------------ the server */}
   {tab==='server'&&<Embed service={mlflow} icon={Flask}/>}

   <p className="small-copy">Tracking store <code>{store.store||store.tracking_uri}</code> · experiment
    {' '}<code>{store.configured_experiment}</code> · {index.note}
    {navigate&&<> · the OmniXAI dashboards are on
     {' '}<button className="link-button" onClick={()=>navigate('xai-dash')}>XAI Dashboard</button>.</>}</p>
  </>}
 </div>;
}
