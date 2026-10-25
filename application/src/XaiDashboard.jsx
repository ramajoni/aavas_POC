import React,{useEffect,useState} from 'react';
import {Brain,ArrowClockwise,Eye} from '@phosphor-icons/react';
import {api,titleCase} from './api';
import {Button,Panel,InfoBox,Empty,DataTable} from './components';
import {useCanOpen} from './access';
import {Embed,StatusBadge,useServices} from './dashboards';

export function XaiDashboard({navigate:go}){
 const can=useCanOpen();const navigate=go&&can('xai')?go:null;
 const {services,error,loading,refresh}=useServices();
 const [reports,setReports]=useState(null);
 const [picked,setPicked]=useState(null);

 useEffect(()=>{api('/xai').then(setReports).catch(()=>{});},[]);

 const dashboards=(services||[]).filter(s=>s.kind==='omnixai');
 const running=dashboards.filter(s=>s.reachable);
 // Default to something that is actually up; fall back to the first listed.
 const current=dashboards.find(s=>s.url===picked)||running[0]||dashboards[0];
 // Which models have an offline brief but no dashboard process right now.
 const analysed=(reports?.models||[]).map(m=>m.model);
 const served=new Set(dashboards.map(s=>s.model).filter(Boolean));

 return <div className="operations">
  <div className="page-intro"><h1>XAI Dashboard</h1>
   <p>The interactive OmniXAI dashboard for a trained model — dataset, prediction, local, global and
    what-if analysis. One server process per model, started outside this application.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  <div className="actions end">
   <span className="server-pill">{running.length} of {dashboards.length||'—'} running</span>
   <Button busy={loading} onClick={refresh}><ArrowClockwise/> Refresh</Button>
  </div>

  {dashboards.length>0&&<Panel title="Dashboard processes" icon={Brain}>
   <DataTable columns={[
     {label:'Model',render:s=>s.model
       ? <button className={`link-button${current&&s.url===current.url?' active':''}`}
           onClick={()=>setPicked(s.url)}>{titleCase(s.model)}</button>
       : <span className="muted">unnamed (port {s.url.split(':').pop()})</span>},
     {label:'Address',render:s=><code>{s.url}</code>},
     {label:'Status',render:s=><StatusBadge up={s.reachable}/>},
   ]} rows={dashboards}/>
   <p className="small-copy">Status is a TCP probe <b>from the API host</b>, not from this browser — the two
    differ the moment the API runs in a container. <code>make xai</code> serves the first three models by
    default, roughly 1.5 GB per process; <code>make xai ARGS="--ui-models risk stage2"</code> picks which.</p>
  </Panel>}

  {current
    ? <Embed service={current} icon={Brain}
        title={current.model?`OmniXAI — ${titleCase(current.model)}`:current.name}/>
    : <Empty icon={Brain} title="No OmniXAI dashboard is running">
        Start one with <code>make xai</code>. Every model still has its offline brief — global importance,
        the credit-score band, per-instance SHAP against LIME — under Model Explainability.</Empty>}

  {analysed.length>0&&<InfoBox>
   <b>{analysed.filter(m=>!served.has(m)).length||'No'} model(s) have an offline report but no dashboard
    process</b>{analysed.filter(m=>!served.has(m)).length>0&&<>: {analysed.filter(m=>!served.has(m)).map(titleCase).join(', ')}</>}.
   {navigate&&<> Their briefs and dashboard captures are on
    {' '}<button className="link-button" onClick={()=>navigate('xai')}>Model Explainability</button>, which needs
    no running process.</>}
  </InfoBox>}
 </div>;
}
