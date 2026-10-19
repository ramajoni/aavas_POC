import React,{useEffect,useState} from 'react';
import {WarningCircle,ArrowRight} from '@phosphor-icons/react';
import {api,number} from '../api';
import {Panel,Badge,InfoBox,Button} from '../components';

// The lifecycle board. Each stage reports what is actually there and what is
// missing - a readiness view that only ever showed green would be decoration.
const STAGES=[
 {key:'configure', n:1, title:'Configure', blurb:'Business, policy, features, agents, data sources.'},
 {key:'train',     n:2, title:'Train',     blurb:'Dataset, features, AutoML, calibration.'},
 {key:'validate',  n:3, title:'Validate & Approve', blurb:'Explainability, gates, simulation, sign-off.'},
 {key:'run',       n:4, title:'Run',       blurb:'Live agent execution and priced decisions.'},
 {key:'observe',   n:5, title:'Observe & Improve', blurb:'Traces, outcomes, drift, retraining.'},
];

const tone=s=>s==='ready'?'green':s==='partial'?'amber':'red';
const Icon=()=><WarningCircle size={17} weight="fill"/>;

export function Overview({go}){
 const [state,setState]=useState(null);const [error,setError]=useState('');const [loading,setLoading]=useState(false);

 async function load(){setLoading(true);setError('');
  const get=p=>api(p).then(r=>r).catch(()=>null);
  try{
   const [config,tracking,xai,agents,health,cards,traces]=await Promise.all([
     get('/config'),get('/tracking'),get('/xai'),get('/agents'),get('/health'),
     get('/model-cards'),get('/agents/traces')]);
   setState({config,tracking,xai,agents,health,cards,traces});
  }catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);

 if(!state)return <div className="os-page"><Panel title="Reading the system">
   {error?<span role="alert">{error}</span>:'Collecting configuration, models, agents and traces…'}</Panel></div>;

 const {config,tracking,xai,agents,health,cards,traces}=state;
 const models=Object.keys(cards||{}).length;
 const withReports=(xai?.models||[]).length;
 const weak=(xai?.models||[]).filter(m=>m.trust?.level==='weak').length;
 const wired=agents?.summary?.wired??0;
 const dormant=agents?.summary?.dormant??0;
 const degraded=agents?.summary?.degraded??0;
 const missing=(agents?.models_missing||[]).length;

 const rows={
  configure:{
   status:config?'ready':'absent',
   facts:[[`${config?.files?.length??0} files`,'versioned'],
          [number(config?.settings??0),'settings'],
          [`${config?.overrides_active?.length??0}`,'env overrides']],
   gap:'Read-only. A config change has no validation and no approval before it takes effect.',
  },
  train:{
   status:(tracking?.mlflow?.searches??0)>0?'partial':'absent',
   facts:[[number(tracking?.mlflow?.searches??0),'AutoML searches'],
          [number(tracking?.mlflow?.candidates??0),'candidates tried'],
          [number(models),'models trained']],
   gap:'AutoML selects and explains a champion; it is not the training path. models/ still comes from make train.',
  },
  validate:{
   status:withReports&&withReports>=models?'partial':'absent',
   facts:[[`${withReports}/${models}`,'models with an XAI brief'],
          [`${weak}`,'weak rank agreement'],
          [`${(tracking?.mlflow?.searches??0)?'yes':'no'}`,'gate results recorded']],
   gap:'Every piece of evidence a reviewer needs is on screen. There is no sign-off: nothing here can be approved.',
  },
  run:{
   status:health?.status==='ok'&&!missing?'ready':'partial',
   facts:[[`${wired}`,'agents wired'],[`${dormant}`,'dormant'],
          [`${degraded||missing}`,'degraded']],
   gap:dormant?`${dormant} agent(s) are built and called by nothing. Wiring them changes what the engine prices on.`:'',
  },
  observe:{
   status:(traces?.traces||[]).length?'partial':'absent',
   facts:[[number((traces?.traces||[]).length),'traced decisions'],
          [`${tracking?.mlflow?.runs??0}`,'tracked runs'],
          [health?.status==='ok'?'live':'degraded','service']],
   gap:'Drift, retraining and override learning run as scripts, not as a loop that records what it decided.',
  },
 };

 return <div className="os-page">
  <div className="os-page-head">
   <div><h1>Lifecycle</h1>
    <p>Configure → Train → Validate → Approve → Run → Observe → Learn. What is in place at each stage,
     and what is not.</p></div>
   <Button busy={loading} onClick={load}>Refresh</Button>
  </div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  <div className="os-stages">
   {STAGES.map(stage=>{const r=rows[stage.key];
    return <button key={stage.key} className={`os-stage ${r.status}`} onClick={()=>go(stage.key)}>
     <span className="os-stage-head">
      <span className="os-stage-n">{stage.n}</span>
      <b>{stage.title}</b>
      <Badge tone={tone(r.status)}>{r.status}</Badge>
     </span>
     <span className="os-stage-blurb">{stage.blurb}</span>
     <span className="os-stage-facts">
      {r.facts.map(([v,l],i)=><span key={i}><b>{v}</b><small>{l}</small></span>)}
     </span>
     {r.gap&&<span className="os-stage-gap"><Icon/> {r.gap}</span>}
     <span className="os-stage-go">Open <ArrowRight size={14}/></span>
    </button>;})}
  </div>

  <Panel title="Versions in force">
   <div className="model-metrics">
    <div><small>Policy</small><b>{health?.versions?.policy_version||'—'}</b></div>
    <div><small>Model config</small><b>{health?.versions?.model_config_version||'—'}</b></div>
    <div><small>Feature contract</small><b>{config?.files?.find(f=>f.name==='features')?.version||'—'}</b></div>
    <div><small>App</small><b>{health?.versions?.app_version||'—'}</b></div>
    <div><small>Models loaded</small><b>{(agents?.models_loaded||[]).length}</b></div>
    <div><small>Service</small><b>{health?.status||'unknown'}</b></div>
   </div>
   <p className="small-copy">Every decision records these. An audit can therefore say which configuration
    priced a given file, not merely which code did. There is no version over the <em>agent set</em> as a
    whole yet, which is what a published, approved release would need.</p>
  </Panel>
 </div>;
}
