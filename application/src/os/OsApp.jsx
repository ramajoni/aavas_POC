import React,{useEffect,useState} from 'react';
import {Sliders,PlugsConnected,Robot,Flask,Brain,Eye,Gauge,ChartLineUp,ShieldCheck,Graph,Broadcast,Waveform,Stack,ArrowSquareOut,CircleNotch} from '@phosphor-icons/react';
import {api} from '../api';
import {Overview} from './Overview';
import {Config} from '../Config';
import {Connectors} from '../Connectors';
import {Agents} from '../Agents';
import {Mlflow} from '../Mlflow';
import {Xai} from '../Xai';
import {CreditScore} from '../CreditScore';
import {XaiDashboard} from '../XaiDashboard';
import {Operations} from '../Operations';
import {ControlTower} from '../ControlTower';
import {Monitoring} from '../Monitoring';
import {SystemOfRecord} from '../SystemOfRecord';
import {Governance} from '../Governance';
import {Features} from '../Features';

// The console is the control tower over one workload: this pricing engine. The
// tower itself sits above the lifecycle - Configure -> Train -> Validate &
// Approve -> Run -> Observe - and each module owns the pages for its stage.
// Several are the same components the lending app's admin view uses, which is
// deliberate: one implementation, two places to reach it from.
const MODULES=[
 {key:'overview', n:'', label:'Control tower', icon:Broadcast, views:[
   {key:'tower', label:'Estate dashboard', icon:Broadcast},
   {key:'lifecycle', label:'Lifecycle board', icon:Graph},
   {key:'record', label:'System of record', icon:Stack}]},
 {key:'configure', n:1, label:'Configure', icon:Sliders, views:[
   {key:'config', label:'Configuration', icon:Sliders},
   {key:'connectors', label:'Connectors', icon:PlugsConnected}]},
 {key:'train', n:2, label:'Train', icon:Flask, views:[
   {key:'mlflow', label:'AutoML & MLflow', icon:Flask},
   {key:'models', label:'Trained models', icon:Brain},
   {key:'features', label:'Feature intelligence', icon:Flask}]},
 {key:'validate', n:3, label:'Validate & Approve', icon:ShieldCheck, views:[
   {key:'xai', label:'Model explainability', icon:Eye},
   {key:'credit-score', label:'Credit score', icon:Gauge},
   {key:'xai-dash', label:'XAI dashboard', icon:Brain},
   {key:'governance', label:'Approvals & gates', icon:ShieldCheck}]},
 {key:'run', n:4, label:'Run', icon:Robot, views:[
   {key:'agents', label:'Agent control plane', icon:Robot}]},
 {key:'observe', n:5, label:'Observe & Improve', icon:ChartLineUp, views:[
   {key:'runtime', label:'Runtime & drift', icon:Waveform},
   {key:'learning', label:'Continuous learning', icon:ChartLineUp},
   {key:'audit', label:'Decision audit', icon:ShieldCheck}]},
];

const BY_KEY=Object.fromEntries(MODULES.map(m=>[m.key,m]));
const route=()=>{const [m,v]=location.hash.slice(2).split('/');
 return {module:BY_KEY[m]?m:'overview', view:v||null};};

// Operations was written for the lending app and reads a draft application to
// prefill the audit search. The console has no draft, so it supplies the demo
// file rather than reaching into state that does not exist here.
const OPS_STATE={applicant:{loan_id:'DEMO-FLOW-001'}};

export function OsApp(){
 const [{module,view},setRoute]=useState(route);
 const [health,setHealth]=useState(null);const [cards,setCards]=useState(null);

 useEffect(()=>{const on=()=>setRoute(route());window.addEventListener('hashchange',on);
  return()=>window.removeEventListener('hashchange',on);},[]);
 useEffect(()=>{const pull=()=>{api('/health').then(setHealth).catch(()=>setHealth(null));};
  pull();api('/model-cards').then(setCards).catch(()=>{});
  const timer=setInterval(pull,30000);return()=>clearInterval(timer);},[]);

 function go(m,v){location.hash=v?`/${m}/${v}`:`/${m}`;setRoute({module:m,view:v||null});window.scrollTo(0,0);}

 const active=BY_KEY[module];
 const current=view&&active.views.find(x=>x.key===view)?view:active.views[0]?.key;

 function render(){
  switch(current){
   case 'tower':       return <ControlTower navigate={to=>to==='runtime'?go('observe','runtime'):go('run','agents')}/>;
   case 'lifecycle':   return <Overview go={m=>go(m)}/>;
   case 'record':      return <SystemOfRecord/>;
   case 'config':       return <Config/>;
   case 'connectors':   return <Connectors/>;
   case 'mlflow':       return <Mlflow navigate={()=>go('validate','xai-dash')}/>;
   case 'features':     return <Features/>;
   case 'models':       return <Operations page="models" cards={cards} state={OPS_STATE} health={health}/>;
   case 'xai':          return <Xai health={health} navigate={()=>go('validate','xai-dash')}/>;
   case 'credit-score': return <CreditScore navigate={()=>go('validate','xai')}/>;
   case 'xai-dash':     return <XaiDashboard navigate={()=>go('validate','xai')}/>;
   case 'governance':   return <Governance/>;
   case 'agents':       return <Agents navigate={()=>go('observe','audit')}/>;
   case 'runtime':      return <Monitoring/>;
   case 'learning':     return <Operations page="learning" cards={cards} state={OPS_STATE} health={health}/>;
   case 'audit':        return <Operations page="audit" cards={cards} state={OPS_STATE} health={health}/>;
   default:             return <ControlTower navigate={()=>go('observe','runtime')}/>;
  }
 }

 return <div className="os-shell">
  <aside className="os-rail">
   <div className="os-brand">
    <span className="os-mark"><Graph size={20} weight="bold"/></span>
    <span><b>Credit Pricing OS</b><small>Agentic</small></span>
   </div>
   <nav aria-label="Lifecycle modules">
    {MODULES.map(m=><button key={m.key} className={m.key===module?'active':''}
      onClick={()=>go(m.key)} aria-current={m.key===module?'page':undefined}>
      <m.icon size={19} weight={m.key===module?'fill':'regular'}/>
      <span>{m.label}</span>
      {m.n!==''&&<i className="os-rail-n">{m.n}</i>}
    </button>)}
   </nav>
   <div className="os-rail-foot">
    <a href="/" className="os-rail-link">Lending workspace <ArrowSquareOut size={13}/></a>
    <a href="/api/docs" target="_blank" rel="noreferrer" className="os-rail-link">API docs <ArrowSquareOut size={13}/></a>
   </div>
  </aside>

  <div className="os-main">
   <header className="os-top">
    <div className="os-crumbs">
     <button onClick={()=>go('overview')}>Control tower</button>
     {module!=='overview'&&<><span>/</span><b>{active.label}</b></>}
    </div>
    <div className="os-versions">
     {health?.versions&&<>
      <span title="Policy version">policy <code>{health.versions.policy_version}</code></span>
      <span title="Model config version">models <code>{health.versions.model_config_version}</code></span>
     </>}
     <span className={`os-dot ${health?.status==='ok'?'ok':'bad'}`}/>
     <span>{health?health.status:<CircleNotch size={13} className="spin"/>}</span>
    </div>
   </header>

   {active.views.length>1&&<div className="os-subnav">
    {active.views.map(v=><button key={v.key} className={v.key===current?'active':''}
      onClick={()=>go(module,v.key)}><v.icon size={15}/> {v.label}</button>)}
   </div>}

   <main className="os-content">{render()}</main>
  </div>
 </div>;
}
