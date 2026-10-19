import React,{useEffect,useState} from 'react';
import {Broadcast,ArrowClockwise,WarningCircle,Question,ArrowRight} from '@phosphor-icons/react';
import {api,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty} from './components';

// The estate at a glance. The only rule this page enforces is the one the API
// enforces: a tile shows a number when something produced it, and says why not
// when nothing did. Rendering an unsourced tile as a dash would hide the finding;
// it is shown with its reason, in the same grid, at the same size.
const tone=s=>s==='ready'?'green':s==='partial'?'amber':s==='absent'?'red':
  s==='ok'?'green':s==='warn'?'amber':'blue';

function Metric({m}){
 if(!m.sourced)return <div className="ct-metric unsourced">
  <small>{m.label}</small><b><Question size={18} weight="bold"/> no source</b>
  <span className="ct-why">{m.detail}</span></div>;
 return <div className={`ct-metric ${m.status||''}`}>
  <small>{m.label}</small>
  <b>{m.value===null||m.value===undefined?'—':typeof m.value==='number'?number(m.value):m.value}
   {m.unit&&<i> {m.unit}</i>}</b>
  {m.detail&&<span className="ct-why">{m.detail}</span>}
  <span className="ct-src">{m.source}</span></div>;
}

export function ControlTower({navigate}){
 const [body,setBody]=useState(null);const [error,setError]=useState('');const [loading,setLoading]=useState(false);

 async function load(){setLoading(true);setError('');
  try{setBody(await api('/control-tower'));}catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);

 if(error&&!body)return <div className="operations"><InfoBox warning><span role="alert">{error}</span></InfoBox></div>;
 if(!body)return <div className="operations"><Empty icon={Broadcast} title="Reading the estate">
   Collecting agents, decisions, models and drift…</Empty></div>;

 return <div className="operations ct">
  <div className="page-intro"><h1>AI Control Tower</h1>
   <p>Configuration, approval, runtime and learning for every agent that prices a file. One estate,
    six control zones, and — deliberately — the tiles that nothing in this system can fill.</p></div>

  <div className="actions end"><Button busy={loading} onClick={load}><ArrowClockwise/> Refresh</Button></div>

  <div className="model-metrics">
   <div><small>Metrics on this page</small><b>{body.counts.metrics}</b></div>
   <div><small>With no source</small><b>{body.counts.unsourced}</b></div>
   <div><small>Champions</small><b>{body.champions.length}</b></div>
   <div><small>Challengers</small><b>{body.challengers.length}</b></div>
   <div><small>Policy</small><b>{body.versions?.policy_version}</b></div>
   <div><small>As at</small><b>{String(body.generated_at).replace('T',' ').slice(0,16)}</b></div>
  </div>

  <Panel title="Control zones">
   <div className="ct-zones">
    {body.zones.map(z=><div key={z.key} className={`ct-zone ${z.status}`}>
     <div className="ct-zone-head"><span className="ct-zone-n">{z.n}</span>
      <b>{z.title}</b><Badge tone={tone(z.status)}>{z.status}</Badge></div>
     <p className="small-copy">{z.controls}</p>
     {z.surfaces.length>0&&<p className="ct-surfaces">{z.surfaces.map(s=><code key={s}>{s}</code>)}</p>}
     <p className="ct-gap"><WarningCircle size={15} weight="fill"/> {z.gap}</p>
    </div>)}
   </div>
  </Panel>

  {body.panels.map(p=><Panel key={p.key} title={p.title}>
   <div className="ct-grid">{p.metrics.map(m=><Metric key={m.label} m={m}/>)}</div>
  </Panel>)}

  <Panel title={`What this estate cannot answer (${body.unsourced.length})`} icon={Question}>
   <p className="small-copy">Each of these is a tile a control-tower mock-up would fill with a
    plausible figure. None of them has anything behind it here, so each is returned by the API as a
    null with its reason attached rather than omitted — the absence is the finding.</p>
   <ul className="limitation-list">
    {body.unsourced.map((u,i)=><li key={i}><b>{u.panel} — {u.label}.</b> {u.reason}</li>)}
   </ul>
  </Panel>

  <Panel title="Where to go next">
   <div className="ct-links">
    <button onClick={()=>navigate?.('runtime')}>Runtime operations, drift and champions <ArrowRight size={15}/></button>
    <button onClick={()=>navigate?.('agents')}>Agent control plane and traces <ArrowRight size={15}/></button>
   </div>
   <p className="small-copy">{body.note}</p>
  </Panel>
 </div>;
}
