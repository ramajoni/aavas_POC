import React,{useCallback,useEffect,useState} from 'react';
import {ArrowSquareOut,WarningCircle} from '@phosphor-icons/react';
import {api} from './api';
import {Panel,Empty,Badge} from './components';

// MLflow and the OmniXAI dashboards are separate servers with their own
// lifecycles. Both pages that show them ask the same question of the API, so
// they ask it the same way and read the same answer.
export function useServices(){
 const [services,setServices]=useState(null);
 const [error,setError]=useState('');const [loading,setLoading]=useState(false);
 const refresh=useCallback(()=>{setLoading(true);
  return api('/tracking/services')
   .then(r=>{setServices(r.services||[]);setError('');})
   .catch(e=>setError(e.message))
   .finally(()=>setLoading(false));},[]);
 useEffect(()=>{refresh();},[refresh]);
 return {services,error,loading,refresh};
}

export const StatusBadge=({up})=><Badge tone={up?'green':'red'}>{up?'running':'not running'}</Badge>;

// A cross-origin frame: this page can open it, never read it. When the server
// is down the panel says how to start it rather than showing an empty box.
export function Embed({service,icon,title,children}){
 if(!service)return null;
 return <Panel title={title||service.name} icon={icon}
   action={service.reachable
     ? <a className="button" href={service.url} target="_blank" rel="noreferrer">Open in a tab <ArrowSquareOut size={15}/></a>
     : <StatusBadge up={false}/>}>
  {service.reachable
    ? <><iframe className="embed-frame" src={service.url} title={service.name} loading="lazy"/>
        {children}
        <p className="small-copy">Embedded from <code>{service.url}</code>. The frame is loaded by your browser,
         so it has to be reachable from where you are sitting — the status is a probe from the API host.</p></>
    : <Empty icon={WarningCircle} title={`${service.name} is not running`}>
        Start it with <code>{service.start}</code>, then refresh this page.</Empty>}
 </Panel>;
}
