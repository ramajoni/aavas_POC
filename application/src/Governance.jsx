import React,{useEffect,useState} from 'react';
import {ShieldCheck,ArrowClockwise,Warning,SealCheck,Prohibit,Gavel,PlayCircle,Clock} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable,ApiKeyGate} from './components';

// Sign-off over the system itself, as distinct from authority over one price.
// The page leads with the authentication state because every signature below it
// means something different depending on that one fact.
const stateTone=s=>s==='approved'?'green':s==='applied'?'green':s==='rejected'?'red':
  s==='withdrawn'?'blue':'amber';
const gateTone=p=>p===true?'green':p===false?'red':'amber';
const TABS=[['approvals','Change requests',Gavel],['monitors','Monitors',Clock],
            ['registries','Registries',SealCheck],['roles','Roles & gates',ShieldCheck]];

export function Governance(){
 const [tab,setTab]=useState('approvals');
 const [body,setBody]=useState(null);const [list,setList]=useState(null);const [mon,setMon]=useState(null);
 const [reg,setReg]=useState(null);const [roles,setRoles]=useState(null);
 const [error,setError]=useState('');const [loading,setLoading]=useState(false);
 const [needsKey,setNeedsKey]=useState(false);

 async function load(){setLoading(true);setError('');
  try{
   const [g,a,m,r,ro]=await Promise.all([
     api('/governance'),api('/governance/approvals'),api('/governance/monitors'),
     api('/governance/registries'),api('/governance/roles')]);
   setBody(g);setList(a);setMon(m);setReg(r);setRoles(ro);
  }catch(e){if(e.authRequired)setNeedsKey(true);else setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);

 if(needsKey)return <ApiKeyGate message="The governance and feature surfaces need a role." onSaved={()=>{setNeedsKey(false);load();}}/>;
 if(error&&!body)return <div className="operations"><InfoBox warning><span role="alert">{error}</span></InfoBox></div>;
 if(!body)return <div className="operations"><Empty icon={ShieldCheck} title="Reading governance">One moment…</Empty></div>;

 const auth=body.auth;const s=body.summary;
 return <div className="operations">
  <div className="page-intro"><h1>Governance</h1>
   <p>Sign-off over the system itself — configuration, models, the agent set — as distinct from the
    authority that signs a single price. Gates run when a request is raised, not when it is approved.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  {/* The warning already opens by naming the mode, so the label does not repeat it. */}
  <InfoBox warning={!auth.enforced} success={auth.enforced}>{auth.warning}</InfoBox>

  <div className="model-metrics">
   <div><small>Requests</small><b>{number(s.total)}</b></div>
   <div><small>Awaiting signature</small><b>{s.pending.length}</b></div>
   <div><small>Over a failed gate</small><b>{s.blocked_by_gates.length}</b></div>
   <div><small>Roles</small><b>{Object.keys(s.roles).length}</b></div>
   <div><small>Config version</small><b>{s.version}</b></div>
   <div><small>You are</small><b>{body.principal.name}</b></div>
  </div>

  <div className="tabs" role="tablist">{TABS.map(([k,label,Icon])=>
    <button key={k} role="tab" aria-selected={k===tab} className={k===tab?'active':''}
      onClick={()=>setTab(k)}><Icon size={15}/> {label}</button>)}</div>
  <div className="actions end"><Button busy={loading} onClick={load}><ArrowClockwise/> Refresh</Button></div>

  {tab==='approvals'&&<Panel title="Change requests" icon={Gavel}>
   <p className="small-copy">Approving a request changes nothing on its own — a separate apply step
    does that, and re-checks the authorisation. &quot;Approved but not released&quot; is the state a
    change process actually lives in.</p>
   <DataTable columns={[
     {label:'Request',render:a=><span><b>{a.approval_id}</b><br/>
       <small className="muted">{a.action} {a.artefact_type} · {a.artefact_id}</small></span>},
     {label:'State',render:a=><Badge tone={stateTone(a.state)}>{a.state}</Badge>},
     {label:'Gates',render:a=>a.gates_passed?<Badge tone="green">clear</Badge>
       :<Badge tone="red">failed</Badge>},
     {label:'Signatures',render:a=>a.signatures.length
       ?<small>{a.signatures.map(x=>`${x.role}: ${x.principal} (${x.verdict})`).join(' · ')}</small>
       :<small className="muted">awaiting {a.awaiting_roles.join(', ')||'—'}</small>},
     {label:'Raised by',render:a=><small>{a.requested_by}<br/>{String(a.requested_at).slice(0,16).replace('T',' ')}</small>},
    ]} rows={list?.approvals||[]} empty="No change has been requested yet."/>

   {(list?.approvals||[]).filter(a=>!a.gates_passed).map(a=>
     <InfoBox key={a.approval_id} warning><Prohibit size={17}/>
      <b>{a.approval_id} was raised over a failed gate.</b>{' '}
      {a.gate_results.filter(g=>g.passed===false).map(g=>g.detail).join('; ')}.{' '}
      <em>Justification recorded:</em> “{a.override_justification}”</InfoBox>)}
  </Panel>}

  {tab==='monitors'&&mon&&<>
   <Panel title="Monitors" icon={Clock}>
    <p className="small-copy">{mon.note}</p>
    <DataTable columns={[
      {label:'Monitor',render:m=><span><b>{titleCase(m[0])}</b><br/><small className="muted">{m[1].description}</small></span>},
      {label:'Last run',render:m=>m[1].never_run
        ?<Badge tone="amber">never run</Badge>
        :<small>{String(m[1].last_run).replace('T',' ').slice(0,16)}</small>},
      {label:'Status',render:m=>m[1].last_status
        ?<Badge tone={m[1].last_status==='ok'?'green':m[1].last_status==='alert'?'red':'amber'}>
          {m[1].last_status}</Badge>:<span className="muted">—</span>},
      {label:'Triggered',render:m=>m[1].triggered?<Badge tone="amber">yes</Badge>:'no'},
     ]} rows={Object.entries(mon.monitors)}/>
   </Panel>
   <Panel title="Run history" icon={PlayCircle}>
    <DataTable columns={[
      {label:'Run',render:r=><b>{r.run_id}</b>},
      {label:'Monitor',render:r=>titleCase(r.monitor)},
      {label:'Status',render:r=><Badge tone={r.status==='ok'?'green':r.status==='failed'?'red':'amber'}>{r.status}</Badge>},
      {label:'Finding',render:r=><small>{JSON.stringify(r.findings).slice(0,140)}</small>},
      {label:'When',render:r=><small>{String(r.started_at).replace('T',' ').slice(0,16)}</small>},
     ]} rows={mon.history||[]} empty="No monitor has run yet."/>
   </Panel>
  </>}

  {tab==='registries'&&reg&&<>
   <Panel title="Prompts" icon={SealCheck}>
    <p className="small-copy">Content-hashed, so an edited prompt gets a new id whether or not anyone
     bumped one. Nothing here is on the priced path today — which is why it exists now.</p>
    <DataTable columns={[
      {label:'Prompt',render:p=><b>{p.prompt_id}</b>},
      {label:'Role',render:p=><Badge>{p.role}</Badge>},
      {label:'Used by',render:p=><small>{p.used_by||'—'}</small>},
      {label:'Size',render:p=>`${number(p.chars)} chars`},
     ]} rows={reg.prompts.registered}/>
   </Panel>
   <Panel title="Knowledge indexes">
    <DataTable columns={[
      {label:'Index',render:i=><b>{i.index_id}</b>},
      {label:'Corpus',render:i=><small>{i.corpus||'—'}</small>},
      {label:'Documents',render:i=>number(i.documents)},
      {label:'Backend',render:i=><small>{i.backend}</small>},
      {label:'Built',render:i=><small>{String(i.built_at).replace('T',' ').slice(0,16)}</small>},
     ]} rows={reg.knowledge.indexes} empty="No index has been registered."/>
   </Panel>
   <Panel title="Federated rounds">
    <DataTable columns={[
      {label:'Run',render:f=><span><b>{f.run_id}</b><br/><small className="muted">{f.global_version}</small></span>},
      {label:'Strategy',render:f=><Badge>{f.strategy}{f.proximal_mu?` μ=${f.proximal_mu}`:''}</Badge>},
      {label:'Participants',render:f=>number(f.participants)},
      {label:'Rounds',render:f=>number(f.rounds)},
      {label:'Final MAE',render:f=>f.final_mae_bps==null?'—':`${f.final_mae_bps} bps`},
     ]} rows={reg.federated.rounds}
     empty="No federated round has been recorded. The implementation exists; run one and it appears here."/>
   </Panel>
   <Panel title="Agent sets">
    <DataTable columns={[
      {label:'Manifest',render:m=><span><b>{m.manifest_id}</b>{m.is_current&&<> <Badge tone="green">current</Badge></>}</span>},
      {label:'Agents',render:m=>`${m.wired} wired / ${m.agents}`},
      {label:'Skills',render:m=>number(m.skills)},
      {label:'Tools',render:m=>number(m.tools)},
     ]} rows={reg.agent_sets.manifests}/>
   </Panel>
  </>}

  {tab==='roles'&&roles&&<>
   <Panel title="Who signs what" icon={ShieldCheck}>
    <DataTable columns={[
      {label:'Role',render:r=><b>{r[1].label}</b>},
      {label:'Key',render:r=><code>{r[0]}</code>},
      {label:'Approves',render:r=><small>{r[1].approves}</small>},
     ]} rows={Object.entries(roles.roles)}/>
   </Panel>
   <Panel title="What each change needs">
    <DataTable columns={[
      {label:'Change',render:a=><b>{a[1].label}</b>},
      {label:'Requires',render:a=><small>{(a[1].requires||[]).join(' AND ')}</small>},
      {label:'Gates',render:a=><small>{(a[1].gates||[]).map(g=><code key={g}>{g} </code>)}</small>},
     ]} rows={Object.entries(roles.artefacts)}/>
    <p className="small-copy"><b>Every</b> listed role must sign — a single signature on a model
     promotion is not a control. A failed gate can be overridden, but only by{' '}
     <code>{roles.gate_override?.requires_role}</code>, only with a written justification of at least{' '}
     {roles.gate_override?.min_justification_chars} characters, and the override is stored against the
     approval permanently. The requester may not sign their own request.</p>
   </Panel>
  </>}
 </div>;
}
