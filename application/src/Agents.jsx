import React,{useEffect,useState} from 'react';
import {Robot,FlowArrow,ArrowClockwise,WarningCircle,Moon,Info,Wrench,Lightning,Plugs,Package} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable} from './components';

const ms=n=>n==null?'—':n<1?`${n.toFixed(2)} ms`:n<1000?`${n.toFixed(1)} ms`:`${(n/1000).toFixed(2)} s`;
const when=s=>s?new Date(s).toLocaleString('en-IN'):'—';
const statusTone=s=>s==='ready'?'green':s==='degraded'?'amber':'blue';

export function Agents({navigate}){
 const [body,setBody]=useState(null);const [trace,setTrace]=useState(null);const [traces,setTraces]=useState([]);
 const [loading,setLoading]=useState(false);const [error,setError]=useState('');const [note,setNote]=useState('');
 const [tab,setTab]=useState('agents');const [tools,setTools]=useState(null);const [skills,setSkills]=useState(null);
 const [pkgs,setPkgs]=useState(null);

 async function load(){setLoading(true);setError('');
  try{
   const [roster,list,tl,sk,pk]=await Promise.all([
     api('/agents'),api('/agents/traces'),api('/agents/tools'),api('/agents/skills'),
     api('/agents/packages')]);
   setBody(roster);setTraces(list.traces||[]);setNote(list.note||'');setTools(tl);setSkills(sk);
   setPkgs(pk);
   try{setTrace(await api('/agents/trace/latest'));}catch{setTrace(null);}
  }catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);

 const agents=body?.agents||[];
 const wired=agents.filter(a=>a.wired);
 const dormant=agents.filter(a=>!a.wired);
 const steps=trace?.steps||[];
 const slowest=Math.max(1,...steps.map(s=>s.ms||0));

 return <div className="operations">
  <div className="page-intro"><h1>Agent Control Plane</h1>
   <p>Who decides what. The pricing engine is a pipeline of agents over a shared digital twin, not a model
    behind an endpoint — this is the roster, the order they run in, and what each one produced on the last
    decision the engine priced.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  {body&&<>
   <div className="model-metrics">
    <div><small>Agents</small><b>{body.summary.agents}</b></div>
    <div><small>Wired in</small><b>{body.summary.wired}</b></div>
    <div><small>Dormant</small><b>{body.summary.dormant}</b></div>
    <div><small>Degraded</small><b>{body.summary.degraded}</b></div>
    <div><small>Pipeline steps</small><b>{body.summary.pipeline_steps}</b></div>
    <div><small>Tools</small><b>{body.capabilities?.tools??'—'}</b></div>
    <div><small>Skills</small><b>{body.capabilities?.skills??'—'}</b></div>
   </div>

   <div className="tabs" role="tablist">
    {[['agents','Agents',Robot],['packages','Reusable packages',Package],['tools','Tools',Wrench],['skills','Skills',Lightning]].map(([k,label,Icon])=>
      <button key={k} role="tab" aria-selected={k===tab} className={k===tab?'active':''}
        onClick={()=>setTab(k)}><Icon size={15}/> {label}</button>)}
   </div>

   <div className="actions end"><Button busy={loading} onClick={load}><ArrowClockwise/> Refresh</Button></div>

   {body.models_missing.length>0&&<InfoBox warning>
     Missing model artefacts: {body.models_missing.join(', ')}. The agents that need them still run and fall
     back, so a number on screen may not have come from the model you think it did.</InfoBox>}

   {tab==='agents'&&<>
   {/* ------------------------------------------------------------ pipeline */}
   <Panel title="The final-assessment pipeline" icon={FlowArrow}
     action={trace&&<Badge tone={trace.failed?.length?'red':'green'}>
       {trace.failed?.length?`${trace.failed.length} failed`:'all steps ok'}</Badge>}>
    {trace
      ? <>
        <div className="model-metrics">
         <div><small>Decision</small><b>{trace.decision_id}</b></div>
         <div><small>File</small><b>{trace.loan_id}</b></div>
         <div><small>Recommended</small><b>{trace.recommended_rate==null?'—':`${trace.recommended_rate}%`}</b></div>
         <div><small>Total</small><b>{ms(trace.total_ms)}</b></div>
         <div><small>Outside any agent</small><b>{ms(trace.unattributed_ms)}</b></div>
         <div><small>Priced</small><b>{when(trace.created_at)}</b></div>
        </div>
        <ol className="pipeline">
         {steps.map(s=><li key={s.order} className={s.status==='ok'?'':'failed'}>
           <span className="pipeline-order">{s.order}</span>
           <span className="pipeline-body">
            <b>{titleCase(s.agent)}</b> <small className="muted">{s.action}</small>
            <span className="pipeline-summary">{s.error||s.summary}</span>
           </span>
           <span className="pipeline-time">
            <span className="bar" style={{width:`${Math.max(2,(s.ms/slowest)*100)}%`}}/>
            <b>{ms(s.ms)}</b>
           </span>
         </li>)}
        </ol>
        <p className="small-copy">Timings are from this one decision, so they are an illustration of where the
         work goes, not a benchmark. Two agents appear twice because they genuinely run twice: policy sets the
         envelope and then enforces it, and sales context is read once against the card rate and again against
         the price actually recommended.</p>
       </>
      : <Empty icon={Info} title="No decision carries a trace yet">
          {note||'Price a file from the assessment workspace and it will appear here.'}
          {navigate&&<> <button className="link-button" onClick={()=>navigate('application')}>Open the workspace</button></>}
        </Empty>}
   </Panel>

   {/* -------------------------------------------------------------- roster */}
   <Panel title="Agents that run" icon={Robot}>
    <DataTable columns={[
      {label:'Agent',render:a=><b>{titleCase(a.name)}</b>},
      {label:'Decides',render:a=><small>{a.decides}</small>},
      {label:'Runs at',render:a=>a.pipeline_positions.length
        ? a.pipeline_positions.map(p=>`step ${p}`).join(', ')
        : <small className="muted">{a.stages.map(s=>s.replace('_',' ')).join(', ')||'—'}</small>},
      {label:'Models',render:a=>a.models.length?a.models.join(', '):<span className="muted">none</span>},
      {label:'Status',render:a=><Badge tone={statusTone(a.status)}>{a.status}</Badge>},
    ]} rows={wired}/>
    <p className="small-copy">&quot;Degraded&quot; means the agent runs but one of its model artefacts is
     missing, so the engine is falling back. Every agent here writes onto the same digital twin, which is what
     makes this a pipeline rather than six models called in sequence.</p>
   </Panel>

   {/* ------------------------------------------------------------- dormant */}
   {dormant.length>0&&<Panel title="Built, not wired in" icon={Moon}>
    <DataTable columns={[
      {label:'Agent',render:a=><b>{titleCase(a.name)}</b>},
      {label:'Would decide',render:a=><small>{a.decides}</small>},
      {label:'Why it is dormant',render:a=><small>{a.dormant_because}</small>},
      {label:'Module',render:a=><code>{a.module}</code>},
    ]} rows={dormant}/>
    <p className="small-copy">These are implemented and tested. Nothing calls them: wiring them in changes what
     the engine prices on, which is a decision for a human rather than a default.</p>
   </Panel>}

   {/* -------------------------------------------------------- other traces */}
   {traces.length>1&&<Panel title="Recent traced decisions" icon={Info}>
    <DataTable columns={[
      {label:'Decision',render:t=><button className={`link-button${trace?.decision_id===t.decision_id?' active':''}`}
        onClick={()=>api(`/agents/trace/${t.decision_id}`).then(setTrace).catch(e=>setError(e.message))}>
        {t.decision_id}</button>},
      {label:'File',render:t=>t.loan_id},
      {label:'Stage',render:t=>titleCase(t.stage||'')},
      {label:'Steps',render:t=>number(t.steps)},
      {label:'Total',render:t=>ms(t.total_ms)},
      {label:'Priced',render:t=><small>{when(t.created_at)}</small>},
    ]} rows={traces}/>
   </Panel>}
   <p className="small-copy">{body.note}</p>
   </>}

   {/* ------------------------------------------------------------- packages */}
   {tab==='packages'&&pkgs&&<>
    <InfoBox><b>Four of the agents are whole subsystems rather than pipeline steps</b>, and each is
     independently useful. {pkgs.note}</InfoBox>
    <div className="model-metrics">
     <div><small>Packages</small><b>{pkgs.packages.length}</b></div>
     <div><small>Standalone</small><b>{pkgs.standalone.length}</b></div>
     <div><small>Coupled</small><b>{pkgs.coupled.length}</b></div>
     <div><small>Shared kernel</small><b>{pkgs.shared_kernel.length}</b></div>
    </div>
    {pkgs.packages.map(p=><Panel key={p.name} title={p.title} icon={Package}
      action={p.standalone?<Badge tone="green">standalone</Badge>:<Badge tone="amber">coupled</Badge>}>
     <p className="small-copy">{p.purpose}</p>
     <div className="model-metrics">
      <div><small>Package</small><b><code>{p.package}</code></b></div>
      <div><small>Version</small><b>{p.version}</b></div>
      <div><small>Public symbols</small><b>{p.public.length}</b></div>
      <div><small>Depends on</small><b>{p.depends_on.join(', ')}</b></div>
     </div>
     {!p.standalone&&<InfoBox warning>{p.not_standalone_because}</InfoBox>}
     <p className="small-copy"><b>Produces:</b> {p.emits}</p>
     <p className="small-copy"><b>Owns:</b> {p.owns_config.map(c=><code key={c}>{c} </code>)}</p>
     <p className="small-copy"><b>Lifting it out:</b> {p.reuse}</p>
     <p className="chip-row">{p.public.map(x=><span className="chip" key={x}>{x.split(':')[1]}</span>)}</p>
    </Panel>)}
    <Panel title="Why a manifest rather than four new directories">
     <p className="small-copy">Each package already had a folder; what was missing was a boundary.
      Measuring the real import graph found the four were nearly independent already, and that every
      cross-package edge was a single pure function living wherever it was first written. Those are
      kernel primitives now. Moving the directories would have renamed the problem; extracting the
      kernel removed it — and <code>tests/test_agent_boundaries.py</code> checks every claim above
      against the actual imports, in both directions.</p>
    </Panel>
   </>}

   {/* ---------------------------------------------------------------- tools */}
   {tab==='tools'&&tools&&<>
    <InfoBox><b>A tool is something outside this process that can be absent.</b> A model artefact on
     disk, a store, an MCP server, an optional library, an LLM endpoint. Each one names what makes it
     unavailable and what runs instead — a tool with no stated fallback is a tool that takes the engine
     down when it goes missing, which is a decision to take on purpose rather than discover.</InfoBox>

    <div className="model-metrics">
     {Object.entries(tools.summary?.tool_kinds||{}).map(([kind,n])=>
       <div key={kind}><small>{titleCase(kind)}</small><b>{n}</b></div>)}
    </div>

    <Panel title="Tools" icon={Wrench}>
     <DataTable columns={[
       {label:'Tool',render:t=><span><b>{t.name}</b><br/><small className="muted">{t.provides}</small></span>},
       {label:'Kind',render:t=><Badge>{t.kind}</Badge>},
       {label:'Unavailable when',render:t=><small>{t.unavailable_when}</small>},
       {label:'Fallback',render:t=><small>{t.fallback}</small>},
       {label:'Used by',render:t=><small>{(tools.used_by[t.name]||[]).map(titleCase).join(', ')||'—'}</small>},
       {label:'Now',render:t=><Badge tone={t.status==='ready'?'green':t.status==='unavailable'?'red':'blue'}>
         {t.status}</Badge>},
      ]} rows={tools.tools}/>
     <p className="small-copy">{tools.note}</p>
    </Panel>

    <Panel title="Why this is the MCP surface" icon={Plugs}>
     <p className="small-copy">Tools are the set an MCP server would expose and an MCP client would
      consume — they already have a name, a stated failure mode and a fallback. Skills are not exposed:
      there is nothing to call remotely, and a remote call would only add a way for arithmetic to become
      unavailable.</p>
    </Panel>
   </>}

   {/* --------------------------------------------------------------- skills */}
   {tab==='skills'&&skills&&<>
    <InfoBox><b>A skill is deterministic logic with no external dependency</b> — so it has no
     availability and no fallback. Nothing can make one unavailable; a skill failing is a bug, and the
     remedy is a test rather than a health check. Each names the config that governs it, because a skill
     whose thresholds can move is only deterministic with respect to a version of that file.</InfoBox>

    <Panel title="Skills" icon={Lightning}>
     <DataTable columns={[
       {label:'Skill',render:s=><b>{titleCase(s.name)}</b>},
       {label:'Does',render:s=><small>{s.does}</small>},
       {label:'Module',render:s=><code>{s.module}</code>},
       {label:'Governed by',render:s=><small>{s.governed_by}</small>},
       {label:'Used by',render:s=><small>{(skills.used_by[s.name]||[]).map(titleCase).join(', ')||'—'}</small>},
      ]} rows={skills.skills}/>
     <p className="small-copy">{skills.note}</p>
    </Panel>
   </>}
  </>}
 </div>;
}
