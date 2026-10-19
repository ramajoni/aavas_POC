import React,{useEffect,useState} from 'react';
import {Stack,ArrowClockwise,Link as LinkIcon,LinkBreak,Database,MagnifyingGlass,Fingerprint} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable,Field} from './components';

// Two halves. The registries say what exists; the chain tests the claim that
// rests on them. A system of record that is described rather than exercised is
// a diagram, so the walker runs against a real decision and reports what breaks.
const kindTone=k=>k==='registry'?'green':k==='versioned config'?'blue':
  k==='per-decision data'?'blue':k==='absent'?'red':'amber';
const linkTone=s=>s==='resolved'?'green':s==='partial'?'amber':'red';

const show=v=>v==null?'—':typeof v==='object'?JSON.stringify(v,null,1).replace(/[{}"]/g,'').trim():String(v);

export function SystemOfRecord(){
 const [body,setBody]=useState(null);const [chain,setChain]=useState(null);const [sets,setSets]=useState(null);
 const [id,setId]=useState('');const [traces,setTraces]=useState([]);
 const [error,setError]=useState('');const [loading,setLoading]=useState(false);

 async function load(){setLoading(true);setError('');
  try{
   const [sor,list,ms]=await Promise.all([
     api('/system-of-record'),api('/agents/traces'),api('/system-of-record/manifests')]);
   setBody(sor);setTraces(list.traces||[]);setSets(ms);
   try{const c=await api('/system-of-record/reconstruct/latest');setChain(c);setId(c.decision_id);}
   catch{setChain(null);}
  }catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);

 async function walk(decisionId){setError('');
  try{const c=await api(`/system-of-record/reconstruct/${encodeURIComponent(decisionId)}`);
   setChain(c);setId(c.decision_id);}
  catch(e){setError(e.message);}}

 if(error&&!body)return <div className="operations"><InfoBox warning><span role="alert">{error}</span></InfoBox></div>;
 if(!body)return <div className="operations"><Empty icon={Stack} title="Reading the registries">One moment…</Empty></div>;

 const s=body.summary;
 return <div className="operations">
  <div className="page-intro"><h1>Pricing System of Record</h1>
   <p>Twelve registries, and the claim that rests on them: that a priced decision can be rebuilt
    exactly — which agent, which version, which skill, which tool, which model, which features, which
    data, which policy, which personalization, which human decision, which final rate.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  <div className="model-metrics">
   <div><small>Registries</small><b>{s.declared}</b></div>
   <div><small>True registries</small><b>{s.by_kind['registry']||0}</b></div>
   <div><small>Versioned config</small><b>{s.by_kind['versioned config']||0}</b></div>
   <div><small>Code declarations</small><b>{s.by_kind['code declaration']||0}</b></div>
   <div><small>Absent</small><b>{s.absent.length}</b></div>
   <div><small>Queryable as of</small><b>{s.queryable_as_of.length}</b></div>
  </div>

  <div className="actions end"><Button busy={loading} onClick={load}><ArrowClockwise/> Refresh</Button></div>

  <InfoBox><b>A registry is queryable as of a past moment.</b> The model registry stores the full card
   against (name, version), so a decision from March resolves to the card that was live in March.
   Versioned config reconstructs only while that version of the file survives. A code declaration —
   agents, skills, tools — describes the system as it is <em>now</em> and cannot say what it was.
   Only {s.queryable_as_of.length} of {s.declared} can answer a question about the past.</InfoBox>

  <Panel title="The twelve registries" icon={Database}>
   <DataTable columns={[
     {label:'Registry',render:r=><b>{r.title}</b>},
     {label:'Kind',render:r=><Badge tone={kindTone(r.kind)}>{r.kind}</Badge>},
     {label:'Backed by',render:r=>r.backing?<code>{r.backing}</code>:<span className="muted">nothing</span>},
     {label:'Entries',render:r=>r.entries?number(r.entries):<span className="muted">0</span>},
     {label:'As of a past moment',render:r=>r.queryable_as_of
       ?<Badge tone="green">yes</Badge>:<Badge tone="amber">no</Badge>},
     {label:'What that means',render:r=><small>{r.note}</small>},
    ]} rows={body.registries}/>
   <p className="small-copy">{body.note}</p>
  </Panel>

  {sets&&<Panel title="Registered agent sets" icon={Fingerprint}>
   <p className="small-copy">The agent, skill and tool declarations are snapshotted into the store and
    the snapshot id is stamped on every decision. The id is a <b>content hash</b> over the declaration,
    not a number someone maintains — a hand-bumped version is a promise that has to be kept on every
    edit, and the edit where it is forgotten is the one that matters.</p>
   <DataTable columns={[
     {label:'Manifest',render:m=><span><b>{m.manifest_id}</b>{m.is_current&&<> <Badge tone="green">current</Badge></>}
       <br/><small className="muted">sha256 {String(m.content_hash).slice(0,16)}…</small></span>},
     {label:'Agents',render:m=>`${m.wired} wired / ${m.agents}`},
     {label:'Skills',render:m=>number(m.skills)},
     {label:'Tools',render:m=>number(m.tools)},
     {label:'Registered',render:m=><small>{String(m.created_at).replace('T',' ').slice(0,16)}</small>},
    ]} rows={sets.manifests}/>
   <p className="small-copy">{sets.note}</p>
  </Panel>}

  <Panel title="Can a decision actually be rebuilt?" icon={LinkIcon}
    action={chain&&<Badge tone={chain.summary.reconstructible?'green':'amber'}>
      {chain.summary.resolved}/{chain.summary.links} links resolved</Badge>}>
   <div className="actions">
    <Field label="Decision id"><input value={id} onChange={e=>setId(e.target.value)}
      placeholder="DEC-…" aria-label="Decision id"/></Field>
    <Button onClick={()=>walk(id)}><MagnifyingGlass/> Walk the chain</Button>
   </div>
   {traces.length>0&&<p className="chip-row">
    {traces.slice(0,6).map(t=><button key={t.decision_id}
      className={`chip${t.decision_id===chain?.decision_id?' active':''}`}
      onClick={()=>walk(t.decision_id)}>{t.decision_id}</button>)}
   </p>}

   {chain?<>
    <div className="model-metrics">
     <div><small>Resolved</small><b>{chain.summary.resolved}</b></div>
     <div><small>Partial</small><b>{chain.summary.partial}</b></div>
     <div><small>Broken</small><b>{chain.summary.broken}</b></div>
     <div><small>File</small><b>{chain.loan_id}</b></div>
    </div>

    <InfoBox warning={!chain.summary.reconstructible} success={chain.summary.reconstructible}>
     {chain.verdict}</InfoBox>

    <ol className="sor-chain">
     {chain.chain.map(l=><li key={l.step} className={l.status}>
      <span className="sor-head">
       {l.status==='broken'?<LinkBreak size={16} weight="fill"/>:<LinkIcon size={16}/>}
       <b>{l.question}</b><Badge tone={linkTone(l.status)}>{l.status}</Badge>
      </span>
      {l.value!=null&&<pre className="sor-value">{show(l.value)}</pre>}
      {l.why&&<span className="sor-why">{l.why}</span>}
     </li>)}
    </ol>
   </>:<Empty icon={LinkIcon} title="No traced decision yet">
     Price a file and the chain will run against it.</Empty>}
  </Panel>
 </div>;
}
