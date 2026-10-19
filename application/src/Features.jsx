import React,{useEffect,useState} from 'react';
import {Flask,ArrowClockwise,Sparkle,Prohibit,Graph,MagnifyingGlass,Warning} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable,ApiKeyGate} from './components';

// Feature discovery, and the four refusals. The page leads with what the agent
// cannot do, because that is the design rather than a caveat on it.
const decisionTone=d=>d==='keep'?'green':d==='review'?'amber':'red';
const TABS=[['discover','Discovery',Sparkle],['cards','Feature cards',Graph],
            ['proposals','Proposals',MagnifyingGlass]];

const num=v=>v==null?'—':typeof v==='number'?v:String(v);

export function Features(){
 const [tab,setTab]=useState('discover');
 const [meta,setMeta]=useState(null);const [report,setReport]=useState(null);
 const [cards,setCards]=useState(null);const [props,setProps]=useState(null);
 const [error,setError]=useState('');const [loading,setLoading]=useState(false);
 const [needsKey,setNeedsKey]=useState(false);

 async function load(){setLoading(true);setError('');
  try{
   const [m,p]=await Promise.all([api('/features'),api('/features/proposals')]);
   setMeta(m);setProps(p);
  }catch(e){if(e.authRequired)setNeedsKey(true);else setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);

 async function run(){setLoading(true);setError('');
  try{setReport((await api('/features/discover',{stage:'stage2',generators:['ratio','gap','log'],limit:300})).report);}
  catch(e){if(e.authRequired)setNeedsKey(true);else setError(e.message);}finally{setLoading(false);}}

 async function loadCards(){setLoading(true);setError('');
  try{setCards(await api('/features/cards?stage=stage2&target=defaulted'));}
  catch(e){if(e.authRequired)setNeedsKey(true);else setError(e.message);}finally{setLoading(false);}}

 if(needsKey)return <ApiKeyGate message="The governance and feature surfaces need a role." onSaved={()=>{setNeedsKey(false);load();}}/>;
 if(error&&!meta)return <div className="operations"><InfoBox warning><span role="alert">{error}</span></InfoBox></div>;
 if(!meta)return <div className="operations"><Empty icon={Flask} title="Reading the feature contract">One moment…</Empty></div>;

 return <div className="operations">
  <div className="page-intro"><h1>Feature Intelligence</h1>
   <p>Candidate features generated from each stage&apos;s own contract, tested for signal,
    redundancy, stability and coverage — and refused on four grounds, one of which is not
    statistical at all.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  <InfoBox><b>Discovery can be automated. Admission cannot.</b> A feature with no
   human-authored reason label cannot appear in a customer-facing explanation, so it cannot be
   admitted however strong it is — and the agent will not write that label itself. Widening the
   contract is an approved change requiring model risk <em>and</em> credit policy.</InfoBox>

  <div className="model-metrics">
   <div><small>Contract version</small><b>{meta.version}</b></div>
   <div><small>Reason labels</small><b>{number(meta.reason_labels)}</b></div>
   <div><small>Proposals</small><b>{number(meta.proposals.proposals)}</b></div>
   <div><small>Blocked only on a label</small><b>{number(meta.proposals.blocked_only_on_a_reason_label)}</b></div>
   <div><small>Generators</small><b>{Object.keys(meta.agent.generators).length}</b></div>
   <div><small>Refusal grounds</small><b>{Object.keys(meta.agent.refusals).length}</b></div>
  </div>

  <div className="tabs" role="tablist">{TABS.map(([k,label,Icon])=>
    <button key={k} role="tab" aria-selected={k===tab} className={k===tab?'active':''}
      onClick={()=>setTab(k)}><Icon size={15}/> {label}</button>)}</div>

  {tab==='discover'&&<>
   <div className="actions end">
    <Button busy={loading} primary onClick={run}><Sparkle/> Run discovery</Button>
    <Button busy={loading} onClick={load}><ArrowClockwise/> Refresh</Button></div>

   {!report&&<Empty icon={Sparkle} title="Nothing generated yet">
     Run discovery to generate candidates from the stage-2 contract and test each one.</Empty>}

   {report&&<>
    <div className="model-metrics">
     <div><small>Generated</small><b>{number(report.generated)}</b></div>
     <div><small>Evaluated</small><b>{number(report.evaluated)}</b></div>
     <div><small>Admissible</small><b>{report.admissible.length}</b></div>
     <div><small>Awaiting a label</small><b>{report.awaiting_a_reason_label.length}</b></div>
    </div>

    <Panel title="Why candidates were refused" icon={Prohibit}>
     <DataTable columns={[
       {label:'Ground',render:r=><b>{titleCase(r[0])}</b>},
       {label:'Candidates',render:r=>number(r[1].count)},
       {label:'What it means',render:r=><small>{r[1].why}</small>},
      ]} rows={Object.entries(report.refusals)}/>
     <p className="small-copy">A refusal is not a low score. A candidate refused on any of these
      cannot be rescued by scoring better elsewhere.</p>
    </Panel>

    <Panel title={`Strong, stable, independent — and unexplainable (${report.awaiting_a_reason_label.length})`}
      icon={Warning}>
     <p className="small-copy">The useful output. Each is a question for a human: <em>is there an
      honest sentence to put next to this?</em> Often the answer is no, and then the candidate
      should not exist.</p>
     <DataTable columns={[
       {label:'Candidate',render:c=><span><b>{c.name}</b><br/><small className="muted"><code>{c.recipe}</code></small></span>},
       {label:'Signal',render:c=>num(c.tests.target_spearman)},
       {label:'Redundancy',render:c=><span>{num(c.tests.max_correlation_with_contract)}
         <br/><small className="muted">vs {c.tests.most_correlated_with||'—'}</small></span>},
       {label:'Stability (PSI)',render:c=>num(c.tests.psi_between_halves)},
       {label:'Coverage',render:c=>num(c.tests.coverage)},
      ]} rows={report.awaiting_a_reason_label}/>
    </Panel>

    <Panel title="The proven function registry">
     <DataTable columns={[
       {label:'Generator',render:g=><b>{titleCase(g[0])}</b>},
       {label:'Produces',render:g=><code>{g[1].produces}</code>},
       {label:'Applies to',render:g=><small>{g[1].applies_to}</small>},
       {label:'Why it exists',render:g=><small>{g[1].why}</small>},
      ]} rows={Object.entries(meta.agent.generators).map(([k])=>[k,{
        produces:{ratio:'a / b',gap:'a - b',log:'log1p(x)',interaction:'a * b'}[k]||'—',
        applies_to:meta.agent.generators[k],why:meta.agent.generators[k]}])}/>
     <p className="small-copy">{meta.agent.leakage}</p>
    </Panel>
   </>}
  </>}

  {tab==='cards'&&<>
   <div className="actions end"><Button busy={loading} primary onClick={loadCards}>
     <Graph/> Score the contract</Button></div>
   {!cards&&<Empty icon={Graph} title="Not scored yet">
     Apply the same evidence standard to the features already in production.</Empty>}
   {cards&&<>
    <InfoBox warning>A contract feature reading <b>review</b> or <b>reject</b> is not a bug — it is
     the standard being applied to what is already live. Two caveats: the decision is
     target-specific, and VIF is not.</InfoBox>
    <div className="model-metrics">
     {Object.entries(cards.by_decision).map(([k,v])=>
       <div key={k}><small>{titleCase(k)}</small><b>{v}</b></div>)}
     <div><small>Target</small><b>{cards.target}</b></div>
     <div><small>Lineage edges</small><b>{cards.lineage.edges.length}</b></div>
    </div>
    <Panel title="Feature cards" icon={Graph}>
     <DataTable columns={[
       {label:'Feature',render:c=><span><b>{c.feature}</b><br/><small className="muted">{c.family}</small></span>},
       {label:'IV',render:c=><span>{num(c.evidence.information_value)}
         <br/><small className="muted">{c.evidence.iv_strength||''}</small></span>},
       {label:'MI',render:c=>num(c.evidence.mutual_information)},
       {label:'p',render:c=>num(c.evidence.p_value)},
       {label:'PSI',render:c=>num(c.evidence.psi)},
       {label:'VIF',render:c=><span className={(c.evidence.vif||0)>5?'':''}>{num(c.evidence.vif)}</span>},
       {label:'Decision',render:c=><span><Badge tone={decisionTone(c.decision)}>{c.decision}</Badge>
         {c.reasons?.length>0&&<><br/><small className="muted">{c.reasons[0]}</small></>}</span>},
      ]} rows={cards.cards}/>
     <p className="small-copy">{cards.note}</p>
    </Panel>
   </>}
  </>}

  {tab==='proposals'&&props&&<Panel title="Candidates put forward" icon={MagnifyingGlass}>
   <p className="small-copy">{props.summary.note}</p>
   <DataTable columns={[
     {label:'Proposal',render:p=><span><b>{p.name}</b><br/><small className="muted"><code>{p.recipe}</code></small></span>},
     {label:'Stage',render:p=><Badge>{p.stage}</Badge>},
     {label:'Refused for',render:p=>p.refusals.length
       ?<small>{p.refusals.join(', ')}</small>:<Badge tone="green">nothing</Badge>},
     {label:'State',render:p=><Badge tone={p.state==='admitted'?'green':p.state==='declined'?'red':'amber'}>{p.state}</Badge>},
     {label:'By',render:p=><small>{p.proposed_by}<br/>{String(p.proposed_at).slice(0,10)}</small>},
    ]} rows={props.proposals} empty="Nothing has been proposed yet."/>
  </Panel>}
 </div>;
}
