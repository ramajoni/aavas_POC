import React,{useEffect,useState} from 'react';
import {Flask,ArrowClockwise,Sparkle,Prohibit,Graph,MagnifyingGlass,Warning,Funnel,ShieldCheck,ListChecks} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable,ApiKeyGate} from './components';
import './features-funnel.css';

// Feature discovery, and the four refusals. The page leads with what the agent
// cannot do, because that is the design rather than a caveat on it.
const decisionTone=d=>d==='keep'?'green':d==='review'?'amber':'red';
const TABS=[['funnel','Funnel',Funnel],['discover','Discovery',Sparkle],['cards','Feature cards',Graph],
            ['proposals','Proposals',MagnifyingGlass]];

const num=v=>v==null?'—':typeof v==='number'?v:String(v);

// ------------------------------------------------------------ the funnel tab
// Raw attributes -> derived candidates -> passed screening -> approved -> in the
// model, measured by GET /features/funnel. Every count can be listed by name, and
// a contract feature that fails a screen is shown failing it.
const SLIDE=['stage2','risk','conversion','stage1'];
const NAMES={stage2:'Risk-adjusted pricing',risk:'Risk (PD)',conversion:'Acceptance',stage1:'Indicative pricing'};
const VERDICT={screened_and_approved:['green','Screened & approved'],failed_screening:['amber','Failed a screen'],
 screened_not_labelled:['amber','No reason label'],excluded_at_raw:['red','Excluded at raw'],not_in_data:['red','Not in the data']};
const SCREEN={thin_coverage:'coverage',degenerate:'degenerate',unstable:'unstable (PSI)',weak_signal:'weak signal',redundant:'redundant'};
const f3=v=>v==null?'—':Number(v).toFixed(3);
const plural=(n,w)=>`${number(n)} ${w}${n===1?'':'s'}`;
const SUB={
 raw_attributes:d=>`${d.numeric} numeric · ${d.categorical} categorical · ${plural(d.excluded,'column')} excluded`,
 derived_candidates:d=>Object.entries(d).map(([k,v])=>`${k} ${number(v)}`).join(' · '),
 passed_screening:d=>`of ${number(d.tested)} tested · ${d.raw} raw, ${d.derived} derived`,
 approved:d=>`${d.raw} raw, ${d.derived} derived · authored reason label`,
 in_model:d=>`${d.screened_and_approved} screened & approved · ${d.failed_or_excluded} failed a screen or excluded`};

function FunnelBars({steps}){
 const max=Math.max(...steps.map(s=>s.count),1),w=n=>n>0?Math.max(1.5,100*Math.log10(n+1)/Math.log10(max+1)):0;
 return <><div className="ff-list" role="list">{steps.map(s=>{const im=s.stage==='in_model',ok=im?s.detail.screened_and_approved:s.count;
  return <div key={s.stage} className="ff-row" role="listitem" title={`${s.label}: ${number(s.count)} — ${SUB[s.stage](s.detail)}`}>
   <div className="ff-label"><b>{s.label}</b><small>{SUB[s.stage](s.detail)}</small></div>
   <div className="ff-track"><span className={`ff-bar ${im?'split':''}`} style={{width:`${w(s.count)}%`}}>
    {im&&<span className="ff-ok" style={{width:`${s.count?100*ok/s.count:0}%`}}/>}</span></div>
   <b className="ff-count">{number(s.count)}</b></div>;})}</div>
  <div className="ff-legend"><span>Bar length on a log scale; the count is exact.</span>
   <span><i/>screened &amp; approved</span><span><i className="hatch"/>in the contract, failed a screen or excluded</span></div></>;
}

function FunnelTab(){
 const [model,setModel]=useState('stage2');const [r,setR]=useState(null);const [busy,setBusy]=useState(false);
 const [error,setError]=useState('');const [all,setAll]=useState(false);
 async function load(m=model,refresh=false){setBusy(true);setError('');
  try{setR(await api(`/features/funnel?model=${encodeURIComponent(m)}${refresh?'&refresh=true':''}`));}
  catch(e){setError(e.message);}finally{setBusy(false);}}
 useEffect(()=>{load(model);},[model]);
 const models=r?.models?[...SLIDE.filter(m=>m in r.models),...Object.keys(r.models).filter(m=>!SLIDE.includes(m))]:SLIDE;
 const label=m=>r?.models?.[m]||NAMES[m]||titleCase(m);
 const st=r?.stages;const rej=r?.rejected_unexplainable||[];
 const exclusions=st?Object.entries(st.raw_attributes.excluded_by_reason).map(([k,v])=>({id:k,reason:k,...v,
   columns:st.raw_attributes.exclusions.filter(e=>e.reason===k)})):[];
 const screens=st?Object.entries(st.screening.screens).map(([k,v])=>({id:k,code:k,...v,
   raw:st.screening.refusals_any[k].raw,derived:st.screening.refusals_any[k].derived,
   first:st.screening.refusals_first[k].raw+st.screening.refusals_first[k].derived})):[];
 const latent=(r?.latent_audit?.rows||[]).filter(x=>(x.association||0)>=0.5);
 const served=r?.served;

 return <>
  <div className="ff-head">
   <div className="chip-row" role="group" aria-label="Model">{models.map(m=>
    <button key={m} className={`chip ${m===model?'active':''}`} aria-pressed={m===model} onClick={()=>{setAll(false);setModel(m);}}>{label(m)}</button>)}</div>
   <Button busy={busy} onClick={()=>load(model,true)}><ArrowClockwise/> Recompute</Button></div>
  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}
  {!r&&<Empty icon={Funnel} title="Measuring the funnel">A saved report is served when it is fresh; otherwise
   the funnel is computed from the training table now, which takes a few seconds.</Empty>}
  {r&&<>
   <Panel title={`${r.display_name} — from raw attributes to the model`} icon={Funnel}>
    <div className="ff-served">
     <Badge tone={served?.from==='saved_report'?'green':'amber'}>{served?.from==='saved_report'?'saved report, fresh':'computed live'}</Badge>
     <span>computed {String(r.computed_at).replace('T',' ').slice(0,16)} UTC · {r.seconds.total}s</span>
     {served?.stale_because?.length>0&&<span>recomputed because {served.stale_because.join('; ')}</span>}
     {busy&&<span>refreshing…</span>}</div>
    <FunnelBars steps={r.funnel}/>
    <div className="model-metrics">
     <div><small>Target</small><b>{r.target||'none'}</b></div>
     <div><small>Rows</small><b>{number(r.rows.population)}</b></div>
     <div><small>Correlation sample</small><b>{number(r.rows.correlation_sample)}</b></div>
     <div><small>Seed</small><b>{r.rows.seed}</b></div>
     <div><small>Unexplainable</small><b>{number(rej.length)}</b></div>
    </div>
    <p className="small-copy">{r.rows.population_rule}. {r.target?'':'With no target there is no signal screen, so more passes. '}
     Approval is an authored reason label — this page counts them and never writes one.</p>
   </Panel>

   <Panel title={`Rejected: strong, stable and unexplainable (${number(rej.length)})`} icon={Warning}>
    <p className="small-copy">Derived candidates that pass every statistical screen and have no human-authored reason
     label, strongest first. Each is a question for a person — <em>is there an honest sentence for this?</em> — and
     until someone writes one, it does not enter a model.</p>
    <DataTable columns={[
      {label:'Candidate',render:c=><span><b>{c.name}</b><br/><small className="muted"><code>{c.recipe}</code></small></span>},
      {label:'Family',render:c=><Badge>{c.family}</Badge>},
      {label:'Signal',render:c=><span className="ff-num">{f3(c.signal)}</span>},
      {label:'Overlap',render:c=><span className="ff-num">{f3(c.overlap)}<br/><small className="muted">vs {c.overlap_with||'—'}</small></span>},
      {label:'PSI',render:c=><span className="ff-num">{f3(c.psi)}</span>},
     ]} rows={all?rej:rej.slice(0,12)} empty="No derived candidate passed every screen."/>
    {rej.length>12&&<div className="actions end"><Button onClick={()=>setAll(!all)}>{all?'Show the top 12':`Show all ${number(rej.length)}`}</Button></div>}
    <p className="small-copy">Signal: {r.target?`Spearman with ${r.target}`:'not measured (no target)'}. Overlap: the largest |Spearman| with
     any {r.model} contract feature. PSI: older half against newer half, by application date.</p>
   </Panel>

   <Panel title={`In the model: every contract feature, reconciled (${st.in_model.count})`} icon={ListChecks}>
    <DataTable columns={[
      {label:'Feature',render:x=><span><b>{x.feature}</b><br/><small className="muted">{x.declared}{x.group?` · ${titleCase(x.group)}`:''}</small></span>},
      {label:'Verdict',render:x=><Badge tone={(VERDICT[x.verdict]||['blue'])[0]}>{(VERDICT[x.verdict]||[0,x.verdict])[1]}</Badge>},
      {label:'Why',render:x=>x.exclusion?<small className="ff-fail">{x.exclusion.reason}: {x.exclusion.detail||x.exclusion.why}</small>
        :x.failed?.length?<small className="ff-fail">{x.failed.map(f=>SCREEN[f]||f).join(', ')}
          {x.tests?.redundant_with&&` (vs ${x.tests.redundant_with})`}{x.missing_explained_by&&` — missing exactly when ${x.missing_explained_by}`}</small>
        :<small className="muted">all screens passed{(x.tests?.overlap||0)>r.thresholds.max_abs_correlation
          &&` — kept as the stronger of a correlated pair with ${x.tests.overlap_with}`}</small>},
      {label:'Signal',render:x=><span className="ff-num">{f3(x.tests?.signal)}</span>},
      {label:'Overlap',render:x=><span className="ff-num">{f3(x.tests?.overlap)}</span>},
      {label:'PSI',render:x=><span className="ff-num">{f3(x.tests?.psi)}</span>},
      {label:'Coverage',render:x=><span className="ff-num">{f3(x.tests?.coverage)}</span>},
      {label:'Label',render:x=><Badge tone={x.reason_label?'green':'red'}>{x.reason_label?'authored':'none'}</Badge>},
     ]} rows={st.in_model.reconciliation}/>
    <p className="small-copy">A contract feature that fails a screen is shown failing it, not dropped and not passed.
     Every screen is univariate: a feature that matters only in combination can fail the signal screen and still earn
     its place in the fitted model, whose attributions are the multivariate check.
     {st.review.approved_not_in_model.length>0&&<> Approved but not in this contract: {st.review.approved_not_in_model.join(', ')}.</>}</p>
   </Panel>

   <Panel title="Screening: why candidates were refused" icon={Prohibit}>
    <DataTable columns={[
      {label:'Screen',render:x=><b>{titleCase(x.label)}</b>},
      {label:'Raw',render:x=><span className="ff-num">{number(x.raw)}</span>},
      {label:'Derived',render:x=><span className="ff-num">{number(x.derived)}</span>},
      {label:'First failure',render:x=><span className="ff-num">{number(x.first)}</span>},
      {label:'What it means',render:x=><small>{x.why}</small>},
     ]} rows={screens}/>
    <p className="small-copy">A candidate can fail several screens; “first failure” counts each once, in the order shown.
     Thresholds: coverage ≥ {r.thresholds.min_coverage}, PSI ≤ {r.thresholds.max_psi}, |signal| ≥ {r.thresholds.min_abs_target_corr}{' '}
     (bias-corrected Cramér&apos;s V ≥ {r.thresholds.min_cramers_v} for categoricals), redundancy ≤ {r.thresholds.max_abs_correlation}.</p>
   </Panel>

   <Panel title={`Exclusion audit: leakage control (${st.raw_attributes.exclusions.length} columns)`} icon={ShieldCheck}>
    <DataTable columns={[
      {label:'Reason',render:x=><span><Badge tone={['latent','outcome','decision','decision_derived'].includes(x.reason)?'red':'blue'}>{titleCase(x.reason)}</Badge></span>},
      {label:'Columns',render:x=><div className="ff-cols">{x.columns.map(c=><code key={c.column} title={c.detail||''}>{c.column}{c.is_target?' (target)':''}</code>)}</div>},
      {label:'Why',render:x=><small>{x.why}</small>},
     ]} rows={exclusions}/>
    {st.raw_attributes.sentinels.map(sn=><p key={sn.column} className="small-copy">Sentinel: <code>{sn.column}</code> {JSON.stringify(sn.rule)} —
     {' '}{number(sn.rows_read_as_missing)} rows read as missing before any test ({sn.source}).</p>)}
    {latent.length>0&&<><InfoBox warning>Excluding the latent columns is half the leakage question. These raw attributes track
     a latent column closely in this synthetic data — evaluated only, nothing here changes a screen.</InfoBox>
     <DataTable columns={[
       {label:'Attribute',render:x=><b>{x.column}</b>},
       {label:'Tracks',render:x=><code>{x.latent}</code>},
       {label:'Association',render:x=><span className="ff-num">{f3(x.association)}</span>},
       {label:'Measure',render:x=><small>{x.measure}</small>},
      ]} rows={latent}/></>}
   </Panel>

   <Panel title="How each number was measured">
    <ul className="ff-method">{Object.entries(r.method).map(([k,v])=><li key={k}><b>{titleCase(k)}.</b> {v}</li>)}</ul>
   </Panel>
  </>}
 </>;
}

export function Features(){
 const [tab,setTab]=useState('funnel');
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

  {tab==='funnel'&&<FunnelTab/>}

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
        produces:{ratio:'a / b',gap:'a - b',log:'log1p(x)',interaction:'a * b',trend:'x - median(x, prior 90d)'}[k]||'—',
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
