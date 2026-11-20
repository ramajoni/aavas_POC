import React,{useEffect,useId,useMemo,useRef,useState} from 'react';
import {FlowArrow,Database,ShieldCheck,Brain,CloudArrowUp,Play,ArrowClockwise,ArrowLeft,ArrowRight,CheckCircle,WarningCircle,XCircle,CircleNotch,Circle,MinusCircle,HourglassMedium,FloppyDisk,Trophy,Terminal,ListChecks,Rocket,Lock,PaperPlaneTilt,ClockCounterClockwise} from '@phosphor-icons/react';
import {api,number,titleCase} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable,Modal} from './components';
import {setValues,nextVersion} from './agentic-yaml';
import './agentic-os.css';

// Agentic OS: rebuild the pricing models as a workflow of agents.
//   Data -> Policy -> Training -> Serving, then Run.
// The settings live in config/agentic_os.yaml and are saved through the same
// versioned, audited /config mechanism as every other configuration file; a run
// always uses the saved settings. Operators (administrator, model validation)
// can check, run and promote; only an administrator can change the settings.

const STEPS=[
 {key:'data',title:'Data',icon:Database},
 {key:'policy',title:'Policy',icon:ShieldCheck},
 {key:'training',title:'Training',icon:Brain},
 {key:'serving',title:'Serving',icon:CloudArrowUp},
 {key:'run',title:'Run',icon:Play},
];
const STEP_TITLE=Object.fromEntries(STEPS.map(s=>[s.key,s.title]));

const MODELS=[
 ['stage1','Indicative rate band','Stage 1: the range quoted before verification'],
 ['stage2','Final rate','Stage 2: the rate after verification'],
 ['conversion','Acceptance','How likely the customer is to take the offer'],
 ['prepayment','Prepayment','How likely the loan is to be paid off early'],
 ['lgd','Loss given default','What is lost if the loan defaults'],
 ['risk','Default risk','How likely the loan is to default'],
 ['override','Override','How likely a rate is to be overridden, and by how much'],
 ['uplift','Discount effect','How much a discount changes acceptance'],
];
const MODEL_NAME=Object.fromEntries(MODELS.map(([k,n])=>[k,n]));

const CHECK_TEXT={
 rates_inside_envelope:'Every rate is inside the policy envelope',
 no_rate_above_card:'No loan is priced above its card rate',
 tenure_in_range:'Every tenure is inside policy limits',
 stressed_affordability:'Every loan passes the stressed affordability test',
 bureau_minimum:'Scored applicants meet the minimum bureau score',
 no_declined_grade:'No loan is in a grade the policy declines',
 validator_clean:'Every row passes ingestion validation',
};

const SOURCES=[
 {value:'synthetic',title:'Synthetic book',text:'Generate a new book of loans under the policy in force.'},
 {value:'live',title:'Live dataset',text:'Use the current dataset. The run works on a copy; the live file is not changed.'},
];
const ENGINES=[
 {value:'local',title:'Local',icon:Brain,text:'Trains on this server. The only engine whose models can be promoted to champion.'},
 {value:'sagemaker',title:'Amazon SageMaker',icon:CloudArrowUp,text:'Runs a SageMaker training job in your AWS account. Needed for serverless serving.'},
 {value:'autogluon_assistant',title:'AutoGluon Assistant',icon:FlowArrow,text:'An AI agent on Amazon Bedrock writes and runs its own AutoGluon code for one model. Kept as a challenger.'},
];
const SERVING=[
 {value:'in_process',title:'In-process',text:'The pricing API loads the models itself. A local run replaces the champions once approved.'},
 {value:'sagemaker_serverless',title:'SageMaker Serverless',text:'Deploys the SageMaker-trained bundle to a serverless endpoint and checks it answers as training recorded.'},
];
const REGISTER=[
 {value:'challenger',title:'Challenger',text:'Kept beside the current champions until someone promotes it.'},
 {value:'champion',title:'Champion',text:'Recorded as champion in the model registry when training finishes.'},
];
const MEMORY=[1024,2048,3072,4096,5120,6144];
const INSTANCE_TYPES=['ml.m5.large','ml.m5.xlarge','ml.m5.2xlarge','ml.m5.4xlarge','ml.c5.xlarge','ml.c5.2xlarge','ml.c5.4xlarge'];

// Every setting this screen can change, as dotted paths into the file.
const EDITABLE=[
 'data.source','data.rows','data.seed',
 'policy.require_consistency_checks','policy.require_sample_cases',
 'training.engine','training.models','training.register_as',
 'training.sagemaker.instance_type','training.sagemaker.max_runtime_minutes','training.sagemaker.bucket','training.sagemaker.prefix',
 'training.autogluon_assistant.model','training.autogluon_assistant.target','training.autogluon_assistant.max_iterations',
 'training.autogluon_assistant.time_limit_minutes','training.autogluon_assistant.instructions',
 'serving.mode','serving.promote_on_success',
 'serving.sagemaker_serverless.endpoint_name','serving.sagemaker_serverless.memory_mb','serving.sagemaker_serverless.max_concurrency',
];

const METRIC_LABEL={coverage_pct:'Coverage (%)',mean_width_bps:'Mean band width (bps)',median_mae_bps:'Median error (bps)',
 mae_bps:'Mean error (bps)',within_25bps_pct:'Within 0.25% (%)',r2:'R²',auc:'AUC',calibration_gap:'Calibration gap',
 base_rate:'Base rate',magnitude_mae_bps:'Size error (bps)',rmse_pts:'RMSE (points)',policy_rmse_pts:'RMSE vs policy (points)',
 qini:'Qini',ate_pts:'Average effect (points)'};
// +1: higher is better, -1: lower is better. Anything else is described, not judged.
const METRIC_DIRECTION={auc:1,r2:1,within_25bps_pct:1,coverage_pct:1,qini:1,mae_bps:-1,median_mae_bps:-1,mean_width_bps:-1,
 calibration_gap:-1,magnitude_mae_bps:-1,rmse_pts:-1,policy_rmse_pts:-1};

const SAMPLE_ROW={segment:'salaried_formal',city:'Jaipur',channel:'branch_walkin',property_type:'resale_flat',
 verified_monthly_income:53000,sanctioned_loan_amount:2000000,appraised_property_value:2900000,tenure_years:10,
 applicant_age:35,declared_obligations:5000,bureau_score:750,bureau_vintage_months:96,delinquency_count_24m:0,
 banking_behaviour_score:75,ltv:0.69,foir:0.38,new_to_credit:false,offered_rate:14.3,competitor_rate:14.1,
 competitor_gap_bps:-20,customer_negotiating:true,prior_negotiation_rounds:1,existing_relationship:false,
 competitor_evidence_verified:false};

const LIVE=new Set(['queued','running']);

// ------------------------------------------------------------------ helpers
const getIn=(obj,path)=>path.split('.').reduce((o,k)=>o==null?undefined:o[k],obj);
const setIn=(obj,[k,...rest],value)=>({...obj,[k]:rest.length?setIn(obj?.[k]||{},rest,value):value});
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const when=s=>{if(!s)return '—';const d=new Date(String(s).replace(' ','T'));
 return Number.isNaN(d.getTime())?String(s):d.toLocaleString('en-IN',{dateStyle:'medium',timeStyle:'short'});};
const took=(a,b)=>{if(!a)return '—';const s=Math.max(0,Math.round(((b?new Date(b):new Date())-new Date(a))/1000));
 return s<60?`${s} s`:s<3600?`${Math.floor(s/60)} min ${s%60} s`:`${Math.floor(s/3600)} h ${Math.floor(s%3600/60)} min`;};
const fmt=v=>{if(v==null||v==='')return '—';if(typeof v!=='number')return String(v);const a=Math.abs(v);
 return a>=1000?number(Math.round(v)):a>=10?v.toFixed(1):a>=1?v.toFixed(2):a>=0.001||a===0?v.toFixed(3):v.toPrecision(2);};
const whole=(v,min,max=Infinity)=>Number.isInteger(v)&&v>=min&&v<=max;
const numberValue=e=>e.target.value===''?'':Number(e.target.value);
const engineTitle=v=>ENGINES.find(e=>e.value===v)?.title||titleCase(v);
const servingTitle=v=>SERVING.find(e=>e.value===v)?.title||titleCase(v);
// A 403 from the API arrives without a status (see api.js); its message names the role.
const refused=e=>e?.status===403||(!e?.status&&/administrator|role|permission|forbidden|not allowed/i.test(e?.message||''));
const saveError=e=>refused(e)?`Not saved: only an administrator can change the Agentic OS settings. (${e.message})`:e.message;

// What would stop the settings being saved or a run working, per step.
function validate(c){
 const out=[];const add=(step,path,message)=>out.push({step,path,message});
 const d=c?.data||{},t=c?.training||{},s=t.sagemaker||{},g=t.autogluon_assistant||{},v=c?.serving||{},e=v.sagemaker_serverless||{};
 if(!whole(d.rows,1))add('data','data.rows','Rows must be a whole number above 0.');
 if(!whole(d.seed,0))add('data','data.seed','Seed must be a whole number, 0 or more.');
 if(t.engine!=='autogluon_assistant'&&!(t.models||[]).length)add('training','training.models','Choose at least one model to train.');
 if(!/^ml\.[a-z0-9]+\.[a-z0-9]+$/.test(s.instance_type||''))add('training','training.sagemaker.instance_type','Instance type must be a SageMaker type, such as ml.m5.xlarge.');
 if(!whole(s.max_runtime_minutes,1))add('training','training.sagemaker.max_runtime_minutes','Maximum run time must be a whole number of minutes above 0.');
 // Empty: the server's AAVAS_SAGEMAKER_BUCKET (it carries the account number, so it is not committed).
 if(s.bucket&&!/^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/.test(s.bucket))add('training','training.sagemaker.bucket','Bucket must be a valid S3 bucket name (lower case, numbers, dots and hyphens).');
 if(!String(s.prefix||'').trim())add('training','training.sagemaker.prefix','Prefix cannot be empty.');
 if(!String(g.model||'').trim())add('training','training.autogluon_assistant.model','Enter the Amazon Bedrock model ID.');
 if(!whole(g.max_iterations,1))add('training','training.autogluon_assistant.max_iterations','Iterations must be a whole number above 0.');
 if(!whole(g.time_limit_minutes,1))add('training','training.autogluon_assistant.time_limit_minutes','Time limit must be a whole number of minutes above 0.');
 if(!/^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$/.test(e.endpoint_name||''))add('serving','serving.sagemaker_serverless.endpoint_name','Endpoint name: letters, numbers and hyphens, up to 63 characters.');
 if(!MEMORY.includes(e.memory_mb))add('serving','serving.sagemaker_serverless.memory_mb','Memory must be 1024 to 6144 MB in steps of 1024.');
 if(!whole(e.max_concurrency,1,200))add('serving','serving.sagemaker_serverless.max_concurrency','Concurrency must be a whole number from 1 to 200.');
 if(v.mode==='sagemaker_serverless'&&t.engine!=='sagemaker')add('serving','serving.mode','Serverless serving needs the SageMaker training engine. Choose SageMaker on the Training step, or serve in-process.');
 return out;
}

// Things worth knowing before a run that do not stop it.
function cautions(c,readiness){
 const out=[];const t=c?.training||{};
 if(t.engine==='sagemaker'&&readiness?.boto3&&!readiness.boto3.ok)out.push('boto3 is not installed on this server, so a SageMaker run cannot start a training job.');
 if(t.engine==='autogluon_assistant'&&readiness?.autogluon_assistant&&!readiness.autogluon_assistant.ok)out.push(`The AutoGluon Assistant command line was not found at ${readiness.autogluon_assistant.cli}.`);
 if(t.engine==='autogluon_assistant'&&readiness?.autogluon_assistant?.ok&&readiness.autogluon_assistant.allowed===false)out.push('The AutoGluon Assistant is installed, but it is not allowed on this host (training.autogluon_assistant.allow_unsandboxed is false), so a run with it will be refused.');
 if(c?.policy&&(!c.policy.require_consistency_checks||!c.policy.require_sample_cases))out.push('A policy check is switched off, so the run can train on a policy that fails it.');
 if(c?.serving?.promote_on_success)out.push('Results are promoted automatically: a passing local run replaces the champions with no separate approval.');
 if(t.register_as==='champion')out.push('Models are registered as champion as soon as training finishes.');
 return out;
}

// ------------------------------------------------------------------ small pieces
function StatusBadge({status}){
 const map={succeeded:['green',CheckCircle,'Succeeded'],failed:['red',XCircle,'Failed'],refused:['red',MinusCircle,'Refused'],
  running:['blue',CircleNotch,'Running'],queued:['amber',HourglassMedium,'Queued'],pending:['grey',Circle,'Pending']};
 const [tone,Icon,label]=map[status]||['grey',Circle,titleCase(status||'unknown')];
 return <Badge tone={tone}><Icon size={12} weight="bold" className={status==='running'?'spin':''} aria-hidden="true"/> {label}</Badge>;
}

function Setting({label,hint,error,children}){
 const id=useId();const hintId=hint?`${id}-hint`:undefined;const errorId=error?`${id}-error`:undefined;
 return <div className="field aos-setting">
  <label htmlFor={id}>{label}</label>
  {React.cloneElement(children,{id,'aria-describedby':[hintId,errorId].filter(Boolean).join(' ')||undefined,'aria-invalid':error?true:undefined})}
  {hint&&<small id={hintId}>{hint}</small>}
  {error&&<small id={errorId} className="aos-error"><WarningCircle size={13} weight="fill" aria-hidden="true"/> {error}</small>}
 </div>;
}

function Options({legend,name,options,value,onChange,hint}){
 return <fieldset className="aos-options-group">
  <legend>{legend}</legend>
  {hint&&<p className="small-copy">{hint}</p>}
  <div className="aos-options">
   {options.map(o=><label key={o.value} className={`aos-option${value===o.value?' selected':''}`}>
    <input type="radio" name={name} value={o.value} checked={value===o.value} onChange={()=>onChange(o.value)}/>
    <span><b>{o.icon&&<o.icon size={16} aria-hidden="true"/>} {o.title}{value===o.value&&<span className="aos-selected-tag">Selected</span>}</b>
     <small>{o.text}</small></span>
   </label>)}
  </div>
 </fieldset>;
}

function Check({checked,onChange,children,hint}){
 return <label className="aos-check"><input type="checkbox" checked={!!checked} onChange={e=>onChange(e.target.checked)}/>
  <span><span className="aos-check-label">{children}</span>{hint&&<small>{hint}</small>}</span></label>;
}

function Chip({ok,children}){
 return <span className={`aos-chip ${ok?'ok':'missing'}`}>{ok?<CheckCircle size={14} weight="fill" aria-hidden="true"/>:<XCircle size={14} weight="fill" aria-hidden="true"/>} {children}</span>;
}

function Facts({items}){
 return <dl className="aos-facts">{items.filter(Boolean).map(([k,v])=><div key={k}><dt>{k}</dt><dd>{v==null||v===''?'—':v}</dd></div>)}</dl>;
}

function KeyValues({obj,skip=[]}){
 const rows=Object.entries(obj||{}).filter(([k])=>!skip.includes(k));
 if(!rows.length)return null;
 return <dl className="aos-kv">{rows.map(([k,v])=><div key={k}><dt>{titleCase(k)}</dt><dd>{v==null?'—':typeof v==='boolean'?(v?'yes':'no')
  :typeof v==='object'?<pre className="aos-json">{JSON.stringify(v,null,1)}</pre>:String(v)}</dd></div>)}</dl>;
}

function ReadOnly({items}){
 return <dl className="aos-kv aos-readonly">{items.map(([k,v])=><div key={k}><dt><Lock size={12} aria-hidden="true"/> {k}</dt><dd><code>{v==null?'—':String(v)}</code></dd></div>)}</dl>;
}

// ------------------------------------------------------------------ step details (run view and policy check)
function DataDetails({d}){
 const p=d?.profile;
 return <>
  {p&&<Facts items={[
   ['Loans',number(p.rows)],
   p.dates&&['Applications',`${p.dates[0]} to ${p.dates[1]}`],
   p.sanctioned_rate&&['Sanctioned rate',`${fmt(p.sanctioned_rate.min)}% to ${fmt(p.sanctioned_rate.max)}% (median ${fmt(p.sanctioned_rate.median)}%)`],
   p.tenure_months&&['Tenure',`${p.tenure_months[0]} to ${p.tenure_months[1]} months`],
   ['Disbursed',p.disbursed_pct==null?'—':`${p.disbursed_pct}%`],
   p.grades&&['Loans by grade',Object.entries(p.grades).map(([g,n])=>`${g}: ${number(n)}`).join(' · ')],
  ]}/>}
  {d?.checks&&<DataTable columns={[
   {label:'Check',render:c=><><span>{CHECK_TEXT[c.check]||titleCase(c.check)}</span> <code className="aos-code">{c.check}</code></>},
   {label:'What it looks for',render:c=><small>{c.what}</small>},
   {label:'Loans breaking it',render:c=>number(c.violations)},
   {label:'Result',render:c=><Badge tone={c.ok?'green':'red'}>{c.ok?'Pass':'Fail'}</Badge>},
  ]} rows={d.checks.map(c=>({...c,id:c.check}))}/>}
 </>;
}

function PolicyDetails({r}){
 if(!r)return null;
 const c=r.consistency||{},s=r.sample_cases||{};
 const failedCases=s.failed||[],failedChecks=c.failed||[];
 return <>
  <Facts items={[
   ['Policy',r.policy_version],['Status',r.status&&titleCase(r.status)],['In force from',r.effective_from],
   ['Consistency checks',c.checks!=null?`${c.passed} of ${c.checks} pass${c.skipped?`, ${c.skipped} skipped`:''}`:'—'],
   ['Sample cases',s.cases!=null?`${s.passed} of ${s.cases} pass`:'—'],
  ]}/>
  {failedChecks.length>0&&<><h4 className="aos-subhead">Consistency checks that fail ({failedChecks.length})</h4>
   <DataTable columns={[{label:'Check',render:x=><code className="aos-code">{x.check}</code>},{label:'What is wrong',render:x=><small>{x.detail}</small>}]}
    rows={failedChecks.map((x,i)=>({...x,id:`${x.check}-${i}`}))}/></>}
  {failedCases.length>0&&<><h4 className="aos-subhead">Sample cases the engine gets wrong ({failedCases.length})</h4>
   <DataTable columns={[
    {label:'Case',render:x=><code className="aos-code">{x.case_id}</code>},
    {label:'Rule',key:'rule'},
    {label:'What it tests',render:x=><small>{x.description}</small>},
    {label:'Card rate',render:x=>x.card_rate==null?'—':`${fmt(x.card_rate)}%`},
    {label:'Engine said',render:x=><small>{[x.verdict,x.approval].filter(Boolean).map(titleCase).join(' · ')||'—'}</small>},
    {label:'Problem',render:x=><small>{Array.isArray(x.problems)?x.problems.join('; '):String(x.problems??'')}</small>},
   ]} rows={failedCases.map((x,i)=>({...x,id:`${x.case_id}-${i}`}))}/></>}
 </>;
}

function Delta({metric,v}){
 if(v?.new==null)return <span className="muted">—</span>;
 if(v.champion==null)return <span className="muted">no champion</span>;
 const d=v.new-v.champion;
 if(Math.abs(d)<1e-9)return <span className="aos-delta same">no change</span>;
 const dir=METRIC_DIRECTION[metric];const verdict=dir?(d*dir>0?'better':'worse'):'';
 return <span className={`aos-delta ${verdict}`}>{d>0?'+':'−'}{fmt(Math.abs(d))}{verdict&&` · ${verdict}`}</span>;
}

function Comparison({rows}){
 if(!rows?.length)return <p className="small-copy">No models came back to compare.</p>;
 return <div className="table-scroll"><table className="aos-compare">
  <caption className="sr-only">This run's models compared with the current champions</caption>
  <thead><tr><th scope="col">Model</th><th scope="col">Learner</th><th scope="col">Trained on</th><th scope="col">Measure</th>
   <th scope="col">This run</th><th scope="col">Champion</th><th scope="col">Change</th></tr></thead>
  <tbody>{rows.map(r=>{const ms=Object.entries(r.metrics||{});const head=n=><>
    <th scope="row" rowSpan={n}><b>{MODEL_NAME[r.model]||titleCase(r.model)}</b><code className="aos-code">{r.model}</code></th>
    <td rowSpan={n}>{titleCase(r.learner)||'—'}</td><td rowSpan={n}>{r.n_train==null?'—':`${number(r.n_train)} loans`}</td></>;
   if(!ms.length)return <tr key={r.model}>{head(1)}<td colSpan={4} className="muted">No headline measures</td></tr>;
   return ms.map(([k,v],j)=><tr key={`${r.model}-${k}`}>{j===0&&head(ms.length)}
    <td>{METRIC_LABEL[k]||titleCase(k)}</td><td>{fmt(v.new)}</td><td>{fmt(v.champion)}</td><td><Delta metric={k} v={v}/></td></tr>);
  })}</tbody></table></div>;
}

function TrainingDetails({d}){
 return <>
  <KeyValues obj={d} skip={['cards','comparison']}/>
  {d?.comparison&&<><h4 className="aos-subhead">New models against the champions</h4><Comparison rows={d.comparison}/>
   <p className="small-copy">“Better” and “worse” follow each measure's direction (for example, a higher AUC and a lower error are better).</p></>}
 </>;
}

function ServingDetails({d}){
 if(!d)return null;
 const auto=Array.isArray(d.promoted);
 return <>
  {'awaiting_promotion' in d&&<p className="aos-detail-line">{d.awaiting_promotion?'Waiting for an approver to promote the new models.':'Nothing to promote: in-process serving keeps the current champions.'}</p>}
  {auto&&<p className="aos-detail-line">Promoted automatically by {d.approved_by}: {d.promoted.join(', ')}.</p>}
  <KeyValues obj={d} skip={['awaiting_promotion',...(auto?['promoted','approved_by']:[])]}/>
 </>;
}

function StepDetails({name,details}){
 if(name==='data')return <DataDetails d={details}/>;
 if(name==='policy')return <PolicyDetails r={details}/>;
 if(name==='training')return <TrainingDetails d={details}/>;
 if(name==='serving')return <ServingDetails d={details}/>;
 return <KeyValues obj={details}/>;
}

// ------------------------------------------------------------------ run view
function RunStepItem({s,index}){
 const has=s.details&&Object.keys(s.details).length>0;
 const [open,setOpen]=useState(s.status==='failed');
 useEffect(()=>{if(s.status==='failed')setOpen(true);},[s.status]);
 const id=useId();
 const fallback=s.status==='pending'?'Waits for the step before it.':s.status==='running'?'Working…':'';
 return <li className={`aos-run-step ${s.status}`}>
  <div className="aos-run-step-head">
   <span className="aos-run-step-n" aria-hidden="true">{index+1}</span>
   <div className="aos-run-step-body">
    <h4>{STEP_TITLE[s.name]||titleCase(s.name)}</h4>
    <p>{s.summary||fallback}</p>
    {s.started_at&&<small>Started {when(s.started_at)}{s.finished_at?` · finished ${when(s.finished_at)}`:''} · {took(s.started_at,s.finished_at)}</small>}
   </div>
   <StatusBadge status={s.status}/>
  </div>
  {has&&<button type="button" className="text-button aos-toggle" aria-expanded={open} aria-controls={id} onClick={()=>setOpen(!open)}>
   {open?'Hide details':'Show details'}</button>}
  {has&&open&&<div id={id} className="aos-run-step-details"><StepDetails name={s.name} details={s.details}/></div>}
 </li>;
}

function LogPanel({lines}){
 const ref=useRef(null);const stick=useRef(true);
 useEffect(()=>{const el=ref.current;if(el&&stick.current)el.scrollTop=el.scrollHeight;},[lines]);
 const text=(lines||[]).map(l=>{const m=/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^ ]*)\s+(.*)$/.exec(l);
  return m?`${new Date(m[1]).toLocaleTimeString('en-IN',{hour12:false})}  ${m[2]}`:l;}).join('\n');
 return <div className="aos-log-wrap">
  <h4 className="aos-subhead"><Terminal size={16} aria-hidden="true"/> Log <small>{lines?.length?`last ${lines.length} lines`:''}</small></h4>
  <pre ref={ref} className="aos-log" tabIndex={0} role="log" aria-live="off" aria-label="Run log"
   onScroll={e=>{const el=e.currentTarget;stick.current=el.scrollHeight-el.scrollTop-el.clientHeight<40;}}>{text||'No log lines yet.'}</pre>
 </div>;
}

function RunView({run,error,champions,onPromote,promoting,promoteError,onResume,resuming,resumeError}){
 const [confirm,setConfirm]=useState(false);
 const serving=run.steps?.find(s=>s.name==='serving');
 const canPromote=Boolean(serving?.details?.awaiting_promotion)&&!run.promotion;
 const newModels=Object.keys(run.steps?.find(s=>s.name==='training')?.details?.cards||{}).length;
 const done=run.steps?.filter(s=>s.status==='succeeded').length||0;
 useEffect(()=>{if(run.promotion)setConfirm(false);},[run.promotion]);
 return <Panel className="aos-run" title={<>Run <code className="aos-run-id">{run.run_id}</code></>} icon={Rocket} action={<StatusBadge status={run.status}/>}>
  {error&&<InfoBox warning><span role="alert">Could not refresh this run: {error}</span></InfoBox>}
  <Facts items={[
   ['Started by',run.started_by],['Started',when(run.started_at||run.created_at)],
   ['Finished',LIVE.has(run.status)?'still running':when(run.finished_at)],['Took',took(run.started_at||run.created_at,run.finished_at)],
   ['Settings version',run.config_version],['Training',engineTitle(run.engine)],['Serving',servingTitle(run.serving_mode)],
   ['Steps done',`${done} of ${run.steps?.length||0}`],
  ]}/>
  <p className="sr-only" role="status">{`Run ${run.status}. ${done} of ${run.steps?.length||0} steps done.`}</p>
  {run.status==='refused'&&<InfoBox warning>This run was refused: another run was still going. Only one run happens at a time.</InfoBox>}
  {run.status==='failed'&&<div className="aos-promote">
   <div><b>This run stopped at a step.</b> <span>Fix the cause, then resume: finished steps keep their results and the run carries on from the step that stopped it.</span></div>
   <Button primary busy={resuming} onClick={onResume}><ArrowClockwise size={17} aria-hidden="true"/> Resume run</Button>
  </div>}
  {resumeError&&<InfoBox warning><span role="alert">Not resumed: {resumeError}</span></InfoBox>}
  {run.promotion&&<InfoBox success>Promoted to champion by <b>{run.promotion.approved_by}</b> on {when(run.promotion.at)}: {(run.promotion.promoted||[]).join(', ')}. The models they replaced are backed up in this run.</InfoBox>}
  {canPromote&&<div className="aos-promote">
   <div><b>{newModels||'The'} new models are ready.</b> <span>Check the comparison under Training, then promote them to make them the champions the pricing API uses.</span></div>
   <Button primary onClick={()=>setConfirm(true)}><Trophy size={17} aria-hidden="true"/> Promote to champion</Button>
  </div>}
  <ol className="aos-run-steps">{(run.steps||[]).map((s,i)=><RunStepItem key={s.name} s={s} index={i}/>)}</ol>
  <LogPanel lines={run.log}/>
  {confirm&&<Modal title="Promote these models?" onClose={()=>setConfirm(false)}>
   <p>This replaces the {champions||8} champion models with the {newModels||''} models from run <b>{run.run_id}</b>. New prices use them straight away.
    The current ones are backed up in the run.</p>
   {promoteError&&<InfoBox warning><span role="alert">{promoteError}</span></InfoBox>}
   <div className="modal-actions"><Button onClick={()=>setConfirm(false)} disabled={promoting}>Cancel</Button>
    <Button primary busy={promoting} onClick={onPromote}><Trophy size={17} aria-hidden="true"/> Promote</Button></div>
  </Modal>}
 </Panel>;
}

// ------------------------------------------------------------------ save
function SaveModal({baseVersion,edits,draft,onClose,onSaved}){
 const [version,setVersion]=useState(()=>nextVersion(baseVersion));
 const [review,setReview]=useState(null);const [raw,setRaw]=useState(null);
 const [busy,setBusy]=useState(true);const [error,setError]=useState('');
 // Rebuild the file from what is saved now, so only the settings changed here move.
 useEffect(()=>{let alive=true;(async()=>{
  try{
   const file=await api('/config/agentic_os');
   if(String(file.version)!==String(baseVersion))throw new Error(`The settings were changed elsewhere since this page loaded (now ${file.version}). Close this, press Refresh and make your changes again.`);
   const text=setValues(file.raw,Object.fromEntries(edits.map(p=>[p,getIn(draft,p)])));
   const r=await api('/config/agentic_os/preview',{raw:text,version:nextVersion(baseVersion)});
   if(alive){setRaw(text);setReview(r);}
  }catch(e){if(alive)setError(saveError(e));}
  finally{if(alive)setBusy(false);}
 })();return()=>{alive=false;};},[]);
 async function save(){setBusy(true);setError('');
  try{onSaved(await api('/config/agentic_os',{raw,version:version.trim()},{method:'PUT'}));}
  catch(e){setError(saveError(e));setBusy(false);}}
 const changes=(review?.changes||[]).filter(c=>c.path!==review.version_key);
 const show=v=>v==null?'—':Array.isArray(v)?v.join(', '):String(v);
 return <Modal wide title="Save the Agentic OS settings" onClose={onClose}>
  <p>Saving writes <code>config/agentic_os.yaml</code> under a new version. The change is checked, recorded with your name and can be undone from Configuration. The next run uses it.</p>
  {busy&&!review&&!error&&<p className="muted" role="status"><CircleNotch size={14} className="spin" aria-hidden="true"/> Checking the change…</p>}
  {review&&<>
   <h3 className="aos-modal-head">{changes.length} setting{changes.length===1?'':'s'} will change</h3>
   <DataTable columns={[
    {label:'Setting',render:c=><code className="aos-code">{c.path}</code>},
    {label:'Now',render:c=><span className="muted">{show(c.before)}</span>},
    {label:'After saving',render:c=><b>{show(c.after)}</b>},
   ]} rows={changes.map(c=>({...c,id:c.path}))}/>
   <div className="aos-version-row">
    <Setting label="Current version"><input value={baseVersion||''} readOnly/></Setting>
    <Setting label="New version" hint="Every change needs a version that has not been used before.">
     <input value={version} onChange={e=>setVersion(e.target.value)} maxLength={64}/></Setting>
   </div>
  </>}
  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}
  <div className="modal-actions">
   <Button onClick={onClose} disabled={busy&&!!review}>Cancel</Button>
   {review&&<Button primary busy={busy} disabled={!version.trim()||version.trim()===baseVersion} onClick={save}><FloppyDisk size={17} aria-hidden="true"/> Save settings</Button>}
  </div>
 </Modal>;
}

// ------------------------------------------------------------------ endpoint
function Endpoint({models,cards}){
 const [state,setState]=useState(null);const [stateError,setStateError]=useState('');const [checking,setChecking]=useState(false);
 const [model,setModel]=useState('stage2');const [row,setRow]=useState(()=>JSON.stringify(SAMPLE_ROW,null,2));const [rate,setRate]=useState('');
 const [result,setResult]=useState(null);const [error,setError]=useState('');const [sending,setSending]=useState(false);
 async function check(){setChecking(true);setStateError('');
  try{setState(await api('/agentic/serving'));}catch(e){setState(null);setStateError(e.message);}finally{setChecking(false);}}
 useEffect(()=>{check();},[]);
 async function send(e){e.preventDefault();setError('');setResult(null);
  let rows;try{const parsed=JSON.parse(row);rows=Array.isArray(parsed)?parsed:[parsed];}
  catch(x){setError(`The row is not valid JSON: ${x.message}`);return;}
  if(!rows.length||rows.some(r=>!r||typeof r!=='object'||Array.isArray(r))){setError('Enter one row as a JSON object, or a list of them.');return;}
  setSending(true);
  try{setResult(await api('/agentic/serving/invoke',{model,rows,...(rate===''?{}:{rate:Number(rate)})}));}
  catch(x){setError(x.message);}finally{setSending(false);}}
 const features=cards?.[model]?.features;
 return <div className="aos-endpoint">
  <div className="aos-section-head"><h3><CloudArrowUp size={18} aria-hidden="true"/> SageMaker endpoint</h3>
   <Button busy={checking} onClick={check}><ArrowClockwise size={16} aria-hidden="true"/> Check endpoint</Button></div>
  {stateError&&<InfoBox warning>The endpoint state is not available: {stateError} An endpoint appears after a run with SageMaker Serverless serving deploys it.</InfoBox>}
  {state&&<>{state.exists===false&&<InfoBox>No endpoint named <code>{state.endpoint_name}</code> is deployed yet. A run with SageMaker Serverless serving creates it.</InfoBox>}
   <Facts items={[['Endpoint',state.endpoint_name],['Status',state.status||(state.exists===false?'Not deployed':'—')],['Region',state.region],
    state.serverless&&['Capacity',`${number(state.serverless.MemorySizeInMB)} MB · up to ${state.serverless.MaxConcurrency} at once`],
    state.last_modified&&['Last changed',when(state.last_modified)]]}/>
   {state.failure_reason&&<InfoBox warning>Why it failed: {state.failure_reason}</InfoBox>}
   <KeyValues obj={state} skip={['mode','endpoint_name','status','region','serverless','last_modified','failure_reason','exists']}/></>}
  <form className="aos-test" onSubmit={send}>
   <h4 className="aos-subhead"><PaperPlaneTilt size={16} aria-hidden="true"/> Test the endpoint</h4>
   <div className="aos-fields">
    <Setting label="Model"><select value={model} onChange={e=>setModel(e.target.value)}>
     {models.map(([k,n])=><option key={k} value={k}>{n} ({k})</option>)}</select></Setting>
    <Setting label="Rate (%), optional" hint="Sent as rate. Leave empty if the model does not take one.">
     <input type="number" step="0.01" min="0" value={rate} onChange={e=>setRate(e.target.value)}/></Setting>
   </div>
   <Setting label="Row to score (JSON)" hint={features?`This model reads: ${features.join(', ')}.`:'One loan as a JSON object. Edit the values to try another case.'}>
    <textarea className="aos-json-input" value={row} spellCheck={false} rows={10} onChange={e=>setRow(e.target.value)}/></Setting>
   {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}
   <div className="actions"><Button primary type="submit" busy={sending}><PaperPlaneTilt size={16} aria-hidden="true"/> Send to endpoint</Button></div>
   {result&&<div className="aos-test-result" role="status">
    <Facts items={[['Model',result.model],['Endpoint',result.endpoint],['Latency',result.latency_ms==null?'—':`${fmt(result.latency_ms)} ms`]]}/>
    <pre className="aos-json">{JSON.stringify(result.predictions,null,1)}</pre></div>}
  </form>
 </div>;
}

// ------------------------------------------------------------------ steps
function DataStep({c,update,errors}){
 const d=c.data||{};
 return <>
  <Options legend="Where the data comes from" name="aos-source" options={SOURCES} value={d.source} onChange={v=>update('data.source',v)}/>
  {d.source==='synthetic'&&<div className="aos-fields">
   <Setting label="Loans to generate" hint="The saved default is 20,000." error={errors['data.rows']}>
    <input type="number" min="1" step="1" inputMode="numeric" value={d.rows??''} onChange={e=>update('data.rows',numberValue(e))}/></Setting>
   <Setting label="Random seed" hint="The same seed builds the same book." error={errors['data.seed']}>
    <input type="number" min="0" step="1" inputMode="numeric" value={d.seed??''} onChange={e=>update('data.seed',numberValue(e))}/></Setting>
  </div>}
 </>;
}

function DataChecks({checks}){
 return <div className="aos-block">
  <h3 className="aos-subhead"><ListChecks size={17} aria-hidden="true"/> Every loan must pass these checks</h3>
  <p className="small-copy">If any loan breaks one, the run stops before training. The list is fixed in the file.</p>
  <ul className="aos-checklist">{(checks||[]).map(k=><li key={k}><CheckCircle size={15} aria-hidden="true"/> <span>{CHECK_TEXT[k]||titleCase(k)} <code className="aos-code">{k}</code></span></li>)}</ul>
 </div>;
}

// The check button is open to every operator; only the two settings are disabled for read-only roles.
function PolicyStep({c,update,readOnly,readiness,check,checking,checkError,onCheck}){
 const p=c.policy||{};const ready=readiness?.policy||{};
 const consistency=check?`${check.consistency?.passed}/${check.consistency?.checks}`:ready.consistency;
 const cases=check?`${check.sample_cases?.passed}/${check.sample_cases?.cases}`:ready.sample_cases;
 const ok=check?(check.consistency?.ok&&!(check.sample_cases?.failed||[]).length):ready.ok;
 const words=v=>{const [a,b]=String(v||'').split('/');return b?`${a} of ${b} pass`:'—';};
 return <>
  <div className="aos-block">
   <Facts items={[
    ['Policy',check?.policy_version||ready.version],['Status',titleCase(check?.status||ready.status)||'—'],
    ['Consistency checks',words(consistency)],['Sample cases',words(cases)],
    ['Result',ok?<span className="aos-ok"><CheckCircle size={14} weight="fill" aria-hidden="true"/> All pass</span>:<span className="aos-bad"><XCircle size={14} weight="fill" aria-hidden="true"/> {ready.error&&!check?'Could not check':'Some fail'}</span>],
   ]}/>
   {ready.error&&!check&&<InfoBox warning>The policy could not be checked: {ready.error}</InfoBox>}
   <div className="actions"><Button busy={checking} onClick={onCheck}><ShieldCheck size={17} aria-hidden="true"/> Run policy checks</Button>
    {check&&!checking&&<span className="small-copy" role="status">Checked just now.</span>}</div>
   {checkError&&<InfoBox warning><span role="alert">{checkError}</span></InfoBox>}
   {check&&(ok?<InfoBox success>All {check.consistency?.checks} consistency checks and all {check.sample_cases?.cases} sample cases pass.</InfoBox>:<PolicyDetails r={check}/>)}
  </div>
  <fieldset className="aos-options-group aos-settings" disabled={readOnly}>
   <legend>Stop the run when</legend>
   <Check checked={p.require_consistency_checks} onChange={v=>update('policy.require_consistency_checks',v)}
    hint="The policy file's own rules must agree with each other.">a consistency check fails</Check>
   <Check checked={p.require_sample_cases} onChange={v=>update('policy.require_sample_cases',v)}
    hint="Worked examples the engine must price exactly as the policy says.">a sample case disagrees with the engine</Check>
  </fieldset>
 </>;
}

function Decisions({check}){
 const list=Object.entries(check?.decisions||{});
 return <div className="aos-block">
  <h3 className="aos-subhead"><ListChecks size={17} aria-hidden="true"/> Decisions in this policy</h3>
  {list.length?<ol className="aos-decisions">{list.map(([k,v])=><li key={k}><b>{k}</b><span>{String(v)}</span></li>)}</ol>
   :<p className="small-copy">Run the policy checks to list the decisions.</p>}
 </div>;
}

function TrainingStep({c,update,errors,readiness}){
 const t=c.training||{},s=t.sagemaker||{},g=t.autogluon_assistant||{};
 const models=t.models||[];
 const toggle=(k,on)=>update('training.models',MODELS.map(([m])=>m).filter(m=>m===k?on:models.includes(m)));
 return <>
  <Options legend="Training engine" name="aos-engine" options={ENGINES} value={t.engine} onChange={v=>update('training.engine',v)}/>
  <div className="aos-chips" aria-label="What this server has">
   <span className="small-copy">On this server:</span>
   <Chip ok={readiness?.boto3?.ok}>boto3 {readiness?.boto3?.ok?'installed':'not installed'}</Chip>
   <Chip ok={readiness?.autogluon_assistant?.ok}>AutoGluon Assistant {readiness?.autogluon_assistant?.ok?'found':'not found'}</Chip>
   {readiness?.autogluon_assistant?.ok&&<Chip ok={readiness.autogluon_assistant.allowed}>AutoGluon Assistant {readiness.autogluon_assistant.allowed?'allowed':'not allowed on this host'}</Chip>}
   <Chip ok={readiness?.bash?.ok}>bash {readiness?.bash?.ok?'found':'not found'}</Chip>
  </div>

  <fieldset className="aos-options-group">
   <legend>Models to train</legend>
   {t.engine==='autogluon_assistant'
    ?<p className="small-copy">AutoGluon Assistant trains one model, chosen under Target below. This list is used by the other engines.</p>
    :<p className="small-copy">{models.length} of {MODELS.length} selected.</p>}
   <div className="aos-models">{MODELS.map(([k,n,text])=><Check key={k} checked={models.includes(k)} onChange={v=>toggle(k,v)} hint={text}>{n} <code className="aos-code">{k}</code></Check>)}</div>
   <div className="actions aos-inline-actions">
    <button type="button" className="text-button" onClick={()=>update('training.models',MODELS.map(([m])=>m))}>Select all</button>
    <button type="button" className="text-button" onClick={()=>update('training.models',[])}>Clear</button>
   </div>
   {errors['training.models']&&<p className="aos-error" role="alert"><WarningCircle size={13} weight="fill" aria-hidden="true"/> {errors['training.models']}</p>}
  </fieldset>

  <Options legend="Register the new models as" name="aos-register" options={REGISTER} value={t.register_as} onChange={v=>update('training.register_as',v)}/>

  {t.engine==='sagemaker'&&<fieldset className="aos-options-group">
   <legend>Amazon SageMaker</legend>
   <div className="aos-fields">
    <Setting label="Instance type" hint="The machine the training job runs on." error={errors['training.sagemaker.instance_type']}>
     <input list="aos-instance-types" value={s.instance_type??''} onChange={e=>update('training.sagemaker.instance_type',e.target.value.trim())}/></Setting>
    <Setting label="Maximum run time (minutes)" hint="The job is stopped after this." error={errors['training.sagemaker.max_runtime_minutes']}>
     <input type="number" min="1" step="1" value={s.max_runtime_minutes??''} onChange={e=>update('training.sagemaker.max_runtime_minutes',numberValue(e))}/></Setting>
    <Setting label="S3 bucket" hint="Where the data and model artefacts are written. Leave empty to use the server's AAVAS_SAGEMAKER_BUCKET." error={errors['training.sagemaker.bucket']}>
     <input value={s.bucket??''} onChange={e=>update('training.sagemaker.bucket',e.target.value.trim())}/></Setting>
    <Setting label="S3 prefix" hint="Folder inside the bucket." error={errors['training.sagemaker.prefix']}>
     <input value={s.prefix??''} onChange={e=>update('training.sagemaker.prefix',e.target.value)}/></Setting>
   </div>
   <datalist id="aos-instance-types">{INSTANCE_TYPES.map(x=><option key={x} value={x}/>)}</datalist>
   <ReadOnly items={[['Region',s.region],['Execution role',s.role_arn||'from the server (AAVAS_SAGEMAKER_ROLE_ARN)'],['Image',s.image]]}/>
   <p className="small-copy">The role and image are set in the file by an administrator. The same image trains and serves, so what is trained loads where it is served.</p>
  </fieldset>}

  {t.engine==='autogluon_assistant'&&<fieldset className="aos-options-group">
   <legend>AutoGluon Assistant</legend>
   <div className="aos-fields">
    <Setting label="Bedrock model ID" hint="The model the agent reasons with." error={errors['training.autogluon_assistant.model']}>
     <input value={g.model??''} onChange={e=>update('training.autogluon_assistant.model',e.target.value.trim())}/></Setting>
    <Setting label="Target" hint="Which model it trains a challenger for.">
     <select value={g.target??'stage2'} onChange={e=>update('training.autogluon_assistant.target',e.target.value)}>
      <option value="stage2">Final rate (stage2)</option><option value="conversion">Acceptance (conversion)</option></select></Setting>
    <Setting label="Attempts" hint="How many times the agent may revise its code." error={errors['training.autogluon_assistant.max_iterations']}>
     <input type="number" min="1" step="1" value={g.max_iterations??''} onChange={e=>update('training.autogluon_assistant.max_iterations',numberValue(e))}/></Setting>
    <Setting label="Time limit (minutes)" error={errors['training.autogluon_assistant.time_limit_minutes']}>
     <input type="number" min="1" step="1" value={g.time_limit_minutes??''} onChange={e=>update('training.autogluon_assistant.time_limit_minutes',numberValue(e))}/></Setting>
   </div>
   <Setting label="Instructions to the agent" hint="Plain words. Say what to predict and what it may use.">
    <textarea rows={4} value={g.instructions??''} onChange={e=>update('training.autogluon_assistant.instructions',e.target.value)}/></Setting>
   <ReadOnly items={[['Provider',g.provider],['Region',g.region],['Command line',g.cli]]}/>
  </fieldset>}
 </>;
}

function Champions({champions}){
 return <div className="aos-block">
  <h3 className="aos-subhead"><Trophy size={17} aria-hidden="true"/> Current champions</h3>
  <DataTable empty="No champion models are registered." columns={[
   {label:'Model',render:m=><><b>{MODEL_NAME[m.model]||titleCase(m.model)}</b> <code className="aos-code">{m.model}</code></>},
   {label:'Learner',render:m=>titleCase(m.learner)||'—'},
   {label:'Trained',render:m=><small>{when(m.trained_at)}</small>},
   {label:'Headline measures',render:m=><small>{Object.entries(m.metrics||{}).map(([k,v])=>`${METRIC_LABEL[k]||titleCase(k)} ${fmt(v)}`).join(' · ')||'—'}</small>},
  ]} rows={(champions||[]).map(m=>({...m,id:m.model}))}/>
 </div>;
}

function ServingStep({c,update,errors,onGo}){
 const v=c.serving||{},e=v.sagemaker_serverless||{};const engine=c.training?.engine;
 const conflict=v.mode==='sagemaker_serverless'&&engine!=='sagemaker';
 return <>
  <Options legend="How the result is served" name="aos-serving" options={SERVING} value={v.mode} onChange={x=>update('serving.mode',x)}/>
  {conflict
   ?<InfoBox warning><span role="alert">Serverless serving needs the SageMaker training engine. Training is set to {engineTitle(engine)}, so the run would stop at this step.</span>
     {' '}<button type="button" className="text-button" onClick={()=>onGo('training')}>Change the training engine <ArrowRight size={14} aria-hidden="true"/></button></InfoBox>
   :<p className="small-copy">Serverless serving needs the SageMaker training engine.</p>}
  <fieldset className="aos-options-group">
   <legend>Approval</legend>
   <Check checked={v.promote_on_success} onChange={x=>update('serving.promote_on_success',x)}
    hint="Off (recommended): an approver promotes the result from the Run step. Only local runs are promoted.">Promote automatically when every step passes</Check>
  </fieldset>
  {v.mode==='sagemaker_serverless'&&<fieldset className="aos-options-group">
   <legend>Serverless endpoint</legend>
   <div className="aos-fields">
    <Setting label="Endpoint name" error={errors['serving.sagemaker_serverless.endpoint_name']}>
     <input value={e.endpoint_name??''} onChange={x=>update('serving.sagemaker_serverless.endpoint_name',x.target.value.trim())}/></Setting>
    <Setting label="Memory" hint="More memory also gives more CPU." error={errors['serving.sagemaker_serverless.memory_mb']}>
     <select value={e.memory_mb??''} onChange={x=>update('serving.sagemaker_serverless.memory_mb',Number(x.target.value))}>
      {!MEMORY.includes(e.memory_mb)&&<option value={e.memory_mb??''}>{e.memory_mb??'—'} MB (not allowed)</option>}
      {MEMORY.map(m=><option key={m} value={m}>{number(m)} MB</option>)}</select></Setting>
    <Setting label="Maximum concurrent requests" hint="1 to 200." error={errors['serving.sagemaker_serverless.max_concurrency']}>
     <input type="number" min="1" max="200" step="1" value={e.max_concurrency??''} onChange={x=>update('serving.sagemaker_serverless.max_concurrency',numberValue(x))}/></Setting>
   </div>
   <ReadOnly items={[['Parity tolerance',e.parity_tolerance]]}/>
   <p className="small-copy">After deploying, the run checks the endpoint gives the same predictions training recorded, within this tolerance.</p>
  </fieldset>}
 </>;
}

function Summary({c,dirty,onGo}){
 const d=c.data||{},p=c.policy||{},t=c.training||{},v=c.serving||{};
 const yes=b=>b?'Yes':'No';
 const groups=[
  ['data','Data',[['Source',SOURCES.find(s=>s.value===d.source)?.title||d.source],...(d.source==='synthetic'?[['Loans',number(d.rows)],['Seed',d.seed]]:[]),['Checks',`${(d.checks||[]).length} must pass`]]],
  ['policy','Policy',[['Stop on a failed consistency check',yes(p.require_consistency_checks)],['Stop on a failed sample case',yes(p.require_sample_cases)]]],
  ['training','Training',[['Engine',engineTitle(t.engine)],
   t.engine==='autogluon_assistant'?['Target',t.autogluon_assistant?.target]:['Models',`${(t.models||[]).length}: ${(t.models||[]).join(', ')||'none'}`],
   ['Register as',titleCase(t.register_as)],
   ...(t.engine==='sagemaker'?[['Instance',t.sagemaker?.instance_type],['Time limit',`${t.sagemaker?.max_runtime_minutes} min`]]:[]),
   ...(t.engine==='autogluon_assistant'?[['Bedrock model',t.autogluon_assistant?.model],['Time limit',`${t.autogluon_assistant?.time_limit_minutes} min`]]:[])]],
  ['serving','Serving',[['Mode',servingTitle(v.mode)],['Promote automatically',yes(v.promote_on_success)],
   ...(v.mode==='sagemaker_serverless'?[['Endpoint',v.sagemaker_serverless?.endpoint_name],['Memory',`${number(v.sagemaker_serverless?.memory_mb)} MB`]]:[])]],
 ];
 return <div className="aos-summary">
  {groups.map(([key,title,rows])=><section key={key} className="aos-summary-card" aria-label={`${title} settings`}>
   <div className="aos-summary-head"><h4>{title}</h4><button type="button" className="text-button" onClick={()=>onGo(key)}>Change<span className="sr-only"> {title.toLowerCase()} settings</span></button></div>
   <dl>{rows.map(([k,val])=><div key={k}><dt>{k}</dt><dd>{val==null||val===''?'—':val}</dd></div>)}</dl>
  </section>)}
  {dirty&&<p className="small-copy aos-summary-note">Shown with your unsaved changes.</p>}
 </div>;
}

// ------------------------------------------------------------------ the page
export function AgenticOS({user,notify=()=>{}}){
 const canEdit=Boolean(user?.all_pages||user?.role==='administrator');
 const [overview,setOverview]=useState(null);const [error,setError]=useState('');const [loading,setLoading]=useState(false);
 const [draft,setDraft]=useState(null);const [step,setStep]=useState('data');
 const [saveOpen,setSaveOpen]=useState(false);
 const [check,setCheck]=useState(null);const [checking,setChecking]=useState(false);const [checkError,setCheckError]=useState('');
 const [runId,setRunId]=useState(null);const [run,setRun]=useState(null);const [runError,setRunError]=useState('');const [pollKey,setPollKey]=useState(0);
 const [starting,setStarting]=useState(false);const [startError,setStartError]=useState('');
 const [promoting,setPromoting]=useState(false);const [promoteError,setPromoteError]=useState('');
 const [resuming,setResuming]=useState(false);const [resumeError,setResumeError]=useState('');
 const runRef=useRef(null);const panelRef=useRef(null);

 async function load(resetDraft=false){setLoading(true);setError('');
  try{const o=await api('/agentic/overview');setOverview(o);
   setDraft(d=>!d||resetDraft?structuredClone(o.config):d);
   setRunId(id=>id||o.active_run||o.runs?.[0]?.run_id||null);}
  catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load(true);},[]);

 // Follow the open run: every 3 s while it is queued or running.
 useEffect(()=>{if(!runId){setRun(null);return;}
  let alive=true,timer=null,wasLive=false;
  async function tick(){
   try{const r=await api(`/agentic/runs/${encodeURIComponent(runId)}`);if(!alive)return;
    setRun(r);setRunError('');
    if(LIVE.has(r.status)){wasLive=true;timer=setTimeout(tick,3000);}
    else if(wasLive){wasLive=false;load();
     notify(r.status==='succeeded'?`Run ${r.run_id} finished: every step passed.`:`Run ${r.run_id} ${r.status}. See the step that stopped it.`,r.status==='succeeded'?'success':'error');}}
   catch(e){if(!alive)return;setRunError(e.message);if(wasLive)timer=setTimeout(tick,6000);}
  }
  setRun(r=>r?.run_id===runId?r:null);tick();
  return()=>{alive=false;clearTimeout(timer);};},[runId,pollKey]);

 async function runCheck(){setChecking(true);setCheckError('');
  try{setCheck(await api('/agentic/policy/check',{}));}catch(e){setCheckError(e.message);}finally{setChecking(false);}}
 // The decisions only come with a check, which takes about a second: run it on first visit.
 useEffect(()=>{if(step==='policy'&&!check&&!checking&&!checkError&&overview)runCheck();},[step,overview]);

 const base=overview?.config;
 const edits=useMemo(()=>base&&draft?EDITABLE.filter(p=>!same(getIn(draft,p),getIn(base,p))):[],[base,draft]);
 const dirty=canEdit&&edits.length>0;
 const problems=useMemo(()=>draft?validate(draft):[],[draft]);
 const errors=useMemo(()=>Object.fromEntries(problems.map(p=>[p.path,p.message])),[problems]);
 const savedProblems=useMemo(()=>base?validate(base):[],[base]);
 const notes=useMemo(()=>cautions(draft,overview?.readiness),[draft,overview]);
 const update=(path,value)=>{if(canEdit)setDraft(d=>setIn(d,path.split('.'),value));};

 function go(key){setStep(key);requestAnimationFrame(()=>panelRef.current?.scrollIntoView({behavior:'smooth',block:'start'}));}
 function openRun(id){setRunId(id);setPollKey(k=>k+1);setStep('run');requestAnimationFrame(()=>runRef.current?.scrollIntoView({behavior:'smooth',block:'start'}));}

 async function startRun(){setStarting(true);setStartError('');
  try{const r=await api('/agentic/runs',{});notify(`Run ${r.run_id} started.`);openRun(r.run_id);setRun(r);load();}
  catch(e){setStartError(e.status===409?`Not started: ${e.message}`:e.message);load();}
  finally{setStarting(false);}}

 async function promote(){if(!run)return;setPromoting(true);setPromoteError('');
  try{const r=await api(`/agentic/runs/${encodeURIComponent(run.run_id)}/promote`,{});
   setRun(x=>({...r,log:x?.log}));notify(`Promoted the models from run ${r.run_id} to champion.`);load();setPollKey(k=>k+1);}
  catch(e){setPromoteError(e.message);}finally{setPromoting(false);}}

 async function resume(){if(!run)return;setResuming(true);setResumeError('');
  try{const r=await api(`/agentic/runs/${encodeURIComponent(run.run_id)}/resume`,{});
   setRun(x=>({...r,log:x?.log}));notify(`Run ${r.run_id} resumed from the step that stopped it.`);load();setPollKey(k=>k+1);}
  catch(e){setResumeError(e.message);}finally{setResuming(false);}}

 function onSaved(result){setSaveOpen(false);notify(`Saved. The Agentic OS settings are now version ${result.after_version}.`);load(true);}

 if(error&&!overview)return <div className="operations aos"><InfoBox warning><span role="alert">The Agentic OS is not available: {error}</span></InfoBox>
  <div className="actions"><Button busy={loading} onClick={()=>load(true)}><ArrowClockwise aria-hidden="true"/> Try again</Button></div></div>;
 if(!overview||!draft)return <div className="operations aos"><Empty icon={FlowArrow} title="Loading the Agentic OS">Reading the settings, readiness and recent runs…</Empty></div>;

 const ready=overview.readiness||{};
 const policyOk=check?(check.consistency?.ok&&!(check.sample_cases?.failed||[]).length):ready.policy?.ok;
 const activeRun=overview.active_run;
 const runLive=run&&LIVE.has(run.status);
 const stepProblems=k=>problems.filter(p=>p.step===k).length;
 const runNote=()=>{if(!run)return runId?['busy','Loading the latest run']:['idle','Not started yet'];
  if(LIVE.has(run.status)){const i=run.steps.findIndex(s=>s.status==='running');return ['busy',i>=0?`Running: ${STEP_TITLE[run.steps[i].name]}`:'Queued'];}
  if(run.status==='failed'){const f=run.steps.find(s=>s.status==='failed');return ['error',`Failed${f?` at ${STEP_TITLE[f.name]}`:''}`];}
  if(run.status==='refused')return ['error','Refused'];
  const waiting=run.steps.find(s=>s.name==='serving')?.details?.awaiting_promotion&&!run.promotion;
  return waiting?['warn','Ready to promote']:['ok',run.promotion?'Promoted':'Succeeded'];};
 const stepper=[
  {key:'data',...(stepProblems('data')?{state:'error',note:`${stepProblems('data')} to fix`}:{state:'ok',note:draft.data?.source==='synthetic'?`Synthetic · ${number(draft.data?.rows)} loans`:'Live dataset'})},
  {key:'policy',state:ready.policy?.error&&!check?'error':policyOk?'ok':'error',note:ready.policy?.error&&!check?'Could not check':policyOk?'Checks pass':'Checks fail'},
  {key:'training',...(stepProblems('training')?{state:'error',note:`${stepProblems('training')} to fix`}
   :draft.training?.engine==='sagemaker'&&!ready.boto3?.ok?{state:'warn',note:'SageMaker · boto3 missing'}
   :draft.training?.engine==='autogluon_assistant'&&!ready.autogluon_assistant?.ok?{state:'warn',note:'AutoGluon · not installed'}
   :{state:'ok',note:engineTitle(draft.training?.engine)})},
  {key:'serving',...(stepProblems('serving')?{state:'error',note:draft.serving?.mode==='sagemaker_serverless'&&draft.training?.engine!=='sagemaker'?'Needs SageMaker training':`${stepProblems('serving')} to fix`}
   :{state:'ok',note:servingTitle(draft.serving?.mode)})},
  {key:'run',...(([state,note])=>({state,note}))(runNote())},
 ].map(s=>({...s,...STEPS.find(x=>x.key===s.key)}));
 const index=STEPS.findIndex(s=>s.key===step);const meta=overview.steps?.find(s=>s.name===step);
 const startBlocked=dirty?'You have unsaved changes. Save them, or discard them, before starting: a run uses the saved settings.'
  :activeRun?`Run ${activeRun} is still going. One run happens at a time.`
  :savedProblems.length?`The saved settings need fixing first: ${savedProblems[0].message}`:'';
 const stateIcon={ok:CheckCircle,warn:WarningCircle,error:XCircle,busy:CircleNotch,idle:Circle};

 return <div className="operations aos">
  <div className="page-intro"><h1>Agentic OS</h1>
   <p>Rebuild the pricing models as a chain of agents: check the data, check the policy, train, then serve.
    Each agent checks its work, and a failed check stops the run with the reason.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  <div className="model-metrics aos-metrics">
   <div><small>Settings version</small><b>{overview.version||'—'}</b></div>
   <div><small>Training engine</small><b>{engineTitle(base.training?.engine)}</b></div>
   <div><small>Serving</small><b>{servingTitle(base.serving?.mode)}</b></div>
   <div><small>Policy</small><b>{ready.policy?.version||'—'} · {policyOk?'passes':'fails'}</b></div>
   <div><small>Champion models</small><b>{overview.champions?.length??0}</b></div>
   <div><small>Run in progress</small><b>{activeRun?'yes':'no'}</b></div>
  </div>

  <div className="actions end"><Button busy={loading} onClick={()=>load()}><ArrowClockwise aria-hidden="true"/> Refresh</Button></div>

  {!canEdit&&<InfoBox><Lock size={15} aria-hidden="true"/> You can run the policy checks, start a run and promote its result. Only an administrator can change these settings.</InfoBox>}

  <nav className="aos-stepper-nav" aria-label="Workflow steps">
   <ol className="aos-stepper">{stepper.map((s,i)=>{const Icon=stateIcon[s.state]||Circle;return <li key={s.key} className={`aos-stepper-item ${s.state}${s.key===step?' current':''}`}>
    <button type="button" onClick={()=>go(s.key)} aria-current={s.key===step?'step':undefined}>
     <span className="aos-step-n" aria-hidden="true">{i+1}</span>
     <span className="aos-step-text"><b>{s.title}</b>
      <small><Icon size={13} weight={s.state==='busy'||s.state==='idle'?'bold':'fill'} className={s.state==='busy'?'spin':''} aria-hidden="true"/> {s.note}</small></span>
    </button></li>;})}</ol>
  </nav>

  {dirty&&<div className="aos-dirty" role="status">
   <span><b>{edits.length} unsaved change{edits.length===1?'':'s'}.</b> A run uses the saved settings.</span>
   <div className="actions"><Button onClick={()=>setDraft(structuredClone(base))}>Discard</Button>
    <Button primary disabled={problems.length>0} onClick={()=>setSaveOpen(true)}><FloppyDisk size={16} aria-hidden="true"/> Review and save</Button></div>
   {problems.length>0&&<small className="aos-error">Fix {problems.length} setting{problems.length===1?'':'s'} first: {problems.map(p=>STEP_TITLE[p.step]).filter((x,i,a)=>a.indexOf(x)===i).join(', ')}.</small>}
  </div>}

  <div ref={panelRef} className="aos-step-anchor"/>
  <Panel className="aos-step" title={<>Step {index+1} of {STEPS.length}: {STEPS[index].title}</>} icon={STEPS[index].icon}>
   {step!=='run'&&meta?.description&&<p className="aos-lead">{meta.description}</p>}
   {step==='run'&&<p className="aos-lead">Check the settings, then start. The four agents run in order; you can follow each one below.</p>}

   {step==='data'&&<><fieldset className="aos-settings" disabled={!canEdit}><DataStep c={draft} update={update} errors={errors}/></fieldset>
    <DataChecks checks={draft.data?.checks}/></>}

   {step==='policy'&&<><PolicyStep c={draft} update={update} readOnly={!canEdit} readiness={ready} check={check} checking={checking} checkError={checkError} onCheck={runCheck}/>
    <Decisions check={check}/></>}

   {step==='training'&&<><fieldset className="aos-settings" disabled={!canEdit}><TrainingStep c={draft} update={update} errors={errors} readiness={ready}/></fieldset>
    <Champions champions={overview.champions}/></>}

   {step==='serving'&&<><fieldset className="aos-settings" disabled={!canEdit}><ServingStep c={draft} update={update} errors={errors} onGo={go}/></fieldset>
    {draft.serving?.mode==='sagemaker_serverless'||base.serving?.mode==='sagemaker_serverless'
     ?<Endpoint models={MODELS} cards={run?.steps?.find(s=>s.name==='training')?.details?.cards}/>
     :<p className="small-copy aos-block">Choose SageMaker Serverless to deploy an endpoint and test it here.</p>}</>}

   {step==='run'&&<>
    <Summary c={draft} dirty={dirty} onGo={go}/>
    {notes.length>0&&<InfoBox warning><ul className="aos-notes">{notes.map(n=><li key={n}>{n}</li>)}</ul></InfoBox>}
    <div className="aos-run-actions">
     {canEdit&&<Button disabled={!dirty||problems.length>0} onClick={()=>setSaveOpen(true)}><FloppyDisk size={17} aria-hidden="true"/> Save settings</Button>}
     <Button primary busy={starting} disabled={!!startBlocked||runLive} onClick={startRun}><Play size={17} weight="fill" aria-hidden="true"/> Start run</Button>
     <span className="small-copy">{startBlocked||(runLive?'The open run is still going.':`Runs with the saved settings, version ${overview.version}.`)}</span>
    </div>
    {startError&&<InfoBox warning><span role="alert">{startError}</span>
     {activeRun&&activeRun!==runId&&<> <button type="button" className="text-button" onClick={()=>openRun(activeRun)}>Open the run in progress <ArrowRight size={14} aria-hidden="true"/></button></>}</InfoBox>}
   </>}

   <div className="aos-step-nav">
    {index>0?<Button onClick={()=>go(STEPS[index-1].key)}><ArrowLeft size={16} aria-hidden="true"/> Back: {STEPS[index-1].title}</Button>:<span/>}
    {index<STEPS.length-1&&<Button primary onClick={()=>go(STEPS[index+1].key)}>Next: {STEPS[index+1].title} <ArrowRight size={16} aria-hidden="true"/></Button>}
   </div>
  </Panel>

  {step==='run'&&<>
   <div ref={runRef} className="aos-step-anchor"/>
   {run?<RunView run={run} error={runError} champions={overview.champions?.length} onPromote={promote} promoting={promoting} promoteError={promoteError} onResume={resume} resuming={resuming} resumeError={resumeError}/>
    :runId?<Panel title="Run"><p className="muted" role="status"><CircleNotch size={14} className="spin" aria-hidden="true"/> Loading run {runId}…</p>
      {runError&&<InfoBox warning><span role="alert">{runError}</span></InfoBox>}</Panel>
    :<Panel title="Run" icon={Rocket}><p className="small-copy">No runs yet. Start one above.</p></Panel>}
   <Panel title="Recent runs" icon={ClockCounterClockwise}>
    <DataTable empty="No runs yet." columns={[
     {label:'Run',render:r=><button type="button" className={`link-button${r.run_id===runId?' active':''}`} onClick={()=>openRun(r.run_id)}
       aria-label={`Open run ${r.run_id}`} aria-current={r.run_id===runId?'true':undefined}>{r.run_id}</button>},
     {label:'Status',render:r=><span className="aos-status-cell"><StatusBadge status={r.status}/>{r.promotion&&<Badge tone="green"><Trophy size={12} aria-hidden="true"/> Promoted</Badge>}</span>},
     {label:'Started by',key:'started_by'},
     {label:'Started',render:r=><small>{when(r.started_at||r.created_at)}</small>},
     {label:'Took',render:r=>r.status==='refused'?'—':took(r.started_at||r.created_at,r.finished_at)},
     {label:'Training',render:r=>engineTitle(r.engine)},
     {label:'Serving',render:r=>servingTitle(r.serving_mode)},
     {label:'Settings',render:r=><code className="aos-code">{r.config_version||'—'}</code>},
    ]} rows={(overview.runs||[]).map(r=>({...r,id:r.run_id}))}/>
   </Panel>
  </>}

  {saveOpen&&<SaveModal baseVersion={overview.version} edits={edits} draft={draft} onClose={()=>setSaveOpen(false)} onSaved={onSaved}/>}
 </div>;
}
