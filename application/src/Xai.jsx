import React,{useEffect,useState} from 'react';
import {Brain,ChartBar,Eye,WarningCircle,ArrowClockwise,ImageSquare,Info,Gauge,ArrowSquareOut} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable} from './components';

// Trust tone for the rank-agreement number. Two methods that share no machinery
// ranking features the same way is what makes a ranking worth acting on; when
// they disagree, the ranking is a property of the method, not of the model.
const trustTone=l=>l==='strong'?'green':l==='moderate'?'amber':'red';
const pct=n=>n==null?'—':`${Number(n).toFixed(1)}%`;
const num=(n,d=3)=>n==null?'—':Number(n).toFixed(d);
const score=n=>n==null?'—':Number(n).toFixed(0);
// A prediction is a probability for the classifiers and % p.a. for the two rate
// models, so the band renders whatever the report says its units are.
const out=(n,units)=>n==null?'—':units==='probability'?Number(n).toFixed(4):Number(n).toFixed(3);

// A contribution is what the method attributed; the value is what the file held.
// They are different numbers and the value is a label for a categorical, so the
// column that reads as an attribution has to carry the contribution.
const contribution=n=>n==null?'—':`${n>0?'+':''}${Number(n).toFixed(4)}`;
const featureValue=v=>v==null?'—':typeof v==='number'?Number(v).toLocaleString('en-IN',{maximumFractionDigits:2}):String(v);
const hasCf=cf=>typeof cf==='string'?cf.length>0:Array.isArray(cf)?cf.length>0:!!cf;

function Attributions({rows}){
 if(!rows?.length)return <p className="small-copy">Not computed for this file.</p>;
 return <ol className="attribution-list">{rows.map((r,j)=><li key={j}>
   <span>{r.label||r.feature}<br/><small className="muted">{featureValue(r.value)}</small></span>
   <b>{contribution(r.contribution??r.attribution)}</b></li>)}</ol>;
}

// One explained file. Shared by the per-instance tab and the credit-score ends,
// so a reviewer reads the same card wherever they arrive from.
function InstanceCard({inst,units,title,attributionUnits}){
 return <Panel title={title||`File ${inst.loan_id}`} icon={Eye}
   action={<Badge tone={inst.top3_agreement>=2?'green':'red'}>{inst.top3_agreement}/3 agreement</Badge>}>
  <div className="model-metrics">
   <div><small>Prediction</small><b>{typeof inst.prediction==='number'?out(inst.prediction,units):inst.prediction}</b></div>
   <div><small>Credit score</small><b>{inst.credit_score==null?'—':inst.credit_score<0?'new to credit':score(inst.credit_score)}</b></div>
   <div><small>Actual</small><b>{inst.actual ?? '—'}</b></div>
   <div><small>Explained class</small><b>{inst.explained_class||'—'}</b></div>
  </div>
  <div className="xai-two-col">
   <div><h4>KernelSHAP</h4><Attributions rows={inst.shap}/></div>
   <div><h4>LIME</h4><Attributions rows={inst.lime}/></div>
  </div>
  {attributionUnits&&<p className="small-copy">Contributions are in <b>{attributionUnits}</b>
   {attributionUnits==='log-odds'&&' — a classifier is explained through the logit link, so these do not add up to the probability above'}. The grey line under each feature is the value the file carried.</p>}
  {hasCf(inst.counterfactuals)&&<p className="small-copy"><Info size={14}/> Counterfactual: {typeof inst.counterfactuals==='string'?inst.counterfactuals:JSON.stringify(inst.counterfactuals).slice(0,240)}</p>}
 </Panel>;
}

export function Xai({health,navigate}){
 const [index,setIndex]=useState(null);const [model,setModel]=useState(null);const [detail,setDetail]=useState(null);
 const [report,setReport]=useState('');const [tab,setTab]=useState('global');const [loading,setLoading]=useState(false);const [error,setError]=useState('');
 // The dashboards are separate servers with their own lifecycle, so their state
 // is asked for separately and a failure here must not take the page down.
 const [live,setLive]=useState([]);

 async function loadIndex(){setLoading(true);setError('');
  try{const r=await api('/xai');setIndex(r);if(r.models?.length&&!model)setModel(r.models[0].model);}
  catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{loadIndex();api('/tracking/services').then(r=>setLive(r.services||[])).catch(()=>{});},[]);

 useEffect(()=>{if(!model)return;let live=true;setLoading(true);setReport('');
  api(`/xai/${model}`).then(d=>{if(live){setDetail(d);setError('');}}).catch(e=>{if(live)setError(e.message);}).finally(()=>{if(live)setLoading(false);});
  fetch(`/api/xai/${model}/report`).then(r=>r.ok?r.text():'').then(t=>{if(live)setReport(t);}).catch(()=>{});
  return()=>{live=false;};},[model]);

 const summary=index?.models?.find(m=>m.model===model);
 const band=detail?.credit_score;const units=detail?.units;
 const liveDash=live.find(s=>s.kind==='omnixai'&&s.model===model&&s.reachable);

 return <div className="operations">
  <div className="page-intro"><h1>Model Explainability</h1>
   <p>What each model does across the whole book, and whether a second method agrees. Offline OmniXAI pass — separate from the reason codes shown on a single decision.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}
  {index&&!index.available&&<Empty icon={WarningCircle} title="No explainability reports found">Generate them with <code>make xai-install &amp;&amp; make xai-reports</code>, then refresh.</Empty>}

  {index?.available&&<>
   <InfoBox>These reports explain the model across a holdout sample. They are regenerated offline, not per request — the age of each report is shown below.</InfoBox>

   <div className="actions end"><Button busy={loading} onClick={loadIndex}><ArrowClockwise/> Refresh Reports</Button></div>

   {/* ------------------------------------------------ model picker + trust */}
   <Panel title="Models with a report" icon={Brain}>
    <DataTable columns={[
      {label:'Model',render:r=><button className={`link-button${r.model===model?' active':''}`} onClick={()=>{setModel(r.model);setTab('global');}}>{titleCase(r.model)}</button>},
      {label:'Version',render:r=><Badge tone="green">{r.model_version}</Badge>},
      {label:'Learner',render:r=>titleCase(r.learner)},
      {label:'Rank agreement',render:r=><span><b>{num(r.rank_agreement_spearman,2)}</b> <Badge tone={trustTone(r.trust.level)}>{r.trust.level}</Badge></span>},
      {label:'How to read it',render:r=><small>{r.trust.reading}</small>},
      {label:'Report age',render:r=><small>{r.age_days==null?'—':`${r.age_days} days`}</small>},
    ]} rows={index.models}/>
    <p className="small-copy">Rank agreement is Spearman correlation between KernelSHAP and permutation importance. It exists to say when <em>not</em> to trust the ranking.</p>
   </Panel>

   {detail&&<>
    {/* ------------------------------------------------------------ header */}
    <div className="model-metrics xai-header">
     <div><small>Target</small><b>{titleCase(detail.target)}</b></div>
     <div><small>Output</small><b>{detail.output} {detail.units}</b></div>
     <div><small>Mode</small><b>{titleCase(detail.mode)}</b></div>
     <div><small>Explained rows</small><b>{number(detail.counts?.n_global_sample)}</b></div>
     <div><small>Holdout</small><b>{number(detail.counts?.n_test)}</b></div>
     <div><small>Attribution units</small><b>{detail.attribution_units||'—'}</b></div>
    </div>

    {summary&&summary.trust.level!=='strong'&&<InfoBox warning>
      Rank agreement on <b>{titleCase(model)}</b> is {num(summary.rank_agreement_spearman,2)} ({summary.trust.level}).
      {' '}The two methods disagree about what matters, so {summary.trust.reading}.
    </InfoBox>}

    <div className="tabs" role="tablist">
     {[['global','Global importance'],['credit','Credit score'],['instances','Per-instance'],['dashboard','OmniXAI dashboard'],['limitations','Limitations'],['brief','Written brief']].map(([k,label])=>
      <button key={k} role="tab" aria-selected={tab===k} className={tab===k?'active':''} onClick={()=>setTab(k)}>{label}</button>)}
    </div>

    {/* ---------------------------------------------------- global importance */}
    {tab==='global'&&<Panel title={`What moves ${titleCase(model)}`} icon={ChartBar}>
     <DataTable columns={[
       {label:'#',key:'rank'},
       {label:'Feature',render:r=><span><b>{r.label||titleCase(r.feature)}</b><br/><small className="muted">{r.feature}</small></span>},
       {label:'SHAP share',render:r=><span className="bar-cell"><span className="bar" style={{width:`${Math.min(r.shap_share_pct,100)}%`}}/><span>{pct(r.shap_share_pct)}</span></span>},
       {label:'Permutation',render:r=>num(r.permutation)},
       {label:'PDP direction',render:r=>titleCase(r.pdp_direction||'—')},
       {label:'Monotone',render:r=>r.monotone_in_pdp?<Badge tone="green">yes</Badge>:<Badge>no</Badge>},
     ]} rows={detail.global_importance||[]}/>
     <p className="small-copy">SHAP share is each feature&apos;s portion of total attribution. Permutation importance is computed independently — the two agreeing is what the rank-agreement number measures. &quot;Varies by category&quot; is reported for unordered categoricals, where a direction would be meaningless.</p>
    </Panel>}

    {/* ------------------------------------------------------- credit score */}
    {tab==='credit'&&(band?<>
     <InfoBox>{band.reading}</InfoBox>

     <Panel title={`${band.label} — what the book looks like at each end`} icon={Gauge}>
      <div className="model-metrics">
       <div><small>Scored files</small><b>{number(band.observed.scored_files)}</b></div>
       <div><small>Score range</small><b>{score(band.observed.score_min)}–{score(band.observed.score_max)}</b></div>
       <div><small>Bottom {band.tail_pct}%</small><b>{out(band.observed.floor_cohort.mean_prediction,units)}</b></div>
       <div><small>Top {band.tail_pct}%</small><b>{out(band.observed.ceiling_cohort.mean_prediction,units)}</b></div>
       <div><small>Gap</small><b>{out(band.observed.gap==null?null:Math.abs(band.observed.gap),units)}</b></div>
       <div><small>New to credit</small><b>{number(band.observed.new_to_credit.files)} ({pct(band.observed.new_to_credit_pct)})</b></div>
      </div>
      <DataTable columns={[
        {label:'Cohort',render:r=><b>{r.title}</b>},
        {label:'Files',render:r=>number(r.files)},
        {label:'Score range',render:r=>r.score_min==null?'—':r.score_min<0?`sentinel ${band.sentinel}`:`${score(r.score_min)}–${score(r.score_max)}`},
        {label:`Mean ${band.output.toLowerCase()}`,render:r=><b>{out(r.mean_prediction,units)}</b>},
        {label:'Median',render:r=>out(r.median_prediction,units)},
      ]} rows={[
        {...band.observed.floor_cohort,title:`Bottom ${band.tail_pct}% of the score`},
        {...band.observed.ceiling_cohort,title:`Top ${band.tail_pct}% of the score`},
        {...band.observed.new_to_credit,title:'New to credit (no score)'},
      ].filter(r=>r.files)}/>
      <p className="small-copy">These are the files as they actually arrived, so the gap carries everything
       that travels with a credit score — income, LTV, bureau vintage — not the score alone. New-to-credit files
       hold the <code>{band.sentinel}</code> sentinel rather than a low score, and are shown for contrast, not as
       the bottom of the band. Valid scores run {band.valid_range[0]}–{band.valid_range[1]}.</p>
     </Panel>

     {band.curve?<Panel title="Holding every other feature fixed" icon={ChartBar}
       action={<Badge tone={band.curve.monotone?'green':'red'}>{band.curve.monotone?'monotone':'not monotone'}</Badge>}>
      <DataTable columns={[
        {label:'Credit score',render:r=><b>{score(r.value)}</b>},
        {label:band.output,render:r=><span className="bar-cell">
          <span className="bar" style={{width:`${r.width}%`}}/><span>{out(r.score,units)}</span></span>},
        {label:'vs the bottom of the band',render:r=>r.delta===0?'—':<span>{r.delta>0?'+':''}{out(r.delta,units)}</span>},
      ]} rows={(()=>{const pts=band.curve.points||[];const vals=pts.map(q=>q.score);
        const lo=Math.min(...vals),hi=Math.max(...vals),span=hi-lo||1;const base=pts[0]?.score??0;
        return pts.map(q=>({...q,width:Math.max(4,((q.score-lo)/span)*100),delta:q.score-base}));})()}/>
      <p className="small-copy">Partial dependence over scored files only: every other feature is held where it is
       and the score alone is moved, so this is the model&apos;s response to the score rather than to the population
       that carries it. The rows are the PDP grid, so the ends are the nearest grid points to the observed range
       rather than the lowest and highest scores in the book. {band.curve.sentinel_points_dropped?.length>0&&<>The raw curve&apos;s low end sits on the
       new-to-credit sentinel (<code>{band.curve.sentinel_points_dropped.join(', ')}</code>) and is excluded here —
       reading it as &quot;worst credit&quot; would be reading the thin-file cohort instead. </>}
       {!band.curve.monotone&&<b>This curve turns at {score(band.curve.peak.value)}: the largest move against the
       overall trend is {out(band.curve.max_reversal,units)}, {pct(band.curve.max_reversal_share*100)} of the swing.
       A reversal worth a fraction of a percent of the swing is grid noise; a large one is a policy question —
       a better credit score should not move the model backwards.</b>}</p>
     </Panel>:<Empty icon={ChartBar} title="No curve for the credit score">
      The PDP pass did not reach this feature in this report. Raise <code>--pdp-features</code> and rerun
      <code> make xai-reports</code>.</Empty>}

     {(()=>{const ends=['lowest credit score','highest credit score']
        .map(k=>({key:k,meta:(band.instances||{})[k]})).filter(e=>e.meta);
      if(!ends.length)return null;
      return <><InfoBox>The two files at the ends of the score, explained twice each. The lowest-scoring file is
       not always the model&apos;s highest prediction — where the two diverge, something other than the score is
       driving the decision.</InfoBox>
       {ends.map(e=>{const full=(detail.instances||[]).find(i=>i.loan_id===e.meta.loan_id);
        return full?<InstanceCard key={e.key} inst={full} units={units} attributionUnits={detail.attribution_units}
          title={`${titleCase(e.key)} — ${e.meta.loan_id} (score ${score(e.meta.credit_score)})`}/>
         :<Panel key={e.key} title={titleCase(e.key)} icon={Eye}>
           <p className="small-copy">{e.meta.loan_id} · score {score(e.meta.credit_score)} · prediction {out(e.meta.prediction,units)}.
            Raise the <code>instances</code> parameter to see its attributions.</p></Panel>;})}</>;})()}
    </>:<Empty icon={Gauge} title="No credit-score band in this report">
      This report predates the credit-score band, or the model&apos;s contract carries no credit score.
      Regenerate with <code>make xai-reports</code>.</Empty>)}

    {/* ------------------------------------------------------- per instance */}
    {tab==='instances'&&<>
     <InfoBox>Each row is explained twice, by two independent methods. Top-3 agreement is how many of the three strongest drivers SHAP and LIME share.</InfoBox>
     {(detail.instances||[]).map((inst,i)=><InstanceCard key={i} inst={inst} units={units} attributionUnits={detail.attribution_units}
       title={`${titleCase(inst.label||'file')} — ${inst.loan_id||i+1}`}/>)}
     {detail.counterfactual_status&&<InfoBox><b>Counterfactuals:</b> {detail.counterfactual_status}</InfoBox>}
    </>}

    {/* ---------------------------------------------------- omnixai captures */}
    {tab==='dashboard'&&<>
     {liveDash
       ? <InfoBox success>The interactive dashboard for <b>{titleCase(model)}</b> is running at
          {' '}<code>{liveDash.url}</code>.{navigate&&<> Open it on
          {' '}<button className="link-button" onClick={()=>navigate('xai-dash')}>XAI Dashboard</button>,</>}
          {' '}or <a href={liveDash.url} target="_blank" rel="noreferrer">in a new tab <ArrowSquareOut size={13}/></a>.
          The captures below are the record of the same views that survives the process being stopped.</InfoBox>
       : <InfoBox>No dashboard process is running for {titleCase(model)}. Start one with <code>make xai</code>
          {navigate&&<> and it appears on <button className="link-button" onClick={()=>navigate('xai-dash')}>XAI Dashboard</button></>}.
          Everything on this page is offline and needs no process.</InfoBox>}
     {detail.captures?.length
      ? <>{detail.captures.map(c=><Panel key={c.view} title={c.label} icon={ImageSquare}>
          <a href={`/api${c.url}`} target="_blank" rel="noreferrer"><img className="xai-capture" src={`/api${c.url}`} alt={`OmniXAI ${c.label} for ${model}`} loading="lazy"/></a>
        </Panel>)}</>
      : <Empty icon={ImageSquare} title="No dashboard captures for this model">Captures exist only for models analysed with the UI pass. Everything else on this page — global importance, the credit-score band, per-instance attributions — is available for {titleCase(model)} without them.</Empty>}
    </>}

    {/* -------------------------------------------------------- limitations */}
    {tab==='limitations'&&<Panel title="Written limitations" icon={WarningCircle}>
     <ul className="limitation-list">{(detail.limitations||[]).map((l,i)=><li key={i}>{l}</li>)}</ul>
     {detail.notes?.length>0&&<><h4>Notes</h4><ul className="limitation-list">{detail.notes.map((n,i)=><li key={i}>{n}</li>)}</ul></>}
     <p className="small-copy">Carried from the model card. These bound how everything above may be read.</p>
    </Panel>}

    {/* ------------------------------------------------------- written brief */}
    {tab==='brief'&&<Panel title={`${titleCase(model)} — generated brief`} icon={Brain}>
     {report?<pre className="xai-brief">{report}</pre>:<Empty icon={WarningCircle} title="No written brief">Run <code>make xai-reports</code> to generate report.md for this model.</Empty>}
    </Panel>}

    <p className="small-copy">Report generated {detail.generated_at?new Date(detail.generated_at).toLocaleString('en-IN'):'—'}
     {detail.age_days!=null&&` · ${detail.age_days} days old`}
     {detail.config_versions&&` · feature config ${detail.config_versions.feature_config_version||'—'}`}
     {' · '}model config {health?.versions?.model_config_version||'—'}</p>
   </>}
  </>}
 </div>;
}
