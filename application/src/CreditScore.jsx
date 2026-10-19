import React,{useEffect,useState} from 'react';
import {Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend} from 'chart.js';
import {Line} from 'react-chartjs-2';
import {Gauge,ChartBar,ArrowClockwise,WarningCircle,Eye,Info} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable} from './components';

ChartJS.register(CategoryScale,LinearScale,PointElement,LineElement,Tooltip,Legend);

const PALETTE=['#0756ff','#159a59','#b3541e','#7c3aed','#0e7490','#b91c1c'];
const score=n=>n==null?'—':Number(n).toFixed(0);
const out=(n,units)=>n==null?'—':units==='probability'?Number(n).toFixed(4):Number(n).toFixed(3);
const signed=(n,units)=>n==null?'—':`${n>0?'+':''}${out(n,units)}`;
const trustTone=l=>l==='strong'?'green':l==='moderate'?'amber':'red';

// Each model answers in its own units, so curves are only comparable within a
// unit. Plotting a probability against a rate on one axis would invent a
// relationship that does not exist.
function BandChart({models,units}){
 const rows=models.filter(m=>m.units===units&&m.curve?.points?.length);
 if(!rows.length)return null;
 const data={datasets:rows.map((m,i)=>({
   label:`${titleCase(m.model)} — ${m.output}`,
   data:m.curve.points.map(p=>({x:p.value,y:p.score})),
   borderColor:PALETTE[i%PALETTE.length],
   backgroundColor:PALETTE[i%PALETTE.length],
   tension:0.25,pointRadius:3,borderWidth:2,
 }))};
 const options={responsive:true,maintainAspectRatio:false,
  interaction:{mode:'nearest',intersect:false},
  plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{size:11}}},
   tooltip:{callbacks:{title:items=>`Credit score ${score(items[0].parsed.x)}`,
     label:item=>`${item.dataset.label}: ${out(item.parsed.y,units)}`}}},
  scales:{x:{type:'linear',title:{display:true,text:'Credit score (scored files only)'},
    ticks:{callback:v=>score(v),font:{size:11}}},
   y:{title:{display:true,text:units},ticks:{callback:v=>out(v,units),font:{size:11}}}}};
 return <div className="band-chart"><Line data={data} options={options}/></div>;
}

export function CreditScore({navigate}){
 const [body,setBody]=useState(null);const [loading,setLoading]=useState(false);const [error,setError]=useState('');

 async function load(){setLoading(true);setError('');
  try{setBody(await api('/xai/credit-score'));}
  catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);

 const models=body?.models||[];
 const first=models[0];
 // Every band is computed over the same holdout, so the scored range and the
 // thin-file cohort are properties of the book, not of a model.
 const ntcShare=first?.observed?.new_to_credit_pct;
 const rateModels=models.filter(m=>m.units!=='probability');
 const risk=models.find(m=>m.model==='risk');
 const ntcDelta=m=>{const o=m.observed;
  return o.new_to_credit.mean_prediction==null||o.floor_cohort.mean_prediction==null
   ? null : o.new_to_credit.mean_prediction-o.floor_cohort.mean_prediction;};
 // The contradiction worth naming: priced above the weakest scored files while
 // being modelled as less likely to default than them.
 const pricedWorse=rateModels.length&&rateModels.every(m=>(ntcDelta(m)??0)>0);
 const riskLower=risk&&(ntcDelta(risk)??0)<0;

 return <div className="operations">
  <div className="page-intro"><h1>Credit Score XAI</h1>
   <p>What the credit score does to a customer across the whole engine — the rate they are quoted, the
    risk they are assessed at, and whether the engine expects them to convert or leave. Each model is read
    two ways: the files that actually sit at each end of the score, and the score moved on its own with
    everything else held fixed.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}
  {body&&!body.available&&<Empty icon={WarningCircle} title="No credit-score band in any report">
    Generate them with <code>make xai-reports</code>, then refresh.</Empty>}

  {body?.available&&<>
   <div className="model-metrics">
    <div><small>Models covered</small><b>{models.length}</b></div>
    <div><small>Scored range</small><b>{score(first.observed.score_min)}–{score(first.observed.score_max)}</b></div>
    <div><small>Valid scores</small><b>{first.valid_range[0]}–{first.valid_range[1]}</b></div>
    <div><small>New to credit</small><b>{number(first.observed.new_to_credit.files)} ({ntcShare}%)</b></div>
    <div><small>Cohort size</small><b>bottom / top {first.tail_pct}%</b></div>
    <div><small>Report age</small><b>{first.age_days==null?'—':`${first.age_days} days`}</b></div>
   </div>

   <div className="actions end"><Button busy={loading} onClick={load}><ArrowClockwise/> Refresh</Button></div>

   {/* -------------------------------------------------- cross-model summary */}
   <Panel title="Bottom of the score against the top, per model" icon={Gauge}>
    <DataTable columns={[
      {label:'Model',render:m=>navigate
        ? <button className="link-button" onClick={()=>navigate('xai')}>{titleCase(m.model)}</button>
        : <b>{titleCase(m.model)}</b>},
      {label:'Predicts',render:m=><span>{m.output}<br/><small className="muted">{m.units}</small></span>},
      {label:`Bottom ${first.tail_pct}%`,
       render:m=><b>{out(m.observed.floor_cohort.mean_prediction,m.units)}</b>},
      {label:`Top ${first.tail_pct}%`,render:m=><b>{out(m.observed.ceiling_cohort.mean_prediction,m.units)}</b>},
      {label:'Gap',render:m=><span>{signed(m.observed.gap,m.units)}</span>},
      {label:'Holding all else fixed',render:m=>m.curve
        ? <span>{titleCase(m.curve.direction)} {out(Math.abs(m.curve.net_change),m.units)}</span>:'—'},
      {label:'Monotone',render:m=>m.curve?.monotone
        ? <Badge tone="green">yes</Badge>
        : <Badge tone="amber">no ({out(m.curve?.max_reversal,m.units)})</Badge>},
      {label:'Ranking trust',render:m=><Badge tone={trustTone(m.trust.level)}>{m.trust.level}</Badge>},
    ]} rows={models}/>
    <p className="small-copy">The gap is what the book looks like, so it carries everything that travels with a
     credit score — income, LTV, bureau vintage — not the score alone. The column beside it is the score moved on
     its own. A &quot;no&quot; under Monotone shows the largest move against the overall trend: a fraction of the
     swing is grid noise, a large one is a policy question. Ranking trust is the model&apos;s SHAP-vs-permutation
     agreement, and a weak one limits how far any of its attributions should be read.</p>
   </Panel>

   {/* -------------------------------------------------------------- curves */}
   {body.units.map(units=>
    <Panel key={units} title={`Response to the score — ${units} models`} icon={ChartBar}>
     <BandChart models={models} units={units}/>
     <p className="small-copy">Partial dependence over scored files only, so this is each model&apos;s response to
      the score rather than to the population that carries it. The grids differ between models because each is the
      report&apos;s own PDP grid, and the low end is the nearest grid point above the new-to-credit sentinel, not
      the lowest score in the book.</p>
    </Panel>)}

   {/* -------------------------------------------------------- new to credit */}
   <Panel title="New to credit — beside the band, never inside it" icon={Info}>
    <DataTable columns={[
      {label:'Model',render:m=><b>{titleCase(m.model)}</b>},
      {label:'New to credit',render:m=><b>{out(m.observed.new_to_credit.mean_prediction,m.units)}</b>},
      {label:`Weakest scored ${first.tail_pct}%`,render:m=>out(m.observed.floor_cohort.mean_prediction,m.units)},
      {label:'Difference',render:m=><span>{signed(ntcDelta(m),m.units)}</span>},
      {label:'Files',render:m=>number(m.observed.new_to_credit.files)},
    ]} rows={models}/>
    <p className="small-copy">A file with no bureau history carries the <code>{first.sentinel}</code> sentinel, which
     is the smallest <em>value</em> in the book and not a low score. Folded into the band it would become the worst
     credit in the book while being a different population entirely, so it is reported here instead.</p>
    {pricedWorse&&riskLower&&<InfoBox warning>
      <b>These files are priced above the weakest scored decile while being modelled as less likely to default than
       it.</b> {rateModels.map(m=>`${titleCase(m.model)} ${signed(ntcDelta(m),m.units)} ${m.units}`).join(', ')};
      risk {signed(ntcDelta(risk),risk.units)}. Both can be defensible — thin files carry uncertainty a point
      estimate does not show — but the two statements are made by the same engine about the same customers, and
      nothing in the pricing path reconciles them.
    </InfoBox>}
   </Panel>

   {/* ------------------------------------------------------ the two extremes */}
   <Panel title="The files at each end" icon={Eye}>
    <DataTable columns={[
      {label:'Model',render:m=><b>{titleCase(m.model)}</b>},
      {label:'Lowest scored file',render:m=>{const e=m.instances?.['lowest credit score'];
        return e?<span><code>{e.loan_id}</code> · score {score(e.credit_score)} · {out(e.prediction,m.units)}</span>:'—';}},
      {label:'Highest scored file',render:m=>{const e=m.instances?.['highest credit score'];
        return e?<span><code>{e.loan_id}</code> · score {score(e.credit_score)} · {out(e.prediction,m.units)}</span>:'—';}},
    ]} rows={models}/>
    <p className="small-copy">The same two applications run through every model. Their per-method attributions —
     KernelSHAP against LIME — are on {navigate
      ? <button className="link-button" onClick={()=>navigate('xai')}>Model Explainability</button>
      : <b>Model Explainability</b>}, under the model&apos;s Credit score tab.</p>
   </Panel>

   <p className="small-copy">{body.note} Reports generated
    {' '}{first.generated_at?new Date(first.generated_at).toLocaleString('en-IN'):'—'}.</p>
  </>}
 </div>;
}
