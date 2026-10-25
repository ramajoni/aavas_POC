import React,{useEffect,useState} from 'react';
import {Stack,ArrowClockwise,WarningCircle,TreeStructure,Scales,Gavel,Trophy} from '@phosphor-icons/react';
import {api,number,titleCase} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable} from './components';
import './ModelPortfolio.css';

// The five deck models with the evidence behind each: how AutoML chose the
// learner (out of time), what the trained artefact scores on the untouched
// holdout, and which features carry its weight. Every gap the API reports -
// a selection the artefact was not trained on, weights from an older artefact -
// is shown on the card, because a portfolio that only shows green is decoration.
const tone=s=>s==='ready'?'green':s==='partial'?'amber':'red';
const fmt=v=>v==null?'—':typeof v!=='number'?String(v):Math.abs(v)>=1000?number(Math.round(v)):Math.abs(v)>=10?v.toFixed(1):v.toFixed(3);
const day=s=>s?String(s).slice(0,10):'—';

function Weights({w}){
 if(!w)return null;const max=Math.max(...w.top.map(r=>r.share_pct||0),1);
 return <div className="mp-weights"><h4>Global weights <small>share of mean |SHAP| · holdout {number(w.holdout?.rows)} files</small></h4>
  {w.top.map(r=><div className="mp-weight" key={r.feature}><span>{r.label}</span><i><b style={{width:`${100*(r.share_pct||0)/max}%`}}/></i><em>{r.share_pct?.toFixed(1)}%</em></div>)}
  <small className="mp-src">{w.engine}{w.units&&w.units!=='output'?` · ${w.units.replace('_','-')}`:''}{w.fallback_reason?` · ${w.fallback_reason}`:''}</small></div>;
}

function Component({c,multi}){
 const a=c.automl,card=c.card;
 return <div className="mp-component">
  {multi&&<div className="mp-role"><b>{c.role||titleCase(c.model)}</b><code>{c.model}</code><Badge tone={tone(c.status)}>{c.status}</Badge></div>}
  <div className="mp-selected"><Trophy size={17} weight="duotone"/><div><small>Selected model</small><b>{a?a.selected.label:'Not selected yet'}</b>
   {a&&<span className="mp-why">{a.decided_by==='validation score'?'best validation score':a.decided_by?'within tolerance · more explainable':''}</span>}</div></div>
  <div className="mp-facts">
   <div><small>Validation</small><b>{a?.metric_text||'—'}</b></div>
   <div><small>Search</small><b>{a?`${a.trials} trials · ${a.families.length} families`:'—'}</b></div>
   <div><small>Target</small><b>{c.target||'—'}</b></div>
  </div>
  {a?.families_skipped?.length>0&&<p className="mp-skipped">{a.families_skipped.map(s=><span key={s.family} title={s.reason}>{s.kind==='excluded'?'excluded':'skipped'}: {s.family.replaceAll('_',' ')}</span>)}</p>}
  {card&&<><h4>Untouched holdout <small>{card.learner_label} · {card.version} · trained {day(card.trained_at)}</small></h4>
   <div className="mp-holdout">{Object.entries(card.holdout_metrics||{}).map(([k,v])=><div key={k}><small>{titleCase(k)}</small><b>{fmt(v)}</b></div>)}</div></>}
  <Weights w={c.weights}/>
  {c.issues?.length>0&&<ul className="mp-issues">{c.issues.map(i=><li key={i}><WarningCircle size={14} weight="fill"/> {i}</li>)}</ul>}
 </div>;
}

export function ModelPortfolio(){
 const [body,setBody]=useState(null);const [error,setError]=useState('');const [loading,setLoading]=useState(false);
 async function load(){setLoading(true);setError('');try{setBody(await api('/models/portfolio'));}catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);
 if(error&&!body)return <InfoBox warning><span role="alert">Model portfolio unavailable: {error}</span></InfoBox>;
 if(!body)return <Empty icon={Stack} title="Reading the model portfolio">Selection reports, model cards and global weights…</Empty>;
 const rule=body.selection_rule,h=rule.holdout;
 return <section className="mp">
  <div className="mp-head"><div><h2><Stack size={21}/> Model portfolio</h2><p className="muted">The five models behind a price, how each learner was chosen, and what it weighs.</p></div>
   <Button busy={loading} onClick={load}><ArrowClockwise/> Refresh</Button></div>
  <InfoBox><b>{rule.statement}</b> Validation is {rule.validation}{h?.from?` — holdout ${h.from} to ${h.to}, ${number(h.rows)} files`:''}. Ties within {(100*(rule.tolerance_rel||0)).toFixed(0)}% go to the more explainable family ({rule.explainability_ranking.map(titleCase).join(' › ')}).{rule.framework&&<> <span className="mp-framework">{rule.framework}</span></>}</InfoBox>
  <div className="mp-grid">{body.models.map((m,i)=><Panel key={m.key} className={`mp-card ${m.components.length>1?'wide':''}`} title={<><span className="mp-n">{i+1}</span>{m.title}</>} action={<Badge tone={tone(m.status)}>{m.status}</Badge>}>
   <p className="mp-stage">{m.stage}</p><p className="mp-question">{m.question}</p>
   <div className="mp-components">{m.components.map(c=><Component key={c.model} c={c} multi={m.components.length>1}/>)}</div>
  </Panel>)}</div>
  <Panel title="AutoML model selection" icon={TreeStructure}>
   <DataTable empty="No selection has been run yet: python scripts/run_model_selection.py" columns={[
    {label:'Model',render:r=><><b>{r.deck_model}</b>{r.role&&<small className="mp-cell-sub">{r.role}</small>}</>},
    {label:'Target',render:r=><code>{r.target}</code>},
    {label:'Algorithm families',render:r=><span className="mp-cell-families">{r.family_names.join(', ')}</span>},
    {label:'Trials',render:r=>number(r.trials)},
    {label:'Selected model',render:r=><><b>{r.selected}</b>{r.decided_by&&r.decided_by!=='validation score'&&<small className="mp-cell-sub">then explainability</small>}</>},
    {label:'Validation metric',key:'metric_text'}]} rows={body.automl_table}/>
  </Panel>
  <div className="mp-aside">
   {body.rules.map(r=><Panel key={r.key} title={r.title} icon={r.key==='guardrails'?Gavel:Scales} action={<Badge>rule</Badge>}>
    <p className="mp-question">{r.question}</p><p className="mp-settings">{r.settings.map(s=><code key={s.key}>{s.key.split('.').pop().replaceAll('_',' ')} = {String(s.value)}</code>)}</p><small className="mp-src">{r.source}</small></Panel>)}
   <Panel title="Supporting models" icon={Stack}>
    {body.supporting.map(s=><div className="mp-support" key={s.model}><b>{titleCase(s.model)}</b><span>{s.card?`${s.card.learner_label} · ${Object.entries(s.card.holdout_metrics||{}).map(([k,v])=>`${titleCase(k)} ${fmt(v)}`).join(' · ')}`:'not trained'}</span></div>)}
   </Panel>
  </div>
 </section>;
}
