import React,{useEffect,useMemo,useState} from 'react';
import {Chart as ChartJS,CategoryScale,LinearScale,BarElement,PointElement,LineElement,BarController,LineController,Tooltip,Legend} from 'chart.js';
import {Chart} from 'react-chartjs-2';
import {Buildings,ArrowClockwise,Sparkle,Binoculars,Detective,ChartLineUp,SealCheck,SealWarning,Question,MapPin,Quotes,WarningCircle} from '@phosphor-icons/react';
import {api,percent,number,titleCase} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable} from './components';
import {MarketPosition,CustomerBehaviour} from './Behaviour';
import './Competitors.css';

ChartJS.register(CategoryScale,LinearScale,BarElement,PointElement,LineElement,BarController,LineController,Tooltip,Legend);

// Competitor intelligence: what the sales notes say about other lenders, read out
// of the text (not the one-slot structured field), joined to what then happened.
// Every number carries its n; thin cells arrive suppressed from the API and are
// shown as such rather than as a number. None of it is a price input.
const share=v=>v==null?'—':percent(v*100,1);
const rate=v=>v==null?'—':percent(v);
const bps=v=>v==null?'—':`${v>0?'+':''}${Number(v).toFixed(0)} bps`;
const pp=v=>v==null?'':`${v>0?'+':''}${Number(v).toFixed(1)} pp`;
const dist=d=>!d?'—':d.suppressed?`n=${d.n} (too few)`:d.median==null?'—':`${rate(d.median)} (${rate(d.p10)}–${rate(d.p90)})`;
const evidenceTone=e=>e==='documented'?'green':e==='undocumented'?'amber':'blue';
const PALETTE=['#0756ff','#159a59','#b3541e','#7c3aed','#0e7490','#b91c1c','#a16207','#475569'];
const axis={grid:{color:'#eef3fa'},ticks:{font:{size:10}}};

function GapCurve({curve,base}){
 const data={labels:curve.map(c=>c.bucket),datasets:[
  {type:'bar',label:'Win rate (disbursed)',data:curve.map(c=>c.rate==null?null:c.rate*100),backgroundColor:curve.map(c=>c.rate==null?'#d8e1ef':'#0756ff'),borderRadius:3},
  {type:'line',label:'Book base rate',data:curve.map(()=>base==null?null:base*100),borderColor:'#b3541e',borderDash:[5,4],pointRadius:0,borderWidth:1.5}]};
 const options={responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{boxWidth:10,font:{size:10}}},
  tooltip:{callbacks:{afterLabel:i=>i.datasetIndex===0?`n = ${curve[i.dataIndex].n}${curve[i.dataIndex].suppressed?' (suppressed)':''}`:''}}},
  scales:{x:{...axis,title:{display:true,text:'Competitor quote minus our sanctioned rate (bps)',font:{size:10}}},y:{...axis,min:0,max:100,title:{display:true,text:'Win rate %',font:{size:10}}}}};
 return <div className="ci-chart"><Chart type="bar" data={data} options={options}/></div>;
}

function Trend({monthly}){
 const data={labels:monthly.map(m=>m.month),datasets:[
  {type:'bar',label:'Files mentioning',data:monthly.map(m=>m.loans_mentioned),backgroundColor:'#c9dafc',yAxisID:'y',order:2},
  {type:'line',label:'Median quote %',data:monthly.map(m=>m.median_quote),borderColor:'#159a59',backgroundColor:'#159a59',yAxisID:'y1',spanGaps:false,pointRadius:2,tension:.25,order:1}]};
 const options={responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{boxWidth:10,font:{size:10}}}},
  scales:{x:{...axis,ticks:{font:{size:9},maxRotation:0,autoSkip:true}},y:{...axis,beginAtZero:true,title:{display:true,text:'Files',font:{size:10}}},
   y1:{...axis,position:'right',beginAtZero:false,grace:'15%',grid:{drawOnChartArea:false},title:{display:true,text:'Median quote %',font:{size:10}}}}};
 return <div className="ci-chart"><Chart type="bar" data={data} options={options}/></div>;
}

function Emerging({rows}){
 const months=[...new Set(rows.flatMap(r=>r.monthly_loans.map(m=>m.month)))].sort();
 const data={labels:months,datasets:rows.map((r,i)=>{const by=Object.fromEntries(r.monthly_loans.map(m=>[m.month,m.loans]));
  return {type:'line',label:r.competitor,data:months.map(m=>by[m]??0),borderColor:PALETTE[i%PALETTE.length],backgroundColor:PALETTE[i%PALETTE.length],pointRadius:1.5,borderWidth:1.6,tension:.25};})};
 const options={responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{boxWidth:10,font:{size:10}}}},scales:{x:{...axis,ticks:{font:{size:9},maxRotation:0,autoSkip:true}},y:{...axis,beginAtZero:true,title:{display:true,text:'Files mentioning',font:{size:10}}}}};
 return <div className="ci-chart small"><Chart type="line" data={data} options={options}/></div>;
}

function Quality({summary}){
 const q=summary?.quality;
 if(!q)return <Empty icon={Detective} title="Reading the extraction">One moment…</Empty>;
 if(!q.available)return <InfoBox warning>{q.reason}</InfoBox>;
 const m=q.competitor_identification.micro,r=q.rates,e=q.evidence_vs_structured_flag,c=summary.coverage;
 const per=Object.entries(q.competitor_identification.per_competitor).map(([name,v])=>({id:name,name,...v}));
 return <>
  <InfoBox warning><b>Measurable only because the data is synthetic.</b> {q.caveat.replace(/^Measurable only because the data is synthetic:\s*/,'')}</InfoBox>
  <div className="model-metrics ci-metrics">
   <div><small>Notes scored</small><b>{number(q.notes_scored)}</b></div>
   <div><small>Competitor precision</small><b>{share(m.precision)}</b></div>
   <div><small>Recall</small><b>{share(m.recall)}</b></div>
   <div><small>F1 (micro, n={number(m.support)})</small><b>{share(m.f1)}</b></div>
   <div><small>Rate exact match (n={number(r.scored)})</small><b>{share(r.exact_match_accuracy)}</b></div>
   <div><small>Evidence vs structured flag (n={number(e.classified)})</small><b>{share(e.agreement)}</b></div>
  </div>
  <div className="ci-two">
   <div><h3>Per competitor</h3><DataTable columns={[{label:'Lender',key:'name'},{label:'Precision',render:x=>share(x.precision)},{label:'Recall',render:x=>share(x.recall)},{label:'Support',render:x=>number(x.support)},{label:'Rates exact',render:x=>`${number(x.rate_exact)} / ${number(x.rate_scored)}`}]} rows={per}/></div>
   <div><h3>What the score does not say</h3><ul className="ci-list">
    <li>Rates: {number(r.exact)} exact, {number(r.wrong)} wrong, {number(r.missing)} missing.{q.examples?.missing_rate?.length>0&&<> A missing one, for example: <i>“{q.examples.missing_rate[0].text}”</i> — truth gives {q.examples.missing_rate[0].competitor} {rate(q.examples.missing_rate[0].truth)}. Where the text ties the number to <i>our</i> rate the extractor reads it as the customer's walk-away rate, not a competitor quote.</>}</li>
    <li>{number(r.truth_unaligned_notes)} notes list fewer true rates than lenders; they are scored by value, not position.</li>
    <li>Evidence: {number(e.confusion.undocumented_but_flag_verified)} quotes the file marks <b>verified</b> are described in the note as verbal, WhatsApp or "no document"; {number(e.confusion.documented_but_flag_unverified)} the other way. The flag is a human entry, not truth — these are findings about the file. {share(e.coverage)} of structured-lender mentions carry an evidence cue at all.</li>
    <li>{number(c.fuzzy_matches)} mentions matched only by near-spelling, {number(c.ambiguous_alias_matches)} by a bare ambiguous word in a lending sentence; {number(c.unrecognised_mentions)} lender-shaped names were unknown.</li>
    <li>The synthetic notes use a narrow vocabulary; expect lower scores on real notes and score a hand-labelled sample there.</li>
   </ul></div>
  </div></>;
}

export function Competitors(){
 const [body,setBody]=useState(null);const [notes,setNotes]=useState(null);const [selected,setSelected]=useState(null);const [profile,setProfile]=useState(null);
 const [error,setError]=useState('');const [loading,setLoading]=useState(false);
 async function load(){setLoading(true);setError('');
  try{const [c,n]=await Promise.all([api('/competitors'),api('/notes/summary')]);setBody(c);setNotes(n);if(!selected&&c.league?.length)setSelected(c.league[0].competitor);}
  catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);
 const [profileError,setProfileError]=useState('');const [attempt,setAttempt]=useState(0);
 useEffect(()=>{if(!selected)return;let live=true;setProfile(null);setProfileError('');api(`/competitors/${encodeURIComponent(selected)}`).then(p=>{if(live){setProfile(p);setError('');}}).catch(e=>live&&setProfileError(e.message));return()=>{live=false;};},[selected,attempt]);
 if(error&&!body)return <div className="operations"><InfoBox warning><span role="alert">{error}</span></InfoBox></div>;
 if(!body)return <div className="operations"><Empty icon={Binoculars} title="Reading the sales notes">The first call after a restart builds the historical tables — a few seconds.</Empty></div>;
 const {coverage:cov,base_rates:base}=body;const q=notes?.quality;
 const league=body.league.map(r=>({...r,id:r.competitor}));
 return <div className="operations ci">
  <div className="page-intro"><h1>Competitor Intelligence</h1>
   <p>Who our customers compare us with, read out of the sales notes — the lender as the branch wrote it, the rate attached to it and whether a document backs it — joined to what then happened to the file.</p></div>
  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}
  <InfoBox><b>Context for reviewers, never a price input.</b> {body.caveats.join(' ')} Cells with fewer than {body.settings.min_n_cell} quotes (or {body.settings.min_n_rate} files for a rate) are suppressed.</InfoBox>
  <div className="model-metrics ci-metrics">
   <div><small>Notes read</small><b>{number(notes?.coverage.notes)}</b></div>
   <div><small>Files with a competitor in the notes</small><b>{number(cov.loans_with_competitor_mentions)} / {number(cov.loans)}</b></div>
   <div><small>Lenders seen</small><b>{cov.competitors_seen}</b></div>
   <div><small>Quoted file-lender pairs</small><b>{number(cov.quoted_pairs)}</b></div>
   <div><small>Book win rate</small><b>{share(base.win_rate)}</b></div>
   <div><small>Extraction F1 (synthetic truth)</small><b>{share(q?.competitor_identification?.micro?.f1)}</b></div>
   <Button onClick={load} busy={loading}><ArrowClockwise/> Refresh</Button>
  </div>

  <MarketPosition league={league} base={base.win_rate}/>
  <Panel title="How acceptance falls as competitors undercut us" icon={ChartLineUp}>
   <p className="small-copy">Every lender together: our win rate by how far the competitor's quote sat below or above the rate we sanctioned, against the book's {share(base.win_rate)}.</p>
   <GapCurve curve={body.gap_win_curve_all} base={base.win_rate}/>
  </Panel>
  <CustomerBehaviour/>

  <Panel title="League table" icon={Buildings} action={<small className="muted">Select a lender to drill down</small>}>
   <DataTable columns={[
    {label:'Lender',render:r=><button className={`text-button ci-pick ${r.competitor===selected?'active':''}`} onClick={()=>setSelected(r.competitor)}>{r.competitor}</button>},
    {label:'Type',render:r=><>{r.lender_type}{!r.in_structured_data&&<> <Badge tone="amber">notes only</Badge></>}</>},
    {label:'Share of files',render:r=><>{share(r.share_of_loans)} <small className="muted">n={number(r.loans_mentioned)}</small></>},
    {label:'Median quote (p10–p90)',render:r=>dist(r.quotes)},
    {label:'Gap to our rate',render:r=>bps(r.gap_to_our_rate_bps.median)},
    {label:'Gap to card',render:r=>bps(r.gap_to_card_bps.median)},
    {label:'Win rate when present',render:r=>r.win_rate.rate==null?`n=${r.win_rate.n}`:<>{share(r.win_rate.rate)} <small className={r.win_rate.lift_pp<0?'red':'green-text'}>{pp(r.win_rate.lift_pp)}</small></>},
    {label:'Documented quotes',render:r=>share(r.evidence.documented_share)},
    {label:'Override when cited',render:r=>r.override_when_cited.rate==null?'—':<>{share(r.override_when_cited.rate)} <small className="muted">median {bps(r.override_when_cited.median_depth_bps)}</small></>},
    {label:'Most aggressive in',render:r=>r.most_aggressive_city?`${r.most_aggressive_city.city} (${bps(r.most_aggressive_city.median_gap_bps)})`:'—'}]} rows={league}/>
   <p className="small-copy">Base rates across the book: win {share(base.win_rate)}, override {share(base.override_rate)}; files whose notes name no competitor win {share(base.win_rate_no_competitor_in_notes)} (n={number(base.n_no_competitor_in_notes)}). Gap is the latest quote in the notes minus the rate we sanctioned.</p>
  </Panel>

  {selected&&<Panel title={`${selected} — in depth`} icon={MapPin} action={profile&&<span className="muted">{number(profile.loans_mentioned)} files · first seen {profile.first_seen}</span>}>
   {profileError?<InfoBox warning><span role="alert">{profileError}</span> <Button onClick={()=>setAttempt(n=>n+1)}><ArrowClockwise/> Retry</Button></InfoBox>:!profile?<Empty icon={Binoculars} title="Loading">One moment…</Empty>:<>
    <div className="model-metrics ci-metrics">
     <div><small>Median quote (p10–p90)</small><b>{dist(profile.quotes)}</b></div>
     <div><small>Gap to our rate (median)</small><b>{bps(profile.gap_to_our_rate_bps.median)}</b></div>
     <div><small>Win rate when present</small><b>{share(profile.win_rate.rate)} <small>vs {share(profile.win_rate.base_rate)}</small></b></div>
     <div><small>Evidence (documented / verbal / none)</small><b>{number(profile.evidence.documented)} / {number(profile.evidence.undocumented)} / {number(profile.evidence.unknown)}</b></div>
     <div><small>Override when cited</small><b>{share(profile.override_when_cited.rate)} <small>median {bps(profile.override_when_cited.median_depth_bps)}</small></b></div>
    </div>
    <div className="ci-two">
     <div><h3>How acceptance falls as they undercut us</h3><GapCurve curve={profile.gap_win_curve} base={profile.win_rate.base_rate}/></div>
     <div><h3>Monthly mentions and median quote</h3><Trend monthly={profile.monthly}/></div>
    </div>
    <div className="ci-two">
     <div><h3>City footprint</h3><DataTable columns={[{label:'City',key:'city'},{label:'Files',render:r=>number(r.loans_mentioned)},{label:'Share of city files',render:r=>share(r.share_of_loans)},{label:'Median quote',render:r=>r.suppressed?`n=${r.n_quotes}`:rate(r.median_quote)},{label:'Median gap',render:r=>r.suppressed?'—':bps(r.median_gap_bps)}]} rows={profile.by_city.map(r=>({...r,id:r.city}))}/></div>
     <div><h3>Where they are most aggressive</h3><DataTable columns={[{label:'City',key:'city'},{label:'Median gap',render:r=>bps(r.median_gap_bps)},{label:'Median quote',render:r=>rate(r.median_quote)},{label:'n',render:r=>number(r.n)}]} rows={profile.most_aggressive_cities.map(r=>({...r,id:r.city}))} empty="Too few quotes in any city."/>
      <h3>By segment</h3><DataTable columns={[{label:'Segment',render:r=>titleCase(r.segment)},{label:'Files',render:r=>number(r.loans_mentioned)},{label:'Median quote',render:r=>r.suppressed?`n=${r.n_quotes}`:rate(r.median_quote)},{label:'Median gap',render:r=>r.suppressed?'—':bps(r.median_gap_bps)}]} rows={profile.by_segment.map(r=>({...r,id:r.segment}))}/>
      <h3>How branches write the name</h3><div className="chip-row">{Object.entries(profile.aliases_seen).map(([a,n])=><span className="chip" key={a}>{a} <small>{number(n)}</small></span>)}</div></div>
    </div></>}
  </Panel>}

  <div className="ci-two">
   <Panel title="Who is new in our market" icon={ChartLineUp}>
    <p className="small-copy">Lenders that appear only in the note text — the structured competitor field never had a slot for them.</p>
    {body.emerging.length?<><Emerging rows={body.emerging}/><DataTable columns={[{label:'Lender',render:r=><button className="text-button ci-pick" onClick={()=>setSelected(r.competitor)}>{r.competitor}</button>},{label:'First seen',key:'first_seen'},{label:'Files',render:r=>number(r.loans_mentioned)},{label:'Footprint',render:r=>`${r.cities} cities · ${r.top_cities.map(c=>c.city).join(', ')}`},{label:'Median quote',render:r=>dist(r.quotes)}]} rows={body.emerging.map(r=>({...r,id:r.competitor}))}/></>
     :<Empty title="No note-only lenders">Every lender in the notes also appears in the structured field.</Empty>}
   </Panel>
   <Panel title="Unrecognised lenders for review" icon={Question}>
    <p className="small-copy">Lender-shaped names ("&lt;Name&gt; Housing Finance") the registry does not know. They are listed, never mapped to the nearest known lender; an analyst adds them to <code>config/competitors.yaml</code> or rejects them.</p>
    {body.unrecognised.length?<DataTable columns={[{label:'Name as written',key:'name'},{label:'Notes',render:r=>number(r.notes)},{label:'First seen',key:'first_seen'},{label:'Cities',render:r=>r.cities.join(', ')},{label:'Median quote',render:r=>rate(r.median_quote)}]} rows={body.unrecognised.map(r=>({...r,id:r.name}))}/>
     :<Empty icon={SealCheck} title="None in this file">All {number(notes?.coverage.mentions)} lender mentions in {number(notes?.coverage.notes)} notes resolved to the registry ({number(notes?.coverage.fuzzy_matches)} by near-spelling).</Empty>}
   </Panel>
  </div>

  <Panel title="Extraction quality" icon={Detective} action={<Badge tone="amber">synthetic truth</Badge>}><Quality summary={notes}/></Panel>
 </div>;
}

// ------------------------------------------------------------- application
// Marks the alias and the rate inside the note so the reviewer reads the claim
// against the words it came from.
function Highlight({text,marks}){
 const spans=[...marks].filter(m=>m.span).sort((a,b)=>a.span[0]-b.span[0]);const out=[];let at=0;
 spans.forEach((m,i)=>{const [s,e]=m.span;if(s<at)return;if(s>at)out.push(text.slice(at,s));out.push(<mark key={i} className={`ci-mark ${m.kind}`} title={m.title}>{text.slice(s,e)}</mark>);at=e;});
 if(at<text.length)out.push(text.slice(at));return <p className="ci-note-text">{out}</p>;
}

const salesFields=sc=>({competitor_name:sc?.competitor_name||null,competitor_rate:sc?.competitor_rate??null,competitor_evidence_verified:sc?.competitor_evidence_verified??null});

export function NoteSignals({notes,salesContext,loanId}){
 const [result,setResult]=useState(null);const [busy,setBusy]=useState(false);const [error,setError]=useState('');
 useEffect(()=>{setResult(null);},[notes?.length]);
 async function run(){setBusy(true);setError('');try{setResult(await api('/notes/extract',{loan_id:loanId,notes:notes.map(n=>n.text),sales_context:salesFields(salesContext)}));}catch(e){setError(e.message);}finally{setBusy(false);}}
 const signals=result?Object.entries(result.customer_signals):[];
 return <div className="ci-extract">
  <div className="actions"><Button primary busy={busy} disabled={!notes?.length} onClick={run}><Sparkle size={17}/> Extract signals</Button>
   {result&&<small className="muted">Competitors: {result.method.competitors}. Customer signals: {String(result.method.customer_signals)}{result.method.llm_enabled?'':' (LLM off)'}.</small>}</div>
  {error&&<div role="alert" className="inline-error">{error}</div>}
  {result&&<>
   {result.conflicts.filter(c=>c.severity==='warning').map((c,i)=><InfoBox warning key={i}><b>{titleCase(c.type)}.</b> {c.message}</InfoBox>)}
   <h3 className="ci-sub">Competitor claims</h3>
   {result.competitors.length?<DataTable columns={[
    {label:'Lender',render:c=><><b>{c.competitor}</b> <small className="muted">{c.lender_type}</small></>},
    {label:'As written',render:c=>c.claims.map(x=>x.alias).filter((v,i,a)=>a.indexOf(v)===i).join(', ')},
    {label:'Quoted rate',render:c=>c.rate==null?'—':<>{rate(c.rate)} <small className="muted">“{c.rate_text}”</small></>},
    {label:'Evidence',render:c=><><Badge tone={evidenceTone(c.evidence)}>{c.evidence}</Badge>{c.evidence_cue&&<small className="muted"> “{c.evidence_cue}”</small>}</>},
    {label:'Note',render:c=>c.rate_note_index==null?c.claims.map(x=>x.note_index+1).join(', '):`#${c.rate_note_index+1}`}]} rows={result.competitors.map(c=>({...c,id:c.competitor}))}/>
    :<p className="muted">No lender is named in these notes.</p>}
   {result.unrecognised_lenders.length>0&&<InfoBox><b>Unrecognised lender{result.unrecognised_lenders.length>1?'s':''} for review:</b> {result.unrecognised_lenders.map(u=>`“${u.alias}”`).join(', ')} — not in the registry, so not mapped.</InfoBox>}
   <h3 className="ci-sub">Customer signals</h3>
   <div className="chip-row">{signals.length?signals.map(([k,v])=><span className="chip" key={k} title={v.quote||''}>{titleCase(k)}: <b>{typeof v.value==='number'?(k.includes('rate')?rate(v.value):number(v.value)):titleCase(String(v.value))}</b> <small>note #{v.note_index+1}</small></span>):<span className="muted">None stated.</span>}</div>
   {result.customer_asks.length>0&&<p className="small-copy">Customer's own ask, kept apart from competitor quotes: {result.customer_asks.map(a=>`${rate(a.rate)} (“${a.cue}”, note #${a.note_index+1})`).join('; ')}.</p>}
   {result.conflicts.filter(c=>c.severity!=='warning').length>0&&<ul className="ci-list">{result.conflicts.filter(c=>c.severity!=='warning').map((c,i)=><li key={i}>{c.message}</li>)}</ul>}
   <h3 className="ci-sub">Quoted spans</h3>
   {result.notes.map(n=><Highlight key={n.note_index} text={n.text} marks={[...n.mentions.flatMap(m=>[{span:m.span,kind:'lender',title:`${m.competitor} (${m.match})`},m.rate_span&&{span:m.rate_span,kind:'rate',title:`quote ${m.rate}%`}].filter(Boolean)),...n.unrecognised_lenders.map(m=>({span:m.span,kind:'unknown',title:'unrecognised lender'}))]}/>)}
   <p className="tiny-note">Pricing effect: {result.pricing_effect.replace(/^none - /,'none — ')}</p>
  </>}
 </div>;
}

export function CompetitorContext({applicant,notes,ourRate,grade}){
 const [ctx,setCtx]=useState(null);const [error,setError]=useState('');
 const request=useMemo(()=>({application:{loan_id:applicant.loan_id,city:applicant.city,segment:applicant.segment,risk_grade:grade||null,our_rate:ourRate??null,sales_context:salesFields(applicant.sales_context)},notes:(notes||[]).map(n=>n.text)}),[applicant.loan_id,applicant.city,applicant.segment,applicant.sales_context,grade,ourRate,notes]);
 const key=JSON.stringify(request);
 useEffect(()=>{let live=true;setError('');api('/competitors/context',request).then(r=>live&&setCtx(r)).catch(e=>live&&setError(e.message));return()=>{live=false;};},[key]);
 if(error)return <p className="ci-line">Market context unavailable: {error}</p>;
 if(!ctx)return <p className="ci-line">Reading market context…</p>;
 const unknown=ctx.conflicts.find(c=>c.type==='unrecognised_structured_competitor');
 return <div className="ci-context">
  {unknown&&<p className="ci-line">{unknown.message}</p>}
  {ctx.competitors.map(c=><div className="ci-context-item" key={c.competitor}>
   <div className="ci-context-head"><b>{c.competitor}</b><span>{rate(c.quote)}</span></div>
   <small className="muted">{c.quote_source==='notes'?'from the notes':c.quote_source==='structured'?'from the application':'no rate quoted'} · <Badge tone={evidenceTone(c.evidence)}>{c.evidence}</Badge></small>
   {c.benchmark.level?<p className="ci-line">History ({c.benchmark.level.replaceAll('+',' + ')}, n={c.benchmark.n}): median {rate(c.benchmark.median)}, p10–p90 {rate(c.benchmark.p10)}–{rate(c.benchmark.p90)}</p>:<p className="ci-line">Too few historical quotes to benchmark.</p>}
   {c.plausibility.checked&&(c.plausibility.flag?<p className="ci-flag"><WarningCircle size={14}/> <span><b>{c.plausibility.message.replace(' - ',' — ')}</b> · below this lender's {c.plausibility.level==='city'?ctx.city:c.plausibility.level} p{c.plausibility.percentile} of {rate(c.plausibility.p5)} (n={c.plausibility.n})</span></p>:<p className="ci-line">{c.plausibility.message}</p>)}
   {c.win_rate_at_gap.available?(c.win_rate_at_gap.beyond_history?<p className="ci-line">Gap {bps(c.win_rate_at_gap.gap_bps)} vs our rate · beyond anything in history ({bps(c.win_rate_at_gap.observed_range_bps[0])} to {bps(c.win_rate_at_gap.observed_range_bps[1])}); no comparable win rate. Widest bucket for reference: {share(c.win_rate_at_gap.all_cities.rate)} of {number(c.win_rate_at_gap.all_cities.n)}.</p>:<p className="ci-line">Gap {bps(c.win_rate_at_gap.gap_bps)} vs our rate · we won {share(c.win_rate_at_gap.all_cities.rate)} of {number(c.win_rate_at_gap.all_cities.n)} past files at a similar gap{c.win_rate_at_gap.city?.rate!=null?` (${share(c.win_rate_at_gap.city.rate)} in ${ctx.city}, n=${c.win_rate_at_gap.city.n})`:''}</p>):<p className="ci-line">Run the assessment to see the win rate at this gap.</p>}
  </div>)}
  {!ctx.competitors.length&&!unknown&&<p className="ci-line">No recognised competitor on this file yet.</p>}
  {ctx.comparable_negotiations?.length>0&&<details className="ci-comparables"><summary><Quotes size={13}/> Comparable past negotiations ({ctx.comparable_negotiations.length})</summary>{ctx.comparable_negotiations.map(h=><div key={h.doc_id}><p>“{h.text}”</p><small className="muted">{h.why_comparable}{h.outcome?` · ${h.outcome.disbursed?'disbursed':'lost'} at ${rate(h.outcome.sanctioned_rate)}`:''}</small></div>)}</details>}
 </div>;
}
