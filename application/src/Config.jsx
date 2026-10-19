import React,{useEffect,useMemo,useState} from 'react';
import {Sliders,MagnifyingGlass,ArrowClockwise,Lock,Terminal,FileText} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable,Field} from './components';

const when=s=>s?new Date(s).toLocaleString('en-IN'):'—';

// A value in these files is a number, a string, a boolean or a list of scalars.
// Booleans get a badge because "enabled: false" is the single most consequential
// shape in this configuration and it should not read as ordinary text.
function Value({value}){
 if(value===null||value===undefined)return <span className="muted">not set</span>;
 if(typeof value==='boolean')return <Badge tone={value?'green':'red'}>{String(value)}</Badge>;
 if(Array.isArray(value))return <span>{value.length?value.join(', '):<span className="muted">empty</span>}</span>;
 return <span>{String(value)}</span>;
}

export function Config(){
 const [index,setIndex]=useState(null);const [name,setName]=useState(null);const [detail,setDetail]=useState(null);
 const [query,setQuery]=useState('');const [hits,setHits]=useState(null);const [section,setSection]=useState('all');
 const [asWritten,setAsWritten]=useState(false);
 const [loading,setLoading]=useState(false);const [error,setError]=useState('');

 async function loadIndex(){setLoading(true);setError('');
  try{const r=await api('/config');setIndex(r);if(!name&&r.files?.length)setName(r.files[0].name);}
  catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{loadIndex();},[]);

 useEffect(()=>{if(!name)return;let live=true;setSection('all');
  api(`/config/${name}`).then(d=>{if(live)setDetail(d);}).catch(e=>{if(live)setError(e.message);});
  return()=>{live=false;};},[name]);

 // Search runs across every file, which is the point: a threshold you half
 // remember is rarely in the file you expected.
 useEffect(()=>{const q=query.trim();
  if(q.length<2){setHits(null);return;}
  const timer=setTimeout(()=>{api(`/config/search?q=${encodeURIComponent(q)}`)
    .then(setHits).catch(e=>setError(e.message));},250);
  return()=>clearTimeout(timer);},[query]);

 const rows=useMemo(()=>{
  const flat=detail?.flat||[];
  return section==='all'?flat:flat.filter(r=>r.path.split(/[.[]/)[0]===section);
 },[detail,section]);

 return <div className="operations">
  <div className="page-intro"><h1>Configuration</h1>
   <p>Every value the engine reads, in one place. Nothing in the code hard-codes a rate, spread, cost or
    threshold — it all lives in these six files, so this is the whole of the product&apos;s configurable
    behaviour.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  {index&&<>
   <div className="model-metrics">
    <div><small>Config files</small><b>{index.files.length}</b></div>
    <div><small>Settings</small><b>{number(index.settings)}</b></div>
    <div><small>Environment overrides</small><b>{index.overrides_active.length}</b></div>
    <div><small>Editable here</small><b>{index.editable?'yes':'no'}</b></div>
    <div><small>Policy version</small><b>{index.files.find(f=>f.name==='policy')?.version||'—'}</b></div>
    <div><small>Feature contract</small><b>{index.files.find(f=>f.name==='features')?.version||'—'}</b></div>
   </div>

   <div className="actions end"><Button busy={loading} onClick={loadIndex}><ArrowClockwise/> Reload</Button></div>

   <InfoBox><Lock size={16}/> {index.note}</InfoBox>

   {/* ---------------------------------------------------------- search */}
   <Panel title="Find a setting" icon={MagnifyingGlass}>
    <Field label="Search every file by key or value" value={query} placeholder="e.g. monotone, max_ece, approval, competitor"
      onChange={e=>setQuery(e.target.value)}/>
    {hits&&(hits.matches.length
      ? <><DataTable columns={[
          {label:'File',render:r=><button className="link-button" onClick={()=>{setName(r.config);setQuery('');}}>{r.file}</button>},
          {label:'Setting',render:r=><code>{r.path}</code>},
          {label:'Value',render:r=><Value value={r.value}/>},
        ]} rows={hits.matches}/>
        <p className="small-copy">{hits.matches.length} match{hits.matches.length===1?'':'es'}
         {hits.truncated&&' (truncated)'} for “{hits.query}”.</p></>
      : <p className="small-copy">Nothing matches “{hits.query}”.</p>)}
   </Panel>

   {/* ----------------------------------------------------- file picker */}
   <Panel title="Configuration files" icon={Sliders}>
    <DataTable columns={[
      {label:'File',render:f=><button className={`link-button${f.name===name?' active':''}`}
        onClick={()=>setName(f.name)}>{f.file}</button>},
      {label:'Governs',render:f=><small>{f.description}</small>},
      {label:'Version',render:f=><Badge tone="green">{f.version||'—'}</Badge>},
      {label:'Settings',render:f=>number(f.settings)},
      {label:'Modified',render:f=><small>{when(f.modified)}</small>},
    ]} rows={index.files}/>
    <p className="small-copy">The version each file declares is recorded on every decision, so an audit can
     say which configuration priced a given file — not merely which code did.</p>
   </Panel>

   {/* ------------------------------------------------------ one file */}
   {detail&&<Panel title={detail.file} icon={FileText}
     action={<Button onClick={()=>setAsWritten(!asWritten)}>
       {asWritten?'Show settings':'Show file as written'}</Button>}>
    <p className="small-copy">{detail.description}</p>

    {asWritten
      ? <><pre className="xai-brief">{detail.raw}</pre>
          <p className="small-copy">As written, comments included. In these files the comment is often the
           control — “never learn that more expensive converts better” is not expressible in YAML, so the
           reason a constraint exists lives beside it rather than in it.</p></>
      : <>
        <div className="chip-row">
         {['all',...detail.sections].map(s=><button key={s}
           className={`chip${section===s?' active':''}`} onClick={()=>setSection(s)}>
           {s==='all'?`All ${detail.settings}`:s}</button>)}
        </div>
        <DataTable columns={[
          {label:'Setting',render:r=><code>{r.path}</code>},
          {label:'Value',render:r=><Value value={r.value}/>},
        ]} rows={rows} empty="No settings in this section."/>
        <p className="small-copy">{rows.length} of {detail.settings} setting(s).
         Version <code>{detail.version}</code>, last modified {when(detail.modified)}.</p>
       </>}
   </Panel>}

   {/* ------------------------------------------------- env overrides */}
   <Panel title="Environment overrides" icon={Terminal}>
    <DataTable columns={[
      {label:'Variable',render:e=><code>{e.name}</code>},
      {label:'Controls',render:e=><small>{e.controls}</small>},
      {label:'Set',render:e=><Badge tone={e.set?'green':'blue'}>{e.set?'set':'unset'}</Badge>},
      {label:'Value',render:e=>e.secret
        ? <span className="muted">withheld — secret</span>
        : e.set?<code>{e.value}</code>:<span className="muted">—</span>},
    ]} rows={index.environment}/>
    <p className="small-copy">An environment variable wins over the file. Secrets are never served by this
     API: a credential is reported as set or unset and never carries a value.</p>
   </Panel>
  </>}

  {index===null&&!error&&<Empty icon={Sliders} title="Loading configuration">Reading the config directory…</Empty>}
 </div>;
}
