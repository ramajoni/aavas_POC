import React,{useEffect,useMemo,useState} from 'react';
import {Sliders,MagnifyingGlass,ArrowClockwise,Lock,Terminal,FileText,PencilSimple,ClockCounterClockwise,CheckCircle,LockOpen} from '@phosphor-icons/react';
import {api,titleCase,number} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable,Field,Modal} from './components';
import './config-editor.css';

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
 const [editing,setEditing]=useState(false);const [history,setHistory]=useState(null);const [saved,setSaved]=useState(null);

 async function loadIndex(){setLoading(true);setError('');
  try{const r=await api('/config');setIndex(r);if(!name&&r.files?.length)setName(r.files[0].name);}
  catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{loadIndex();},[]);

 function loadFile(n=name){if(!n)return Promise.resolve();
  return Promise.all([api(`/config/${n}`),api(`/config/${n}/history`).catch(()=>({changes:[]}))])
   .then(([d,h])=>{setDetail(d);setHistory(h.changes);}).catch(e=>setError(e.message));}
 useEffect(()=>{if(!name)return;setSection('all');setEditing(false);setSaved(null);loadFile(name);},[name]);
 async function afterSave(result){setSaved(result);setEditing(false);await Promise.all([loadIndex(),loadFile(name)]);}

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

   <InfoBox>{index.editable?<LockOpen size={16}/>:<Lock size={16}/>} {index.note}</InfoBox>

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
      ...(index.editable?[{label:'Editable',render:f=>f.locked?<span title={f.locked}><Lock size={14}/> no</span>:'yes'}]:[]),
      {label:'Settings',render:f=>number(f.settings)},
      {label:'Modified',render:f=><small>{when(f.modified)}</small>},
    ]} rows={index.files}/>
    <p className="small-copy">The version each file declares is recorded on every decision, so an audit can
     say which configuration priced a given file — not merely which code did.</p>
   </Panel>

   {/* ------------------------------------------------------ one file */}
   {detail&&<Panel title={detail.file} icon={FileText}
     action={!editing&&<div className="actions">
       <Button onClick={()=>setAsWritten(!asWritten)}>{asWritten?'Show settings':'Show file as written'}</Button>
       {index.editable&&!detail.locked&&<Button primary onClick={()=>{setEditing(true);setSaved(null);}}><PencilSimple size={17}/> Edit file</Button>}</div>}>
    <p className="small-copy">{detail.description}</p>
    {index.editable&&detail.locked&&<InfoBox><Lock size={16}/> Not editable here. {detail.locked}</InfoBox>}
    {saved&&<InfoBox success><CheckCircle size={16}/> Applied: {saved.file} is now version <b>{saved.after_version}</b> (was {saved.before_version}),
      {' '}{saved.changes.length} setting{saved.changes.length===1?'':'s'} changed. New decisions are priced with it.</InfoBox>}

    {editing
      ? <ConfigEditor detail={detail} onCancel={()=>setEditing(false)} onSaved={afterSave}/>
      : asWritten
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

   {detail&&history&&<ChangeHistory file={detail.file} name={detail.name} changes={history}
     editable={index.editable} onRestored={afterSave}/>}

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

// A suggested next version: "X" -> "X-r1", "X-r1" -> "X-r2".
const nextVersion=v=>{const m=/^(.*)-r(\d+)$/.exec(v||'');return m?`${m[1]}-r${Number(m[2])+1}`:`${v}-r1`;};
const show=v=>v===null||v===undefined?'—':Array.isArray(v)?v.join(', '):String(v);

// Edit one file as written, comments included. Review shows every setting that
// would move; Save applies it, and the server prices a sample application with it
// and rolls it back if that fails.
function ConfigEditor({detail,onCancel,onSaved}){
 const [raw,setRaw]=useState(detail.raw);const [version,setVersion]=useState(nextVersion(detail.version));
 const [review,setReview]=useState(null);const [error,setError]=useState('');const [busy,setBusy]=useState(false);
 const reviewed=review&&review.raw===raw&&review.after_version===version;
 const body=()=>({raw,version});
 async function check(){setBusy(true);setError('');setReview(null);
  try{const r=await api(`/config/${detail.name}/preview`,body());setReview({...r,raw,after_version:version});}
  catch(e){setError(e.message);}finally{setBusy(false);}}
 async function save(){setBusy(true);setError('');
  try{onSaved(await api(`/config/${detail.name}`,body(),{method:'PUT'}));}
  catch(e){setError(e.message);}finally{setBusy(false);}}
 return <div className="config-editor">
  <InfoBox warning>Changes apply to the next priced decision. The file must keep every section and carry a new
   version; a sample application is priced with it before it is kept, and every change is recorded and can be undone.</InfoBox>
  <div className="config-editor-version">
   <Field label={`Current version`}><input value={detail.version||''} readOnly aria-label="Current version"/></Field>
   <Field label="New version" hint="Stamped on every decision priced with this file">
    <input value={version} onChange={e=>{setVersion(e.target.value);setReview(null);}} aria-label="New version"/></Field>
  </div>
  <label className="config-editor-text"><span>{detail.file}</span>
   <textarea value={raw} spellCheck={false} aria-label={`Contents of ${detail.file}`}
     onChange={e=>{setRaw(e.target.value);setReview(null);}}/></label>
  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}
  {reviewed&&<div className="config-review">
   <h3>{review.changes.length} setting{review.changes.length===1?'':'s'} will change · {review.before_version} → {review.after_version}</h3>
   <DataTable columns={[
     {label:'Setting',render:c=><code>{c.path}</code>},
     {label:'Change',render:c=><Badge tone={c.change==='changed'?'blue':c.change==='added'?'green':'red'}>{c.change}</Badge>},
     {label:'Before',render:c=><span className="muted">{show(c.before)}</span>},
     {label:'After',render:c=><b>{show(c.after)}</b>}]} rows={review.changes.map(c=>({...c,id:c.path}))}/>
  </div>}
  <div className="actions">
   <Button onClick={onCancel} disabled={busy}>Cancel</Button>
   <Button onClick={()=>{setRaw(detail.raw);setReview(null);setError('');}} disabled={busy||raw===detail.raw}>Discard edits</Button>
   {!reviewed
     ? <Button primary busy={busy} onClick={check} disabled={raw===detail.raw}>Review changes</Button>
     : <Button primary busy={busy} onClick={save}><CheckCircle size={17}/> Save and apply</Button>}
  </div>
 </div>;
}

function ChangeHistory({file,name,changes,editable,onRestored}){
 const [confirm,setConfirm]=useState(null);const [busy,setBusy]=useState(false);const [error,setError]=useState('');
 async function restore(){setBusy(true);setError('');
  try{const r=await api(`/config/${name}/history/${confirm.change_id}/restore`,{});setConfirm(null);onRestored(r);}
  catch(e){setError(e.message);}finally{setBusy(false);}}
 return <Panel title={`Change history · ${file}`} icon={ClockCounterClockwise}>
  {changes.length
   ? <DataTable columns={[
      {label:'When',render:c=><small>{when(c.changed_at)}</small>},
      {label:'By',key:'changed_by'},
      {label:'Version',render:c=><>{c.before_version} → <b>{c.after_version}</b></>},
      {label:'What changed',render:c=><small>{c.kind==='restore'?'Restored an earlier version · ':''}{c.changes.filter(x=>!x.path.endsWith('_version')).slice(0,3).map(x=>x.path).join(', ')||'version only'}{c.changes.length>4?` and ${c.changes.length-4} more`:''}</small>},
      ...(editable?[{label:'',render:c=><Button onClick={()=>setConfirm(c)}><ClockCounterClockwise size={16}/> Undo</Button>}]:[])]}
     rows={changes.map(c=>({...c,id:c.change_id}))}/>
   : <p className="small-copy">No changes have been made to this file from the console.</p>}
  {confirm&&<Modal title="Undo this change?" onClose={()=>setConfirm(null)}>
   <p>{file} goes back to version <b>{confirm.before_version}</b> as it was before {confirm.changed_by}&apos;s change on {when(confirm.changed_at)}.
    The undo is checked and test-priced like any edit, and recorded as a change of its own.</p>
   {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}
   <div className="modal-actions"><Button onClick={()=>setConfirm(null)}>Cancel</Button>
    <Button primary busy={busy} onClick={restore}>Undo change</Button></div>
  </Modal>}
 </Panel>;
}
