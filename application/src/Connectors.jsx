import React,{useEffect,useState} from 'react';
import {PlugsConnected,ArrowClockwise,Lock,ShieldWarning,Database,ChatCircleText} from '@phosphor-icons/react';
import {api,titleCase} from './api';
import {Button,Panel,Badge,InfoBox,Empty,DataTable} from './components';

// Three layers, deliberately kept apart. A WhatsApp thread is a customer's own
// words and becomes a *signal*; a Salesforce record is a typed field on the
// application. Showing them in one undifferentiated list is how the second
// quietly becomes the first.
const LAYERS=[
 {key:'acceptance_signals', icon:ChatCircleText, title:'Communication channels',
  blurb:'Free text in the customer’s own words. Reaches the engine as an acceptance signal with a verbatim quote — never as a model feature.'},
 {key:'application_record', icon:Database, title:'System of record — the application',
  blurb:'Typed fields the engine already has columns for. Subject to the feature contract and its leakage guard; only the declared fields are accepted.'},
 {key:'economics', icon:Database, title:'System of record — economics',
  blurb:'The cost of a rupee lent. A change here moves every price, so values are proposed for approval rather than applied on read.'},
 {key:'outcomes', icon:Database, title:'Outcomes', blurb:'What actually happened: disbursal, repayment, default.'},
];

export function Connectors(){
 const [body,setBody]=useState(null);const [loading,setLoading]=useState(false);const [error,setError]=useState('');

 async function load(){setLoading(true);setError('');
  try{setBody(await api('/connectors'));}
  catch(e){setError(e.message);}finally{setLoading(false);}}
 useEffect(()=>{load();},[]);

 const all=body?.connectors||[];
 const columns=[
  {label:'System',render:c=><span><b>{c.system}</b><br/><small className="muted">{c.name}</small></span>},
  {label:'Supplies',render:c=><small>{c.description}</small>},
  {label:'Reliability',render:c=><Badge tone={c.reliability==='verified'?'green':'amber'}>{c.reliability.replace('_',' ')}</Badge>},
  {label:'Personal data',render:c=>c.personal_data?<Badge tone="amber">yes</Badge>:<Badge>no</Badge>},
  {label:'MCP',render:c=><small><code>{c.transport?.server}</code><br/>{c.transport?.tool}</small>},
  {label:'Status',render:c=><span>
    <Badge tone={c.status==='ready'?'green':'red'}>{c.status}</Badge>
    {c.blocked_reason&&<><br/><small className="muted">{c.blocked_reason}</small></>}</span>},
 ];

 return <div className="operations">
  <div className="page-intro"><h1>Connectors</h1>
   <p>The external systems this engine reads from, all of them over MCP — one client, one audit path, one
    place to revoke. Each one reports which gate is holding it: the switch, the server, or consent.</p></div>

  {error&&<InfoBox warning><span role="alert">{error}</span></InfoBox>}

  {body&&<>
   <div className="model-metrics">
    <div><small>Connectors</small><b>{all.length}</b></div>
    <div><small>Ready</small><b>{body.ready.length}</b></div>
    <div><small>Blocked</small><b>{body.blocked.length}</b></div>
    <div><small>Consent store</small><b>{body.consent?.configured?body.consent.backend:'none'}</b></div>
    <div><small>MCP client</small><b>{body.transport?.client_available?'available':'missing'}</b></div>
    <div><small>Version</small><b>{body.version}</b></div>
   </div>

   <div className="actions end"><Button busy={loading} onClick={load}><ArrowClockwise/> Re-probe</Button></div>

   {!body.consent?.configured&&<InfoBox warning><ShieldWarning size={17}/>
    <b>No consent store is configured.</b> Every connector that reads a customer&apos;s own words is blocked
    whatever its switch says — WhatsApp, Gmail and Outlook cannot be collected from until consent is recorded
    per customer and per purpose. That is the intended failure: a pricing engine that read a customer&apos;s
    messages because a YAML flag was flipped would have a consent problem, not a configuration problem.</InfoBox>}

   {!body.transport?.client_available&&<InfoBox warning>{body.transport?.reason}</InfoBox>}

   {LAYERS.map(layer=>{const rows=all.filter(c=>c.consumes===layer.key);
    if(!rows.length)return null;
    return <Panel key={layer.key} title={layer.title} icon={layer.icon}>
     <p className="small-copy">{layer.blurb}</p>
     <DataTable columns={columns} rows={rows}/>
     {layer.key==='application_record'&&rows.some(r=>r.fields?.length)&&
      <p className="small-copy"><b>Accepted fields:</b> {rows.flatMap(r=>r.fields).map(f=><code key={f}>{f} </code>)}
       — anything else the system returns is dropped. A connector must not be able to widen the feature contract.</p>}
     {layer.key==='economics'&&rows.some(r=>r.proposes?.length)&&
      <p className="small-copy"><b>Proposes:</b> {rows.flatMap(r=>r.proposes).map(f=><code key={f}>{f} </code>)}
       — proposed for approval, never applied on read.</p>}
    </Panel>;})}

   <Panel title="How a connector is switched on" icon={Lock}>
    <ol className="limitation-list">
     <li><b>The switch.</b> <code>enabled: true</code> in <code>config/connectors.yaml</code>, owned by the
      business, versioned with the file.</li>
     <li><b>The server.</b> An MCP endpoint the bank runs. Nothing here ships a credential or assumes a vendor
      endpoint; reachability is probed from the API host, which is not the same as reachable from a browser.</li>
     <li><b>Consent</b>, for any channel carrying a customer&apos;s own words — recorded per customer and per
      purpose, with messages older than {body.consent?.max_message_age_days??'—'} days out of scope even when
      consent exists. A two-year-old thread is not evidence about today&apos;s decision.</li>
    </ol>
    <p className="small-copy">{body.note}</p>
   </Panel>
  </>}

  {!body&&!error&&<Empty icon={PlugsConnected} title="Reading the connector registry">One moment…</Empty>}
 </div>;
}
