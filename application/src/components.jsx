import {apiKey, setApiKey} from './api';
import React, { useEffect, useRef } from 'react';
import { ArrowRight, X, CircleNotch, Info, CheckCircle, WarningCircle } from '@phosphor-icons/react';
export function Button({children, primary, busy, className='', ...props}) { return <button className={`button ${primary ? 'primary' : ''} ${className}`} disabled={busy || props.disabled} {...props}>{busy && <CircleNotch className="spin" size={17}/>} {children}</button>; }
export function Panel({title, icon: Icon, action, children, className=''}) { return <section className={`panel ${className}`}>{title && <div className="panel-head"><h2>{Icon && <Icon size={21}/>} {title}</h2>{action}</div>}{children}</section>; }
export function Badge({children, tone='blue'}) {return <span className={`badge ${tone}`}>{children}</span>;}
export function InfoBox({children, warning, success}) { const Icon = warning ? WarningCircle : success ? CheckCircle : Info; return <div className={`info-box ${warning ? 'warning' : success ? 'success' : ''}`}><Icon size={20}/><div>{children}</div></div>; }
export function Empty({icon: Icon = Info, title, children, action}) { return <div className="empty"><Icon size={32} weight="duotone"/><h3>{title}</h3><p>{children}</p>{action}</div>; }
export function Field({label, children, hint, ...props}) {return <label className="field"><span>{label}</span>{children || <input aria-label={label} {...props}/ >}{hint && <small>{hint}</small>}</label>;}
export function Modal({title, children, onClose, wide}) {
 const ref=useRef(null);
 useEffect(()=>{const previous=document.activeElement;ref.current?.showModal();return()=>previous?.focus();},[]);
 return <dialog className={`modal ${wide?'wide':''}`} ref={ref} onCancel={onClose} onClick={e=>{if(e.target===ref.current)onClose();}}><div className="modal-title"><h2>{title}</h2><button className="icon-button" aria-label="Close dialog" onClick={onClose}><X size={22}/></button></div>{children}</dialog>;
}
export function DataTable({columns, rows, empty='No records yet.'}) {return <div className="table-scroll"><table><thead><tr>{columns.map(c=><th key={c.label}>{c.label}</th>)}</tr></thead><tbody>{rows.map((row,i)=><tr key={row.id||i}>{columns.map(c=><td key={c.label}>{c.render?c.render(row):row[c.key]}</td>)}</tr>)}{!rows.length && <tr><td colSpan={columns.length}>{empty}</td></tr>}</tbody></table></div>;}
export const LinkButton = ({children,onClick}) => <button className="text-button" onClick={onClick}>{children}<ArrowRight size={16}/></button>;

// Shown when the API answers 401/403. The service is behaving correctly; the
// browser simply has no identity. Kept as a component rather than a global
// redirect because only two pages need a role - the priced path does not.
export function ApiKeyGate({message, onSaved}) {
  const [value, setValue] = React.useState('');
  const save = () => { setApiKey(value.trim()); onSaved && onSaved(); };
  return <div className="operations">
    <InfoBox warning>
      <b>This service is running with authentication enabled.</b> {message}{' '}
      Paste the API key for your role to continue. It is stored in this browser only —
      a real deployment replaces this with an SSO session.
    </InfoBox>
    <Panel title="Sign in">
      <div className="actions">
        <Field label="API key">
          <input type="password" value={value} autoComplete="off" aria-label="API key"
            onChange={e => setValue(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') save(); }}/>
        </Field>
        <Button primary onClick={save} disabled={!value.trim()}>Use this key</Button>
        {apiKey() && <Button onClick={() => { setApiKey(''); onSaved && onSaved(); }}>Forget key</Button>}
      </div>
      <p className="small-copy">Roles are declared in <code>config/governance.yaml</code> and granted
       per key through <code>AAVAS_API_KEYS</code>. A key with no recognised role is accepted and
       grants nothing, which is deliberate: an unknown role is dropped rather than honoured.</p>
    </Panel>
  </div>;
}
