import React, { useState } from 'react';
import { Copy, Key } from '@phosphor-icons/react';
import { api } from './api';
import { Button, InfoBox, Modal } from './components';
import './customer-login.css';

// The customer's sign-in details, shown to the RM once: the temporary password is
// not stored in clear and cannot be shown again (a reset issues a new one).
export function CustomerLoginNotice({ login, name, onClose }) {
  const [copied, setCopied] = useState('');
  async function copy(label, text) {
    try { await navigator.clipboard.writeText(text); setCopied(label); } catch { setCopied(''); }
  }
  const linked = login.temporary_password == null;
  return <Modal title={linked ? 'Customer already has a login' : 'Customer login'} onClose={onClose}>
    {linked
      ? <p>{name} already has a login, <b>{login.username}</b>. This file is linked to it; their password is unchanged.</p>
      : <>
        <p>Give these to {name}. They sign in at this website and choose their own password straight away; the temporary
          password then stops working.</p>
        <dl className="cl-details">
          <div><dt>Username</dt><dd><code>{login.username}</code>
            <button type="button" className="link-button" onClick={() => copy('username', login.username)}><Copy size={14}/> {copied === 'username' ? 'Copied' : 'Copy'}</button></dd></div>
          <div><dt>Temporary password</dt><dd><code>{login.temporary_password}</code>
            <button type="button" className="link-button" onClick={() => copy('password', login.temporary_password)}><Copy size={14}/> {copied === 'password' ? 'Copied' : 'Copy'}</button></dd></div>
        </dl>
        <InfoBox warning>This password is shown only now. If it is lost, reset it from the file.</InfoBox>
      </>}
    <div className="modal-actions"><Button primary onClick={onClose}>Done</Button></div>
  </Modal>;
}

// On a file: who the customer signs in as, and a way to issue or reset their login.
export function CustomerLoginControl({ loanId, username, name, notify, onChange }) {
  const [busy, setBusy] = useState(false);
  const [login, setLogin] = useState(null);
  async function issue() {
    if (username && !window.confirm(`Reset ${name}'s password? Their current password stops working.`)) return;
    setBusy(true);
    try { const r = await api(`/applications/${encodeURIComponent(loanId)}/customer-login`, {}); setLogin(r); onChange?.(); }
    catch (e) { notify?.(e.message, 'error'); } finally { setBusy(false); }
  }
  return <span className="cl-control">
    {username ? <code>{username}</code> : <span className="muted">No login yet</span>}
    <Button busy={busy} onClick={issue}><Key size={15}/> {username ? 'Reset password' : 'Issue login'}</Button>
    {login && <CustomerLoginNotice login={login} name={name} onClose={() => setLogin(null)}/>}
  </span>;
}
