import React, { useEffect, useState } from 'react';
import { ArrowClockwise, FileText } from '@phosphor-icons/react';
import { api } from './api';
import { Button, Empty, InfoBox, Modal } from './components';
import { ApplicationDocuments } from './Documents';
import './applications.css';

const lakh = n => n == null ? '—' : `₹${(n / 100000).toFixed(1)} L`;
const range = r => r ? `${r.min_rate.toFixed(2)}%–${r.max_rate.toFixed(2)}%` : '—';
const quote = a => a.competitor_rate ? `${Number(a.competitor_rate).toFixed(2)}%, ${a.competitor_verified ? 'verified' : 'unverified'}` : 'None';
function waiting(iso) {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86400000);
  return { text: days <= 0 ? 'Today' : days === 1 ? '1 day' : `${days} days`, late: days >= 2 };
}

// "My applications": every application a borrower has submitted online, with what
// they uploaded. Open loads one into the loan assessment workspace.
export function Applications({ user, onOpen, notify }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [docsFor, setDocsFor] = useState(null);

  function load() {
    setBusy(true); setError('');
    api('/applications').then(r => setRows(r.applications)).catch(e => setError(e.message)).finally(() => setBusy(false));
  }
  useEffect(load, []);

  async function open(row) {
    try { const detail = await api(`/applications/${encodeURIComponent(row.loan_id)}`); onOpen(detail.applicant); }
    catch (e) { notify(e.message, 'error'); }
  }

  const awaiting = rows?.filter(r => r.status === 'awaiting_documents').length ?? 0;
  const received = rows?.filter(r => r.status === 'documents_received').length ?? 0;

  return <div className="operations my-applications">
    <div className="page-intro">
      <p className="eyebrow">{user.name} · {user.title}</p>
      <h1>My applications</h1>
    </div>
    <div className="app-kpis">
      <div className="app-kpi primary"><small>Documents received</small><b>{rows ? received : '—'}</b></div>
      <div className="app-kpi"><small>Awaiting documents</small><b>{rows ? awaiting : '—'}</b></div>
      <div className="app-kpi"><small>All applications</small><b>{rows ? rows.length : '—'}</b></div>
    </div>
    {error && <InfoBox warning><span role="alert">{error}</span></InfoBox>}
    {rows && !rows.length && <Empty icon={FileText} title="No applications yet">
      Applications appear here when a borrower applies online after seeing their indicative offer.</Empty>}
    {rows && rows.length > 0 && <div className="app-table">
      <table>
        <thead><tr><th>Applicant</th><th>Loan</th><th>Documents</th><th>Indicative range</th><th>Competitor quote</th><th>Waiting</th><th>Action</th></tr></thead>
        <tbody>{rows.map(r => {
          const w = waiting(r.updated_at);
          return <tr key={r.loan_id}>
            <td><b>{r.applicant_name}</b><small>{r.loan_id}{r.city ? ` · ${r.city}` : ''}</small></td>
            <td>{lakh(r.loan_amount)}</td>
            <td>{r.documents
              ? <button type="button" className="link-button doc-ok" onClick={() => setDocsFor(r)}>Received ({r.documents})</button>
              : <span className="doc-wait">Awaiting</span>}</td>
            <td>{range(r.indicative)}</td>
            <td>{quote(r)}</td>
            <td className={w.late ? 'late' : ''}>{w.text}</td>
            <td><button type="button" className="link-button" onClick={() => open(r)}>Open</button></td>
          </tr>;
        })}</tbody>
      </table>
    </div>}
    <div className="actions end"><Button busy={busy} onClick={load}><ArrowClockwise/> Refresh</Button></div>
    {docsFor && <Modal wide title={`Documents · ${docsFor.applicant_name} · ${docsFor.loan_id}`} onClose={() => setDocsFor(null)}>
      <ApplicationDocuments loanId={docsFor.loan_id} notify={notify}/>
    </Modal>}
  </div>;
}
