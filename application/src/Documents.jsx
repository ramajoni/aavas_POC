import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, DownloadSimple, FileArrowUp, FileText } from '@phosphor-icons/react';
import { api, downloadFile } from './api';
import { Button, DataTable, Field, InfoBox, Panel } from './components';

const size = bytes => bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
const when = iso => iso ? new Date(iso).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—';

export function useDocumentTypes() {
  const [types, setTypes] = useState(null);
  useEffect(() => { api('/applications/document-types').then(setTypes).catch(() => setTypes({ document_types: {} })); }, []);
  return types;
}

// The borrower's side, after the indicative offer: apply, then upload documents.
// `onStatus` tells the journey panel how far the borrower has got.
export function BorrowerDocuments({ applicant, notify, onStatus }) {
  const loanId = applicant.loan_id;
  const types = useDocumentTypes();
  const [application, setApplication] = useState(undefined);   // undefined: loading, null: not applied
  const [kind, setKind] = useState('income_proof');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const file = useRef(null);

  // The borrower's own list first: asking for an application not yet submitted
  // would be an expected 404, and the browser logs every 404 as an error.
  async function load() {
    try {
      const { applications } = await api('/applications');
      const a = applications.some(x => x.loan_id === loanId) ? await api(`/applications/${encodeURIComponent(loanId)}`) : null;
      setApplication(a);
      onStatus?.({ applied: Boolean(a), documents: a?.documents || 0 });
    } catch (e) { setApplication(null); setError(e.message); }
  }
  useEffect(() => { load(); }, [loanId]);

  async function apply() {
    setBusy(true); setError('');
    try { await api('/applications', { loan_id: loanId, applicant }); await load(); notify('Application submitted. Please upload your documents.'); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function upload() {
    const chosen = file.current?.files?.[0];
    if (!chosen) { setError('Choose a file to upload.'); return; }
    const form = new FormData();
    form.append('document_type', kind);
    form.append('file', chosen);
    setBusy(true); setError('');
    try { await api(`/applications/${encodeURIComponent(loanId)}/documents`, form); file.current.value = ''; await load(); notify(`${chosen.name} uploaded.`); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  if (application === undefined) return null;
  if (application === null) return <div className="offer-next">
    <InfoBox success>Application <b>{loanId}</b>. Next, we’ll ask for your income and property documents.</InfoBox>
    {error && <p className="form-error" role="alert">{error}</p>}
    <Button primary type="button" busy={busy} onClick={apply}>Apply now <ArrowRight/></Button>
  </div>;

  const labels = types?.document_types || {};
  return <div className="offer-next borrower-documents">
    <InfoBox success>Application <b>{loanId}</b> submitted. {application.documents
      ? `${application.documents} document${application.documents === 1 ? '' : 's'} received — add any that are still missing.`
      : 'Upload your documents so we can verify them and confirm your rate.'}</InfoBox>
    {/* Not a <form>: this sits inside the offer page's form, and a nested form
        would submit that one natively and reload the page. */}
    <div className="document-upload" role="group" aria-label="Upload a document">
      <Field label="Document type">
        <select value={kind} onChange={e => setKind(e.target.value)} aria-label="Document type">
          {Object.entries(labels).map(([k, label]) => <option key={k} value={k}>{label}</option>)}
        </select>
      </Field>
      <Field label="File" hint={types ? `${(types.allowed_formats || []).map(f => f.toUpperCase()).join(', ')} · up to ${types.max_file_mb} MB` : ''}>
        <input ref={file} type="file" aria-label="File to upload" accept=".pdf,.jpg,.jpeg,.png,application/pdf,image/jpeg,image/png"/>
      </Field>
      <Button primary type="button" busy={busy} onClick={upload}><FileArrowUp size={18}/> Upload</Button>
    </div>
    {error && <p className="form-error" role="alert">{error}</p>}
    {application.document_list?.length > 0 && <DataTable
      columns={[
        { label: 'Document', render: d => labels[d.document_type] || d.document_type },
        { label: 'File', render: d => <span className="document-name"><FileText size={16}/> {d.filename}</span> },
        { label: 'Size', render: d => size(d.size_bytes) },
        { label: 'Uploaded', render: d => when(d.uploaded_at) },
      ]}
      rows={application.document_list.map(d => ({ ...d, id: d.document_id }))}/>}
  </div>;
}

// Staff side: what the borrower uploaded, each one downloadable.
export function ApplicationDocuments({ loanId, notify }) {
  const types = useDocumentTypes();
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { api(`/applications/${encodeURIComponent(loanId)}`).then(setDetail).catch(e => setError(e.message)); }, [loanId]);
  const labels = types?.document_types || {};

  if (error) return <InfoBox warning>{error}</InfoBox>;
  if (!detail) return <p className="muted">Loading documents…</p>;
  if (!detail.document_list.length) return <InfoBox>The borrower has not uploaded any documents yet.</InfoBox>;
  return <DataTable
    columns={[
      { label: 'Document', render: d => labels[d.document_type] || d.document_type },
      { label: 'File', render: d => <span className="document-name"><FileText size={16}/> {d.filename}</span> },
      { label: 'Size', render: d => size(d.size_bytes) },
      { label: 'Uploaded', render: d => when(d.uploaded_at) },
      { label: '', render: d => <Button onClick={() => downloadFile(`/applications/${encodeURIComponent(loanId)}/documents/${d.document_id}`, d.filename).catch(e => notify(e.message, 'error'))}><DownloadSimple size={17}/> Download</Button> },
    ]}
    rows={detail.document_list.map(d => ({ ...d, id: d.document_id }))}/>;
}

export { size as fileSize, when as shortTime };
