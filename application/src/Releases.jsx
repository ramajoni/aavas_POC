import React, { useEffect, useMemo, useState } from 'react';
import { Newspaper, CaretDown, CircleNotch, MagnifyingGlass, WarningCircle } from '@phosphor-icons/react';
import { api } from './api';
import { Empty } from './components';
import { Markdown } from './markdown';
import './release-notes.css';

// Every change to the application is versioned; this page is the notes for each release.
// The list comes from GET /releases (highlights only); a release's full notes are fetched
// from GET /releases/{version} when it is opened.
const niceDate = d => {
  if (!d) return '';
  const t = new Date(`${d}T00:00:00`);
  return Number.isNaN(t.getTime()) ? d : t.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
};

export function Releases() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(null);       // set of expanded versions, once the list is in
  const [full, setFull] = useState({});         // version -> {loading, error, markdown}

  async function load() {
    setError('');
    try {
      const r = await api('/releases');
      setData(r);
      setOpen(s => s || new Set(r.releases.filter(x => x.current).map(x => x.version)));
    } catch (e) { setError(e.message || 'The release notes could not be loaded.'); }
  }
  useEffect(() => { load(); }, []);

  async function fetchFull(version) {
    setFull(f => ({ ...f, [version]: { loading: true } }));
    try {
      const r = await api(`/releases/${encodeURIComponent(version)}`);
      setFull(f => ({ ...f, [version]: { markdown: r.markdown } }));
    } catch (e) { setFull(f => ({ ...f, [version]: { error: e.message || 'These notes could not be loaded.' } })); }
  }

  function toggle(version) {
    const wasOpen = open.has(version);
    setOpen(s => { const n = new Set(s); wasOpen ? n.delete(version) : n.add(version); return n; });
    if (!wasOpen && !full[version]?.markdown && !full[version]?.loading) fetchFull(version);
  }

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (data?.releases || []).filter(r => !q || r.version.toLowerCase().includes(q) || (r.summary || '').toLowerCase().includes(q));
  }, [data, query]);

  if (error && !data) return <div className="operations"><div role="alert" className="inline-error">{error} <button type="button" className="link-button" onClick={load}>Try again</button></div></div>;
  if (!data) return <div className="operations"><Empty icon={Newspaper} title="Reading the release notes">One moment…</Empty></div>;

  const now = data.releases.find(r => r.current);
  return <div className="operations rn">
    <div className="page-intro rn-head">
      <h1>Release notes</h1>
      <p className="rn-running">Running <b>v{data.current}</b>{now?.date ? <> · released {niceDate(now.date)}</> : <> · not yet released</>}</p>
      <p>Every change to the application is versioned, and these are the notes for each release.</p>
    </div>

    <label className="rn-filter">
      <MagnifyingGlass size={17} aria-hidden="true"/>
      <input type="search" aria-label="Filter releases" placeholder="Filter by version or what changed…" value={query} onChange={e => setQuery(e.target.value)}/>
    </label>

    {shown.length === 0 && <Empty icon={Newspaper} title="No release matches">Try a version number such as 0.9 or a word from the summary.</Empty>}

    <ol className="rn-list">
      {shown.map(r => {
        const isOpen = open.has(r.version);
        const f = full[r.version];
        const panelId = `rn-${r.version}`;
        return <li key={r.version} className={`rn-item${r.current ? ' current' : ''}${isOpen ? ' open' : ''}`}>
          <button type="button" className="rn-row" aria-expanded={isOpen} aria-controls={panelId} onClick={() => toggle(r.version)}>
            <span className="rn-chip">v{r.version}</span>
            <span className="rn-date">{niceDate(r.date)}</span>
            {r.current && <span className="rn-now">running now</span>}
            <span className="rn-summary">{r.summary || 'No summary'}</span>
            <CaretDown size={16} className="rn-caret" aria-hidden="true"/>
          </button>
          {isOpen && <div className="rn-body" id={panelId}>
            {f?.loading && <p className="rn-status"><CircleNotch size={16} className="rn-spin"/> Loading the full notes…</p>}
            {f?.error && <p className="rn-status error" role="alert"><WarningCircle size={16}/> {f.error} <button type="button" className="link-button" onClick={() => fetchFull(r.version)}>Try again</button></p>}
            {f?.markdown
              ? <Markdown source={f.markdown}/>
              : <>
                  {!f?.loading && !f?.error && <>
                    <h3 className="rn-sub">Highlights</h3>
                    <Markdown source={r.highlights}/>
                    <button type="button" className="rn-more" onClick={() => fetchFull(r.version)}>{r.has_notes ? 'Read the full release notes' : 'Show the full changelog entry'}</button>
                  </>}
                </>}
          </div>}
        </li>;
      })}
    </ol>
  </div>;
}
