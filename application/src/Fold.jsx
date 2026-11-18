import React, { useState } from 'react';
import { CaretDown } from '@phosphor-icons/react';

// A section that shows its one-line summary and opens on click: the quick read
// first, everything else a click away.
export function Fold({ title, summary, children, open: start = false }) {
  const [open, setOpen] = useState(start);
  return <section className={`rv-card rv-fold${open ? ' open' : ''}`}>
    <button type="button" className="rv-fold-head" aria-expanded={open} onClick={() => setOpen(!open)}>
      <span><b>{title}</b>{summary && <small>{summary}</small>}</span><CaretDown size={18} aria-hidden="true"/></button>
    {open && <div className="rv-fold-body">{children}</div>}
  </section>;
}
