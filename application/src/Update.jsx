import React, { useEffect, useState } from 'react';
import { ArrowClockwise } from '@phosphor-icons/react';

// The page a browser already has open keeps running the code it loaded, so a change
// published since stays invisible until a reload. Compare the script the hosted page
// now names with the one running, on focus and every two minutes, and offer the reload
// (never force it: a half-filled form would be lost).
const running = () => document.querySelector('script[src*="/assets/main-"]')?.getAttribute('src');

export function UpdateBanner() {
  const [stale, setStale] = useState(false);
  useEffect(() => {
    const mine = running();
    if (!mine) return undefined;                       // the development server: nothing to compare
    const check = () => fetch('/', { cache: 'no-store' }).then(r => r.text()).then(html => {
      const live = html.match(/\/assets\/main-[A-Za-z0-9_-]+\.js/)?.[0];
      if (live && !mine.endsWith(live)) setStale(true);
    }).catch(() => {});
    const timer = setInterval(check, 120000);
    window.addEventListener('focus', check);
    return () => { clearInterval(timer); window.removeEventListener('focus', check); };
  }, []);
  if (!stale) return null;
  return <div className="update-banner" role="status">
    <span>This page has been updated since you opened it.</span>
    <button type="button" onClick={() => location.reload()}><ArrowClockwise size={15}/> Reload to see the latest</button>
  </div>;
}
