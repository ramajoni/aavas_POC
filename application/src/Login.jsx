import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, CircleNotch, WarningCircle } from '@phosphor-icons/react';
import { personas as loadPersonas, signIn } from './session';
import './login.css';

// The sign-in screen: one card per role, as in the wireframe. Choosing a card
// asks for that person's password; nothing past this screen renders without a
// session. `consoleOnly` narrows the cards to roles that may open the OS console.
export function Login({ onSignedIn, consoleOnly = false, title = 'Home Loan Pricing' }) {
  const [people, setPeople] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [active, setActive] = useState(null);       // a username, or 'other'
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  function load() {
    setLoadError('');
    loadPersonas()
      .then(list => setPeople(consoleOnly ? list.filter(p => p.console) : list))
      .catch(() => setLoadError('The sign-in service is unavailable. Check that the pricing API is running on port 8000.'));
  }
  useEffect(load, [consoleOnly]);

  function choose(key) { setActive(current => (current === key ? null : key)); setError(''); }

  async function submit(username, password) {
    setBusy(true); setError('');
    try { onSignedIn(await signIn(username, password)); }
    catch (e) { setError(e.status === 401 ? e.message : 'The sign-in service is unavailable. Please try again.'); }
    finally { setBusy(false); }
  }

  return <div className="login-page">
    <main className="login-main">
      <div className="login-logo"><img src="/assets/aavas-logo.png" alt="Aavas Financiers"/></div>
      <h1>{title}</h1>
      <p className="login-lead">Choose a role to sign in.</p>

      {loadError && <div className="login-alert" role="alert"><WarningCircle size={20}/><span>{loadError}</span>
        <button type="button" onClick={load}>Try again</button></div>}
      {!people && !loadError && <p className="login-lead"><CircleNotch className="spin" size={18}/> Loading…</p>}

      {people && <div className="login-grid">
        {people.map(p => <PersonaCard key={p.username} person={p} open={active === p.username}
          onChoose={() => choose(p.username)} onSubmit={password => submit(p.username, password)}
          busy={busy} error={active === p.username ? error : ''}/>)}
      </div>}

      {people && <div className="login-other">
        {active === 'other'
          ? <OtherSignIn onSubmit={submit} onCancel={() => choose('other')} busy={busy} error={error}/>
          : <button type="button" className="login-other-link" onClick={() => choose('other')}>
              Sign in with a username instead</button>}
      </div>}
    </main>
  </div>;
}

function PersonaCard({ person, open, onChoose, onSubmit, busy, error }) {
  const [password, setPassword] = useState('');
  const input = useRef(null);
  const first = person.name.split(' ')[0];
  useEffect(() => { if (open) input.current?.focus(); else setPassword(''); }, [open]);

  return <section className={`login-card ${open ? 'open' : ''}`}>
    <button type="button" className="login-card-head" onClick={onChoose} aria-expanded={open}>
      <span className="login-avatar" aria-hidden="true">{person.initials}</span>
      <span><small>{person.title}</small><b>{person.name}</b></span>
    </button>
    <p>{person.description}</p>
    {open
      ? <form className="login-form" onSubmit={e => { e.preventDefault(); onSubmit(password); }}
          onKeyDown={e => { if (e.key === 'Escape') onChoose(); }}>
          <input type="text" name="username" value={person.username} autoComplete="username" readOnly hidden/>
          <label><span>Password for {person.name}</span>
            <input ref={input} type="password" value={password} autoComplete="current-password" required
              onChange={e => setPassword(e.target.value)} aria-invalid={Boolean(error)}
              aria-describedby={error ? `login-error-${person.username}` : undefined}/></label>
          {error && <p className="login-error" id={`login-error-${person.username}`} role="alert">{error}</p>}
          <div className="login-form-actions">
            <button type="submit" className="login-submit" disabled={busy || !password}>
              {busy ? <CircleNotch className="spin" size={17}/> : null} Sign in as {first}</button>
            <button type="button" className="login-cancel" onClick={onChoose}>Cancel</button>
          </div>
        </form>
      : <button type="button" className="login-link" onClick={onChoose}>Sign in as {first} <ArrowRight size={15}/></button>}
  </section>;
}

function OtherSignIn({ onSubmit, onCancel, busy, error }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  return <form className="login-card login-form other open" onSubmit={e => { e.preventDefault(); onSubmit(username, password); }}>
    <label><span>Username</span>
      <input autoFocus value={username} autoComplete="username" required onChange={e => setUsername(e.target.value)}/></label>
    <label><span>Password</span>
      <input type="password" value={password} autoComplete="current-password" required onChange={e => setPassword(e.target.value)}/></label>
    {error && <p className="login-error" role="alert">{error}</p>}
    <div className="login-form-actions">
      <button type="submit" className="login-submit" disabled={busy || !username || !password}>
        {busy ? <CircleNotch className="spin" size={17}/> : null} Sign in</button>
      <button type="button" className="login-cancel" onClick={onCancel}>Cancel</button>
    </div>
  </form>;
}

// Shown when someone signed in to a surface their role does not include.
export function NoAccess({ user, onSignOut, children }) {
  return <div className="login-page">
    <main className="login-main">
      <div className="login-logo"><img src="/assets/aavas-logo.png" alt="Aavas Financiers"/></div>
      <h1>Not available for your role</h1>
      <p className="login-lead">You are signed in as {user.name} ({user.role_label}). {children}</p>
      <div className="login-form-actions">
        <a className="login-submit" href="/">Go to your workspace</a>
        <button type="button" className="login-cancel light" onClick={onSignOut}>Sign out</button>
      </div>
    </main>
  </div>;
}
