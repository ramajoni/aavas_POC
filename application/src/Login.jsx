import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, CircleNotch, WarningCircle } from '@phosphor-icons/react';
import { changePassword, personas as loadPersonas, signIn, register, registrationOpen } from './session';
import './login.css';

// The sign-in screen: one card per role, as in the wireframe. Choosing a card
// asks for that person's password; nothing past this screen renders without a
// session. `consoleOnly` narrows the cards to roles that may open the OS console.
export function Login({ onSignedIn, consoleOnly = false, title = 'Home Loan Pricing' }) {
  const [people, setPeople] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [active, setActive] = useState(null);       // a username, 'other' or 'register'
  const [canRegister, setCanRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  function load() {
    setLoadError('');
    loadPersonas()
      .then(list => setPeople(consoleOnly ? list.filter(p => p.console) : list))
      .catch(e => setLoadError(e.status === 403 ? e.message : 'The sign-in service is not responding right now. Please try again in a minute.'));
  }
  useEffect(load, [consoleOnly]);
  useEffect(() => { if (!consoleOnly) registrationOpen().then(setCanRegister); }, [consoleOnly]);

  function choose(key) { setActive(current => (current === key ? null : key)); setError(''); }

  async function submit(username, password) {
    setBusy(true); setError('');
    try { onSignedIn(await signIn(username, password)); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function createAccount(fields) {
    setBusy(true); setError('');
    try { onSignedIn(await register(fields)); }
    catch (e) { setError(e.message); }
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
          : active === 'register'
            ? <RegisterForm onSubmit={createAccount} onCancel={() => choose('register')} busy={busy} error={error}/>
            : <div className="login-other-links">
                {canRegister && <button type="button" className="login-register" onClick={() => choose('register')}>
                  New borrower? Create an account</button>}
                <button type="button" className="login-other-link" onClick={() => choose('other')}>
                  Sign in with a username instead</button>
              </div>}
      </div>}
      <p className="login-provider">Solution provider: <a href="https://www.ajonitech.com/" target="_blank" rel="noopener noreferrer">Ajoni Technologies</a></p>
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

// The same rules the server applies (access.username_problem), so the person
// sees the problem as they type rather than after pressing the button.
const suggestUsername = text => (text.toLowerCase().match(/[a-z0-9]+/g) || []).join('.').slice(0, 64);
function usernameProblem(raw) {
  const u = raw.trim().toLowerCase();
  if (!u) return '';
  if (/\s/.test(u)) return `A username cannot contain spaces: try ${suggestUsername(u)}`;
  if (u.includes('@')) return /^[a-z0-9._-]+@[a-z0-9.-]+\.[a-z]{2,}$/.test(u) ? '' : 'That email address is not valid as a username';
  if (u.length < 3) return 'A username needs at least 3 characters';
  const bad = [...new Set(u.replace(/[a-z0-9._-]/g, ''))];
  if (bad.length) return `Use only letters, digits, dots, dashes and underscores (not ${bad.join(' ')}), or your email address`;
  if (!/^[a-z0-9]/.test(u)) return 'A username must start with a letter or a digit';
  return '';
}
const passwordProblem = p => !p ? '' : p.length < 8 ? 'At least 8 characters' : !/[A-Za-z]/.test(p) || !/\d/.test(p) ? 'Include a letter and a digit' : '';

// A borrower's own account. Staff accounts are created by the bank, never here.
function RegisterForm({ onSubmit, onCancel, busy, error }) {
  const [f, setF] = useState({ name: '', mobile: '', email: '', username: '', password: '', confirm: '' });
  const [touched, setTouched] = useState(false);        // has the person typed a username themselves?
  const [mismatch, setMismatch] = useState(false);
  const set = key => e => {
    const value = e.target.value;
    setF(v => {
      const next = { ...v, [key]: value };
      // Until a username is typed, offer one built from the name.
      if (key === 'name' && !touched) next.username = suggestUsername(value);
      return next;
    });
    if (key === 'username') setTouched(true);
    setMismatch(false);
  };
  const uProblem = usernameProblem(f.username), pProblem = passwordProblem(f.password);
  function submit(e) {
    e.preventDefault();
    if (uProblem || pProblem) return;
    if (f.password !== f.confirm) { setMismatch(true); return; }
    const { confirm, ...fields } = f;
    onSubmit({ ...fields, username: fields.username.trim().toLowerCase(), email: fields.email || null });
  }
  return <form className="login-card login-form register open" onSubmit={submit}>
    <div className="login-register-head"><b>Create your borrower account</b>
      <small>Apply for a home loan, see your indicative offer and upload your documents.</small></div>
    <div className="login-register-grid">
      <label><span>Full name</span><input autoFocus value={f.name} autoComplete="name" required onChange={set('name')}/></label>
      <label><span>Mobile number</span><input value={f.mobile} inputMode="tel" autoComplete="tel-national" placeholder="10 digits" required onChange={set('mobile')}/></label>
      <label><span>Email (optional)</span><input type="email" value={f.email} autoComplete="email" onChange={set('email')}/></label>
      <label><span>Username</span><input value={f.username} autoComplete="username" placeholder="e.g. priya.menon" required onChange={set('username')} aria-invalid={Boolean(uProblem)}/>
        {uProblem ? <small className="login-field-error" role="alert">{uProblem}</small> : <small>Letters, digits, dots, dashes or underscores, or your email address.</small>}</label>
      <label><span>Password</span><input type="password" value={f.password} autoComplete="new-password" required minLength={8} onChange={set('password')} aria-invalid={Boolean(pProblem)}/>
        {pProblem ? <small className="login-field-error" role="alert">{pProblem}</small> : <small>At least 8 characters, with a letter and a digit.</small>}</label>
      <label><span>Confirm password</span><input type="password" value={f.confirm} autoComplete="new-password" required onChange={set('confirm')} aria-invalid={mismatch}/></label>
    </div>
    {mismatch && <p className="login-error" role="alert">The passwords do not match.</p>}
    {error && <p className="login-error" role="alert">{error}</p>}
    <div className="login-form-actions">
      <button type="submit" className="login-submit" disabled={busy || Boolean(uProblem || pProblem)}>{busy ? <CircleNotch className="spin" size={17}/> : null} Create account</button>
      <button type="button" className="login-cancel" onClick={onCancel}>Cancel</button>
    </div>
  </form>;
}

// A customer signing in with the temporary password the branch gave them sets
// their own before anything else; the server allows nothing else until then.
export function SetPassword({ user, onDone, onSignOut }) {
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [again, setAgain] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const weak = next && (next.length < 8 || !/[A-Za-z]/.test(next) || !/\d/.test(next));
  async function submit(e) {
    e.preventDefault();
    if (next !== again) { setError('The two new passwords do not match.'); return; }
    setBusy(true); setError('');
    try { onDone(await changePassword(current, next)); }
    catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }
  return <div className="login-page">
    <main className="login-main login-narrow">
      <div className="login-logo"><img src="/assets/aavas-logo.png" alt="Aavas Financiers"/></div>
      <h1>Set your password</h1>
      <p className="login-lead">Welcome, {user.name}. Your relationship manager gave you a temporary password for {user.username}. Choose your own to continue.</p>
      <form className="login-card login-form open" onSubmit={submit} autoComplete="off">
        <label><span>Temporary password</span>
          <input type="password" value={current} autoComplete="current-password" required autoFocus onChange={e => setCurrent(e.target.value)}/></label>
        <label><span>New password</span>
          <input type="password" value={next} autoComplete="new-password" required aria-invalid={Boolean(weak)} onChange={e => setNext(e.target.value)}/></label>
        <small className="login-hint">At least 8 characters, with a letter and a digit.</small>
        <label><span>New password again</span>
          <input type="password" value={again} autoComplete="new-password" required onChange={e => setAgain(e.target.value)}/></label>
        {error && <p className="login-error" role="alert">{error}</p>}
        <div className="login-form-actions">
          <button type="submit" className="login-submit" disabled={busy || !current || !next || weak || !again}>
            {busy ? <CircleNotch className="spin" size={17}/> : null} Save and continue</button>
          <button type="button" className="login-cancel" onClick={onSignOut}>Sign out</button>
        </div>
      </form>
      <p className="login-provider">Solution provider: <a href="https://www.ajonitech.com/" target="_blank" rel="noopener noreferrer">Ajoni Technologies</a></p>
    </main>
  </div>;
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
      <p className="login-provider">Solution provider: <a href="https://www.ajonitech.com/" target="_blank" rel="noopener noreferrer">Ajoni Technologies</a></p>
    </main>
  </div>;
}
