// The signed-in user, shared by the lending workspace and the OS console.
//
// The token is a bearer token from POST /auth/login. It lives in localStorage so
// the workspace, the OS console and a loan explanation opened in its own window
// all share one sign-in. The server decides what each role may reach; `canOpen`
// only decides what the browser bothers to show.
const TOKEN_KEY = 'aavas.session';
const USER_KEY = 'aavas.user';
export const SESSION_ENDED = 'aavas:session-ended';

const read = key => { try { return localStorage.getItem(key); } catch { return null; } };
const write = (key, value) => {
  try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); }
  catch { /* private window: the session lasts as long as the page */ }
};

let memory = { token: read(TOKEN_KEY), user: (() => { try { return JSON.parse(read(USER_KEY)); } catch { return null; } })() };

export const sessionToken = () => memory.token;
// For the few requests that cannot go through api(): raw text and image fetches.
export const authHeaders = () => (memory.token ? { Authorization: `Bearer ${memory.token}` } : {});
export const sessionUser = () => memory.user;

export function startSession(token, user) {
  memory = { token, user };
  write(TOKEN_KEY, token); write(USER_KEY, JSON.stringify(user));
}

export function endSession() {
  const had = Boolean(memory.token);
  memory = { token: null, user: null };
  write(TOKEN_KEY, null); write(USER_KEY, null);
  if (had) window.dispatchEvent(new Event(SESSION_ENDED));
}

async function call(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(memory.token ? { Authorization: `Bearer ${memory.token}` } : {}), ...options.headers },
  });
  let body = null;
  try { body = await response.json(); } catch { /* no body */ }
  if (!response.ok) {
    const error = new Error(problem(response.status, body));
    error.status = response.status;
    throw error;
  }
  return body;
}

// What went wrong, in words a borrower or a branch can act on. A 403 that is not
// the API's own JSON comes from the firewall in front of it, which limits how many
// sign-in attempts one network may make.
export function problem(status, body) {
  const detail = body?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map(d => d.msg).filter(Boolean).join('; ') || 'Please check the details and try again.';
  if (status === 403 || status === 429) return 'Too many attempts from your network. Please wait a few minutes and try again.';
  if (status >= 500 || status === 0) return 'The pricing service is not responding right now. Please try again in a minute.';
  return 'Something went wrong. Please try again.';
}

export async function signIn(username, password) {
  const body = await call('/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) });
  startSession(body.token, body.user);
  return body.user;
}

// A borrower's own account: created, then signed in, in one step.
export async function register(fields) {
  const body = await call('/auth/register', { method: 'POST', body: JSON.stringify(fields) });
  startSession(body.token, body.user);
  return body.user;
}

// A customer replaces the temporary password the branch gave them with their own.
export async function changePassword(currentPassword, newPassword) {
  const body = await call('/auth/password', { method: 'POST',
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }) });
  startSession(memory.token, body.user);
  return body.user;
}

export const registrationOpen = () => call('/auth/registration').then(b => b.borrowers).catch(() => false);

export async function signOut() {
  try { await call('/auth/logout', { method: 'POST' }); } catch { /* signing out locally is what matters */ }
  endSession();
}

// Confirms a stored session is still good, and picks up any role change made on
// the server since sign-in.
export async function refreshUser() {
  if (!memory.token) return null;
  try {
    const { user } = await call('/auth/me');
    startSession(memory.token, user);
    return user;
  } catch (e) {
    if (e.status === 401) { endSession(); return null; }
    return memory.user;
  }
}

export const personas = () => call('/auth/personas').then(b => b.personas);

export const canOpen = (user, page) => Boolean(user && (user.all_pages || user.pages.includes(page)));
