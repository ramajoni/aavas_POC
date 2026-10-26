import { sessionToken, endSession, problem } from './session';

// The deployment-local API key, for AAVAS_AUTH_MODE=apikey. Signed-in sessions
// (the default) send a bearer token from ./session instead.
const KEY_STORAGE = 'aavas.apiKey';
export const apiKey = () => { try { return localStorage.getItem(KEY_STORAGE) || ''; } catch { return ''; } };
export const setApiKey = value => {
  try { value ? localStorage.setItem(KEY_STORAGE, value) : localStorage.removeItem(KEY_STORAGE); }
  catch { /* private window: the key simply does not persist */ }
};

export class AuthRequiredError extends Error {
  constructor(message) { super(message); this.name = 'AuthRequiredError'; this.authRequired = true; }
}

// `body` is sent as JSON, or as multipart when it is FormData (file uploads).
export async function api(path, body, options = {}) {
  const key = apiKey();
  const token = sessionToken();
  const form = body instanceof FormData;
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { ...(form ? {} : { 'Content-Type': 'application/json' }), ...(token ? { Authorization: `Bearer ${token}` } : {}), ...(key ? { 'X-API-Key': key } : {}) },
    ...(body === undefined ? {} : { body: form ? body : JSON.stringify(body) }),
    ...options,
  });
  if (response.status === 401 || response.status === 403) {
    let detail = '';
    try { detail = (await response.json()).detail; } catch { /* no body */ }
    // A signed-in session the server no longer accepts has expired or been
    // revoked: end it, and the app returns to the sign-in screen.
    if (response.status === 401 && token) endSession();
    // Signed in but outside this role: an ordinary error the page can show.
    if (response.status === 403 && token) throw new Error(detail || problem(403, null));
    throw new AuthRequiredError(detail || 'This service requires authentication.');
  }
  let data;
  try { data = await response.json(); } catch { throw new Error(problem(response.status, null)); }
  if (!response.ok) {
    const detail = data.detail;
    const error = new Error(Array.isArray(detail) ? detail.map(e => `${e.loc?.slice(1).join(' ')}: ${e.msg}`).join(' · ') : typeof detail === 'string' ? detail : JSON.stringify(detail || data));
    error.status = response.status;
    throw error;
  }
  return data;
}
// Downloads a protected file with the session token and hands it to the browser
// as a save. A plain link cannot carry the token.
export async function downloadFile(path, filename) {
  const token = sessionToken();
  const response = await fetch(`/api${path}`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (response.status === 401 && token) endSession();
  if (!response.ok) throw new Error('This document could not be downloaded.');
  const url = URL.createObjectURL(await response.blob());
  const link = Object.assign(document.createElement('a'), { href: url, download: filename });
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
export const money = n => n == null ? '—' : new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(n);
export const number = n => n == null ? '—' : new Intl.NumberFormat('en-IN').format(n);
export const percent = (n, digits = 2) => n == null ? '—' : `${Number(n).toFixed(digits)}%`;
export const titleCase = s => String(s || '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
export const emi = (amount, rate, years) => { const r = rate / 1200; const n = years * 12; return r ? amount * r * (1+r)**n / ((1+r)**n - 1) : amount/n; };
export const newId = () => `APP${Date.now()}`;
// A new file starts empty: nothing is filled in that the customer or the RM did
// not enter. Choices with no sensible default start unselected too.
export const emptyApplicant = () => ({
  loan_id: newId(), name: '', mobile: '', email: '',
  segment: '', city: '', channel: 'branch_walkin', property_type: '',
  declared_monthly_income: '', requested_loan_amount: '', requested_tenure_years: '',
  applicant_age: '', declared_obligations: '', estimated_property_value: '',
  bureau_score: '', new_to_credit: false,
  sales_context: { competitor_rate: null, competitor_name: null, competitor_evidence_verified: false },
});
const preKeys = ['loan_id','segment','city','channel','property_type','declared_monthly_income','requested_loan_amount','requested_tenure_years','applicant_age','declared_obligations','estimated_property_value','bureau_score','new_to_credit','sales_context'];
const finalKeys = ['verified_monthly_income','appraised_property_value','sanctioned_loan_amount','tenure_years','bureau_vintage_months','delinquency_count_24m','banking_behaviour_score','deviation_count','legal_flag','technical_flag'];
const given = v => v !== '' && v != null;
// Until a figure is verified, pricing uses the customer's own declared figure -
// never a made-up one. Anything not given is left out for the service to treat as unknown.
const PROVISIONAL = { verified_monthly_income: 'declared_monthly_income', appraised_property_value: 'estimated_property_value',
  sanctioned_loan_amount: 'requested_loan_amount', tenure_years: 'requested_tenure_years' };
export const payload = (a, final = false) => {
  const out = {};
  for (const k of [...preKeys, ...(final ? finalKeys : [])]) {
    const v = given(a[k]) ? a[k] : (final && PROVISIONAL[k] ? a[PROVISIONAL[k]] : undefined);
    if (k === 'sales_context') out[k] = Object.fromEntries(Object.entries(v || {}).filter(([, x]) => given(x)));
    else if (given(v)) out[k] = v;
  }
  return out;
};
// Whether a file has been opened or entered at all.
export const hasFile = a => Boolean(a && String(a.name || '').trim() && given(a.requested_loan_amount));
