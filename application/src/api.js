// The deployment-local API key, when the service runs with AAVAS_AUTH_MODE=apikey.
// localStorage is appropriate for a key a developer pastes into their own browser
// and wrong for a real credential: an SSO session cookie replaces this entirely,
// and `apiKey`/`setApiKey` are the only two places that would change.
const KEY_STORAGE = 'aavas.apiKey';
export const apiKey = () => { try { return localStorage.getItem(KEY_STORAGE) || ''; } catch { return ''; } };
export const setApiKey = value => {
  try { value ? localStorage.setItem(KEY_STORAGE, value) : localStorage.removeItem(KEY_STORAGE); }
  catch { /* private window: the key simply does not persist */ }
};

export class AuthRequiredError extends Error {
  constructor(message) { super(message); this.name = 'AuthRequiredError'; this.authRequired = true; }
}

export async function api(path, body, options = {}) {
  const key = apiKey();
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json', ...(key ? { 'X-API-Key': key } : {}) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    ...options,
  });
  if (response.status === 401 || response.status === 403) {
    let detail = '';
    try { detail = (await response.json()).detail; } catch { /* no body */ }
    throw new AuthRequiredError(detail || 'This service requires authentication.');
  }
  let data;
  try { data = await response.json(); } catch { throw new Error('The pricing service is unavailable. Please check that the API is running on port 8000.'); }
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(Array.isArray(detail) ? detail.map(e => `${e.loc?.slice(1).join(' ')}: ${e.msg}`).join(' · ') : typeof detail === 'string' ? detail : JSON.stringify(detail || data));
  }
  return data;
}
export const money = n => n == null ? '—' : new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(n);
export const number = n => n == null ? '—' : new Intl.NumberFormat('en-IN').format(n);
export const percent = (n, digits = 2) => n == null ? '—' : `${Number(n).toFixed(digits)}%`;
export const titleCase = s => String(s || '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
export const emi = (amount, rate, years) => { const r = rate / 1200; const n = years * 12; return r ? amount * r * (1+r)**n / ((1+r)**n - 1) : amount/n; };
export const newId = () => `APP${Date.now()}`;
export const initialApplicant = () => ({
  loan_id: newId(), name: 'Amit Singh', mobile: '9876543210', email: 'amit.singh@example.com',
  segment: 'salaried_formal', city: 'Jaipur', channel: 'branch_walkin', property_type: 'resale_flat',
  declared_monthly_income: 55000, requested_loan_amount: 2000000, requested_tenure_years: 20,
  applicant_age: 35, declared_obligations: 5000, estimated_property_value: 3000000,
  bureau_score: 750, new_to_credit: false, verified_monthly_income: 53000,
  appraised_property_value: 2900000, sanctioned_loan_amount: 2000000, tenure_years: 20,
  bureau_vintage_months: 96, delinquency_count_24m: 0, banking_behaviour_score: 75,
  deviation_count: 0, legal_flag: 0, technical_flag: 0,
  sales_context: { competitor_rate: 11.5, competitor_name: 'Other housing finance company',
    competitor_evidence_verified: false, sales_target_pct: 85, branch_target_pct: 90,
    customer_negotiating: true, prior_negotiation_rounds: 1, existing_relationship: false,
    month_end: false, days_to_month_end: 15 },
});
const preKeys = ['loan_id','segment','city','channel','property_type','declared_monthly_income','requested_loan_amount','requested_tenure_years','applicant_age','declared_obligations','estimated_property_value','bureau_score','new_to_credit','sales_context'];
const finalKeys = ['verified_monthly_income','appraised_property_value','sanctioned_loan_amount','tenure_years','bureau_vintage_months','delinquency_count_24m','banking_behaviour_score','deviation_count','legal_flag','technical_flag'];
export const payload = (a, final = false) => Object.fromEntries([...preKeys, ...(final ? finalKeys : [])].map(k => [k, a[k]]));
