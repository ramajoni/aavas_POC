"""Data drift, model drift and concept drift between a reference and a current window.

    data drift     the inputs moved          PSI and KS / chi-square on every contract feature
    model drift    the predictions moved     PSI and KS on each champion's output, scored on both windows
    concept drift  the relationship moved    performance on outcomes against the model card's own
                                             out-of-sample figures, per window and per month, plus
                                             CUSUM change detection on each error stream, dated

Concept drift is judged against the card - the champion's performance on rows it never
saw - not against its fit on the rows it learned from: that comparison would read
ordinary overfitting as drift.

``inject`` simulates drift on a copy of the current window, so the detectors can be
shown to fire (and the tests can prove it): ``data_shift``, ``price_war``, ``default_surge``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..config import feature_config, load
from ..governance import model_registry
from ..ingestion import loader
from ..kernel.statistics import category_psi, cusum, ks, psi
from ..models import base as model_base

SCENARIOS = ("none", "data_shift", "price_war", "default_surge")
OUTPUTS = {
    "stage1_indicative_rate": "Stage-1 indicative rate (%)",
    "stage1_band_width_bps": "Stage-1 quote band width (bps)",
    "stage2_rate": "Stage-2 market-practice rate (%)",
    "conversion_probability": "Conversion probability at the sanctioned rate",
    "prepayment_probability": "Balance-transfer probability at the sanctioned rate",
    "risk_pd": "Annual probability of default",
    "override_probability": "Probability a human overrides the price",
}
_RANK = {"insufficient": -1, "ok": 0, "warn": 1, "alert": 2}


def cfg() -> dict[str, Any]:
    return load("monitoring")


# ------------------------------------------------------------------ windows


def windows(table: pd.DataFrame, conf: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference = the rows the champions were fitted on; current = the latest rows."""
    train, holdout = loader.train_test_split(table)
    if conf["windows"]["current"] == "last_days":
        dates = pd.to_datetime(table["application_date"])
        cut = dates.max() - pd.Timedelta(days=int(conf["windows"]["last_days"]))
        return train, table[dates > cut].copy()
    return train, holdout


def inject(current: pd.DataFrame, scenario: str, seed: int = 7) -> pd.DataFrame:
    """Simulate drift on a copy of the current window."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    cur = current.copy()
    rng = np.random.default_rng(seed)
    if scenario == "data_shift":                 # the mix of files changes; outcomes do not
        for col in ("declared_monthly_income", "verified_monthly_income"):
            if col in cur:
                cur[col] = cur[col] * 0.80
        scored = cur["bureau_score"] >= 0
        cur.loc[scored, "bureau_score"] = (cur.loc[scored, "bureau_score"] - 35).clip(lower=300)
        move = rng.random(len(cur)) < 0.30
        cur.loc[move, "segment"] = "self_employed_non_professional"
        cur.loc[move, "channel"] = "dsa"
    elif scenario == "price_war":                # competitors cut; the same price now converts worse
        cur["competitor_rate"] = cur["competitor_rate"] - 0.75
        cur["sanctioned_rate"] = (cur["sanctioned_rate"] - 0.40).clip(lower=13.0)
        priced_above = cur["sanctioned_rate"] > cur["competitor_rate"]
        lost = priced_above & cur["disbursed"].astype(bool) & (rng.random(len(cur)) < 0.40)
        cur.loc[lost, "disbursed"] = False
    elif scenario == "default_surge":            # the same files now default more often
        booked = cur["disbursed"].astype(bool) & ~cur["defaulted"].astype(bool)
        cur.loc[booked & (rng.random(len(cur)) < 0.06), "defaulted"] = True
    return cur


# ------------------------------------------------------------------ scoring


def predictions(bundle, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if bundle.stage1 is not None:
        s1 = bundle.stage1.predict_frame(df)
        out["stage1_indicative_rate"] = s1["indicative_rate"].to_numpy()
        out["stage1_band_width_bps"] = s1["band_width_bps"].to_numpy()
    if bundle.stage2 is not None:
        out["stage2_rate"] = np.asarray(bundle.stage2.predict_frame(df), dtype=float)
    rate = df["sanctioned_rate"].to_numpy(float)
    if bundle.conversion is not None:
        out["conversion_probability"] = np.asarray(bundle.conversion.predict_at_rate(df, rate), dtype=float)
    if bundle.prepayment is not None:
        out["prepayment_probability"] = np.asarray(bundle.prepayment.predict_at_rate(df, rate), dtype=float)
    if bundle.risk is not None:
        assessed = bundle.risk.assess(df)
        out["risk_pd"] = assessed["pd"].to_numpy(float)
        out["risk_lifetime_pd"] = assessed["lifetime_pd"].to_numpy(float)
    if bundle.override is not None:
        out["override_probability"] = bundle.override.predict(df)["override_probability"].to_numpy(float)
    return out


# ------------------------------------------------------------------ verdicts


def _status(th: dict, psi_value: float | None = None, ks_stat: float | None = None) -> str:
    worst = "ok"
    for value, warn, alert in ((psi_value, th["psi_warn"], th["psi_alert"]), (ks_stat, th["ks_warn"], th["ks_alert"])):
        if value is None or not np.isfinite(value):
            continue
        level = "alert" if value >= alert else "warn" if value >= warn else "ok"
        worst = max(worst, level, key=_RANK.get)
    return worst


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def data_drift(ref: pd.DataFrame, cur: pd.DataFrame, th: dict) -> list[dict]:
    fc = feature_config()
    numeric, categorical = [], []
    for contract in ("stage1", "stage2", "conversion"):
        numeric += list(fc[contract]["numeric"])
        categorical += list(fc[contract]["categorical"])
    numeric = [c for c in dict.fromkeys(numeric) if c in ref and c in cur]
    categorical = [c for c in dict.fromkeys(categorical) if c in ref and c in cur and c not in numeric]
    rows = []
    for col in numeric:
        p = psi(ref[col].to_numpy(), cur[col].to_numpy())
        stat, pval = ks(ref[col].to_numpy(), cur[col].to_numpy())
        rows.append({"kind": "data", "name": col, "type": "numeric", "psi": _r(p), "ks_stat": _r(stat),
                     "p_value": _r(pval, 6), "reference_value": _r(pd.to_numeric(ref[col], errors="coerce").mean()),
                     "current_value": _r(pd.to_numeric(cur[col], errors="coerce").mean()),
                     "status": _status(th, p, stat)})
    for col in categorical:
        p, pval = category_psi(ref[col], cur[col])
        top_ref = ref[col].astype(str).value_counts(normalize=True)
        top_cur = cur[col].astype(str).value_counts(normalize=True)
        lead = top_ref.index[0] if len(top_ref) else None
        rows.append({"kind": "data", "name": col, "type": "categorical", "psi": _r(p), "ks_stat": None,
                     "p_value": _r(pval, 6), "reference_value": _r(top_ref.get(lead, np.nan)),
                     "current_value": _r(top_cur.get(lead, np.nan)), "detail": f"share of '{lead}'",
                     "status": _status(th, p)})
    return sorted(rows, key=lambda r: (-_RANK[r["status"]], -(r["psi"] or 0)))


def model_drift(ref_pred: pd.DataFrame, cur_pred: pd.DataFrame, th: dict) -> list[dict]:
    rows = []
    for col, label in OUTPUTS.items():
        if col not in ref_pred or col not in cur_pred:
            continue
        p = psi(ref_pred[col].to_numpy(), cur_pred[col].to_numpy())
        stat, pval = ks(ref_pred[col].to_numpy(), cur_pred[col].to_numpy())
        rows.append({"kind": "model", "name": col, "label": label, "psi": _r(p), "ks_stat": _r(stat),
                     "p_value": _r(pval, 6), "reference_value": _r(ref_pred[col].mean()),
                     "current_value": _r(cur_pred[col].mean()), "status": _status(th, p, stat)})
    return sorted(rows, key=lambda r: (-_RANK[r["status"]], -(r["psi"] or 0)))


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def concept_drift(cur: pd.DataFrame, cur_pred: pd.DataFrame, cards: dict[str, dict], th: dict,
                  ph: dict) -> list[dict]:
    rows = []
    order = np.argsort(pd.to_datetime(cur["application_date"]).to_numpy())
    dates = pd.to_datetime(cur["application_date"]).to_numpy()[order]

    def detect(stream: np.ndarray, target: float, slack: float, threshold: float, side: str) -> str | None:
        if target is None or not np.isfinite(target):
            return None
        at = cusum(stream[order], target, slack, threshold, int(ph["min_observations"]), side)
        return None if at is None else str(pd.Timestamp(dates[at]).date())

    # Stage-2: rate error in bps against the card's holdout MAE.
    if "stage2_rate" in cur_pred:
        err = np.abs(cur_pred["stage2_rate"].to_numpy() - cur["sanctioned_rate"].to_numpy(float)) * 100
        ref = (cards.get("stage2") or {}).get("metrics", {}).get("mae_bps")
        change = None if not ref else 100 * (err.mean() - ref) / ref
        status = "ok" if change is None else ("alert" if change >= th["mae_increase_alert_pct"]
                                              else "warn" if change >= th["mae_increase_warn_pct"] else "ok")
        changed_at = detect(err, ref, ph["error_slack"] * (ref or 0), ph["error_threshold"] * (ref or 0), "upper")
        if changed_at and status == "ok":
            status = "warn"
        rows.append({"kind": "concept", "name": "stage2_mae_bps", "model": "stage2", "reference_value": _r(ref, 3),
                     "current_value": _r(err.mean(), 3), "change_pct": _r(change, 1), "change_detected_at": changed_at,
                     "status": status, "detail": "rate error against the sanctioned rate, bps"})

    # Conversion: ranking (AUC) and calibration (observed minus predicted disbursal).
    if "conversion_probability" in cur_pred:
        y = cur["disbursed"].astype(bool).to_numpy()
        p = cur_pred["conversion_probability"].to_numpy()
        auc, ref_auc = _auc(y, p), (cards.get("conversion") or {}).get("metrics", {}).get("auc")
        drop = None if ref_auc is None or not np.isfinite(auc) else ref_auc - auc
        rows.append({"kind": "concept", "name": "conversion_auc", "model": "conversion", "reference_value": _r(ref_auc),
                     "current_value": _r(auc), "change": _r(-drop if drop is not None else None),
                     "status": "ok" if drop is None else ("alert" if drop >= th["auc_drop_alert"]
                                                          else "warn" if drop >= th["auc_drop_warn"] else "ok")})
        gap = 100 * (y.mean() - p.mean())
        changed_at = detect(y.astype(float) - p, 0.0, ph["residual_slack"], ph["residual_threshold"], "both")
        status = "alert" if abs(gap) >= th["calibration_gap_alert_pts"] else \
            "warn" if abs(gap) >= th["calibration_gap_warn_pts"] else "ok"
        rows.append({"kind": "concept", "name": "conversion_calibration_gap_pts", "model": "conversion",
                     "reference_value": 0.0, "current_value": _r(gap, 2), "change_detected_at": changed_at,
                     "status": status, "detail": "observed minus predicted disbursal rate, points"})

    # Prepayment (balance transfer) ranking, among disbursed loans.
    if "prepayment_probability" in cur_pred and "bt_out_24m" in cur:
        booked = cur["disbursed"].astype(bool).to_numpy()
        auc = _auc(cur["bt_out_24m"].astype(bool).to_numpy()[booked], cur_pred["prepayment_probability"].to_numpy()[booked])
        ref_auc = (cards.get("prepayment") or {}).get("metrics", {}).get("auc")
        drop = None if ref_auc is None or not np.isfinite(auc) else ref_auc - auc
        rows.append({"kind": "concept", "name": "prepayment_auc", "model": "prepayment", "reference_value": _r(ref_auc),
                     "current_value": _r(auc), "change": _r(-drop if drop is not None else None),
                     "status": "ok" if drop is None else ("alert" if drop >= th["auc_drop_alert"]
                                                          else "warn" if drop >= th["auc_drop_warn"] else "ok")})

    # Risk: observed defaults against what the PD model expected, among booked loans.
    if "risk_lifetime_pd" in cur_pred and "defaulted" in cur:
        booked = cur["disbursed"].astype(bool).to_numpy()
        observed = cur["defaulted"].astype(bool).to_numpy()[booked].mean()
        expected = cur_pred["risk_lifetime_pd"].to_numpy()[booked].mean()
        ratio = observed / expected if expected else float("nan")
        lo_a, hi_a = th["default_ratio_alert"]
        lo_w, hi_w = th["default_ratio_warn"]
        status = "alert" if not lo_a <= ratio <= hi_a else "warn" if not lo_w <= ratio <= hi_w else "ok"
        rows.append({"kind": "concept", "name": "risk_observed_to_expected_defaults", "model": "risk",
                     "reference_value": 1.0, "current_value": _r(ratio, 3), "status": status,
                     "detail": f"observed {observed:.2%} against expected {expected:.2%} lifetime default"})
    return rows


def monthly(cur: pd.DataFrame, cur_pred: pd.DataFrame) -> list[dict]:
    frame = cur.assign(_month=pd.to_datetime(cur["application_date"]).dt.to_period("M").dt.to_timestamp())
    rows = []
    for month, idx in frame.groupby("_month").groups.items():
        part, pred = frame.loc[idx], cur_pred.loc[idx]
        booked = part["disbursed"].astype(bool)
        row = {"month": str(month.date()), "n": int(len(part)),
               "sanctioned_rate_mean": _r(part["sanctioned_rate"].mean(), 3),
               "competitor_rate_mean": _r(part["competitor_rate"].mean(), 3),
               "income_mean": _r(part["verified_monthly_income"].mean(), 0),
               "observed_conversion_pct": _r(100 * booked.mean(), 2)}
        if "stage2_rate" in pred:
            row["stage2_mae_bps"] = _r((np.abs(pred["stage2_rate"] - part["sanctioned_rate"]) * 100).mean(), 3)
        if "conversion_probability" in pred:
            row["predicted_conversion_pct"] = _r(100 * pred["conversion_probability"].mean(), 2)
        if "risk_lifetime_pd" in pred and booked.any():
            row["default_observed_to_expected"] = _r(part.loc[booked, "defaulted"].astype(bool).mean()
                                                     / max(pred.loc[booked, "risk_lifetime_pd"].mean(), 1e-9), 3)
        rows.append(row)
    return rows


def _overall(findings: list[dict], th: dict, n_cur: int) -> str:
    if n_cur < th["min_rows"]:
        return "insufficient"
    return max((f["status"] for f in findings), key=_RANK.get, default="ok")


def report(scenario: str = "none", bundle=None, table: pd.DataFrame | None = None) -> dict[str, Any]:
    conf = cfg()
    th, ph = conf["thresholds"], conf["change_detection"]
    table = loader.load_training_table() if table is None else table
    ref, cur = windows(table, conf)
    cur = inject(cur, scenario) if scenario != "none" else cur
    bundle = bundle or model_registry.load_bundle()
    sample = ref.sample(min(len(ref), int(conf["windows"]["reference_sample"])), random_state=11)
    ref_pred, cur_pred = predictions(bundle, sample), predictions(bundle, cur)
    cards = {}
    for path in model_base.registry_dir().glob("*.card.json"):
        import json
        card = json.loads(path.read_text(encoding="utf-8"))
        cards[card.get("name")] = card

    data = data_drift(ref, cur, th)
    model = model_drift(ref_pred, cur_pred, th)
    concept = concept_drift(cur, cur_pred, cards, th, ph)
    status = {"data": _overall(data, th, len(cur)), "model": _overall(model, th, len(cur)),
              "concept": _overall(concept, th, len(cur))}
    status["overall"] = max(status.values(), key=_RANK.get)
    d = pd.to_datetime
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": scenario, "config_version": conf.get("monitoring_config_version"),
        "windows": {"reference": {"from": str(d(ref["application_date"]).min().date()),
                                  "to": str(d(ref["application_date"]).max().date()), "n": int(len(ref))},
                    "current": {"from": str(d(cur["application_date"]).min().date()),
                                "to": str(d(cur["application_date"]).max().date()), "n": int(len(cur))}},
        "status": status, "thresholds": th,
        "data": data, "model": model, "concept": concept, "monthly": monthly(cur, cur_pred),
        "counts": {k: {s: sum(1 for f in v if f["status"] == s) for s in ("ok", "warn", "alert")}
                   for k, v in (("data", data), ("model", model), ("concept", concept))},
    }


def reports_dir():
    from ..paths import REPORTS_DIR
    return REPORTS_DIR / "drift"


def save(report_: dict[str, Any]):
    """Keep the report as reports/drift/<run_id>.json; a run on the live book (no simulated
    scenario) also becomes latest.json, the one model risk reads. Returns the folder."""
    import json
    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report_, indent=1, default=str)
    (out / f"{report_['run_id']}.json").write_text(text, encoding="utf-8")
    if report_.get("scenario", "none") == "none":
        (out / "latest.json").write_text(text, encoding="utf-8")
    return out
