"""Kestrel claim-review service.

POST /score   one claim as JSON -> fraud probability, expected rupees, decision, reasons
GET  /        the review screen
GET  /health  liveness + which snapshot is loaded

No external APIs. Needs artifacts/ produced by `python src/train.py`.
"""
import json
import sys
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402

ART = ROOT / "artifacts"
if not (ART / "model.joblib").exists():
    raise SystemExit("Model not found. Put the CSVs in ./data and run: python src/train.py")

BUNDLE = joblib.load(ART / "model.joblib")
MODEL, FEATS, MEDIANS = BUNDLE["model"], BUNDLE["features"], BUNDLE["medians"]
SNAP = json.load(open(ART / "partner_snapshot.json"))
PARTNERS = SNAP["partners"]
PRODUCTS = pd.read_csv(ROOT / "data" / "products.csv").set_index("sku")

app = FastAPI(title="Kestrel claim review")


class Claim(BaseModel):
    claim_id: Optional[str] = None
    submitted_at: Optional[str] = Field(None, description="IST, e.g. 2026-09-14 10:30")
    partner_id: str
    sku: str
    product_serial: Optional[str] = None
    days_since_purchase: int = Field(ge=0, le=2000)
    claim_amount_inr: float = Field(gt=0, le=200000)
    photo_attached: Literal["Y", "N"] = "N"
    partner_inspected: Literal["Y", "N"] = "N"
    claim_description: Optional[str] = None
    inspector_note: Optional[str] = None
    customer_prior_claims: int = Field(0, ge=0, le=50)


# Plain-English reason for each feature when it pushes the score up.
REASONS = {
    "p_hist_rate": lambda r: f"{r['_pid']} has {r['p_hist_frauds']:.0f} past claims confirmed as fraud "
                             f"({r['p_hist_rate']:.0%} of its decided claims, smoothed)",
    "small_uninspected": lambda r: "Under Rs 2,000 and not inspected, so it would be auto-approved",
    "near_threshold": lambda r: "Amount sits just under the Rs 2,000 auto-approve limit",
    "p_small_uninsp_60d": lambda r: f"{r['_pid']} sent {r['p_small_uninsp_60d']:.0f} uninspected small "
                                    f"claims in the last 60 days",
    "p_near_thr_60d": lambda r: f"{r['_pid']} sent {r['p_near_thr_60d']:.0f} claims priced just under "
                                f"Rs 2,000 in the last 60 days",
    "p_growth": lambda r: "This partner's claim volume has jumped compared with its usual pace",
    "p_hist_frauds": lambda r: REASONS["p_hist_rate"](r),
    "p_claims_60d": lambda r: f"{r['_pid']} sent {r['p_claims_60d']:.0f} claims in the last 60 days",
    "amount_to_price": lambda r: f"Claim is {r['amount_to_price']:.0%} of the product's list price",
    "log_amount": lambda r: f"Large claim amount (Rs {r['_amt']:,.0f})",
    "prior_claims": lambda r: f"Customer has {r['prior_claims']:.0f} earlier warranty claims",
    "inspected": lambda r: "No inspection sign-off",
    "note_blank": lambda r: "No inspector note",
    "photo": lambda r: "No photo attached",
    "pt_franchise": lambda r: "Franchise partners have a higher historical fraud rate",
    "warranty_used": lambda r: f"Claimed {r['days_since_purchase']:.0f} days after purchase",
    "days_since_purchase": lambda r: f"Claimed {r['days_since_purchase']:.0f} days after purchase",
    "post_policy": lambda r: "Submitted after the May 2026 auto-approve change",
    "pt_freelance": lambda r: "Partner type",
}


def to_features(c: Claim):
    if c.sku not in PRODUCTS.index:
        raise HTTPException(422, f"Unknown SKU '{c.sku}'. Check the product code on the claim.")
    prod = PRODUCTS.loc[c.sku]
    part = PARTNERS.get(c.partner_id)
    notes = []
    if part is None:
        notes.append(f"Partner {c.partner_id} is not in the partner list; scored with average partner history.")
        part = dict(partner_type="unknown", p_claims_60d=0, p_small_uninsp_60d=0, p_near_thr_60d=0,
                    p_growth=0, p_hist_frauds=0, p_hist_rate=0.012)
    ts = pd.Timestamp(c.submitted_at) if c.submitted_at else pd.Timestamp.now()
    amt, insp = float(c.claim_amount_inr), c.partner_inspected == "Y"
    row = dict(
        log_amount=np.log1p(amt), amount_to_price=amt / prod.list_price_inr,
        near_threshold=int(1900 <= amt < F.AUTO_APPROVE_LIMIT),
        small_uninspected=int(amt < F.AUTO_APPROVE_LIMIT and not insp),
        post_policy=int(ts >= F.POLICY_CHANGE), photo=int(c.photo_attached == "Y"), inspected=int(insp),
        note_blank=int(not (c.inspector_note or "").strip()),
        warranty_used=c.days_since_purchase / (prod.warranty_months * 30.4),
        days_since_purchase=c.days_since_purchase, prior_claims=c.customer_prior_claims,
        pt_franchise=int(part["partner_type"] == "franchise"),
        pt_freelance=int(part["partner_type"] == "freelance_technician"),
        **{k: part[k] for k in ["p_claims_60d", "p_small_uninsp_60d", "p_near_thr_60d", "p_growth",
                                "p_hist_frauds", "p_hist_rate"]},
    )
    if c.days_since_purchase > prod.warranty_months * 30.4:
        notes.append("Claim is outside the standard warranty period for this product (check Kestrel Shield).")
    return pd.DataFrame([row])[FEATS], notes, amt


def explain(X, p, amt, pid, k=3):
    """Which features moved this claim's score most? Reset each to a typical value and re-score."""
    probes = []
    for f in FEATS:
        Xb = X.copy(); Xb[f] = MEDIANS[f]
        probes.append(Xb)
    base = MODEL.predict_proba(pd.concat(probes))[:, 1]
    deltas = sorted(zip(FEATS, p - base), key=lambda t: -t[1])
    r = X.iloc[0].to_dict(); r.update(_pid=pid, _amt=amt)
    out, seen = [], set()
    for f, d in deltas:
        if d <= 0.002 or len(out) >= k:
            break
        text = REASONS.get(f, lambda r: f)(r)
        if text not in seen:
            seen.add(text); out.append(dict(reason=text, effect=round(float(d), 3)))
    return out


@app.post("/score")
def score(c: Claim):
    X, notes, amt = to_features(c)
    p = float(MODEL.predict_proba(X)[:, 1][0])
    ev = float(F.expected_saving(p, amt))
    decision = "review" if ev > 0 else "pay"
    reasons = explain(X, p, amt, c.partner_id) if p >= 0.03 else []
    pct = "under 1%" if p < 0.01 else f"about {p:.0%}"
    summary = (f"Send to the investigation desk: {pct} chance of fraud; reviewing it saves "
               f"Rs {ev:,.0f} on average." if decision == "review" else
               f"Pay normally: {pct} chance of fraud; a review would cost about Rs {-ev:,.0f} "
               f"more than it saves.")
    return dict(claim_id=c.claim_id, fraud_probability=round(p, 4), expected_saving_rs=round(ev),
                decision=decision, summary=summary, reasons=reasons, notes=notes,
                partner_data_as_of=SNAP["as_of"])


@app.get("/health")
def health():
    return dict(status="ok", partner_data_as_of=SNAP["as_of"], partners=len(PARTNERS))


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
