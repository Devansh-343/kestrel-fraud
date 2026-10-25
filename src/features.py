"""Feature engineering for Kestrel warranty-claim fraud scoring.

Every feature here is something Kestrel knows at the moment a claim is
submitted. Partner "history" features only look at claims submitted
*before* the claim being scored (point-in-time), so the backtest is honest.
"""
import re
import numpy as np
import pandas as pd

AUTO_APPROVE_LIMIT = 2000          # ops-policy §5: auto-approve below this from 1 May 2026
POLICY_CHANGE = pd.Timestamp("2026-05-01")
GOODWILL_COST = 380                # §4: genuine claim held for review
CONTACT_COST = 260                 # §4: blended cost of one review contact
REVIEWS_PER_MONTH = 40             # §5: investigation desk capacity

INSPECTOR_NOTES = [
    "Minor fault, part swapped", "PCB replaced under warranty",
    "Customer has bill, serial verified", "Unit inspected, fault confirmed",
    "Motor winding failure confirmed", "Photos match fault, approved",
    "Heating element open circuit",
]
PARTNER_TYPES = ["authorised_service_centre", "franchise", "freelance_technician"]

FEATURES = [
    "log_amount", "amount_to_price", "near_threshold", "small_uninspected",
    "post_policy", "photo", "inspected", "note_blank",
    "warranty_used", "days_since_purchase", "prior_claims",
    "pt_franchise", "pt_freelance",
    "p_claims_60d", "p_small_uninsp_60d", "p_near_thr_60d", "p_growth",
    "p_hist_frauds", "p_hist_rate",
]


def normalise_serial(s):
    """Partners type serials by hand: 'kh-123', ' KH123' -> 'KH123'."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def clean_claims(df):
    """Dedupe re-submitted claims (keep the latest submission) and parse time."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["submitted_at"])
    df = df.sort_values("ts").drop_duplicates("claim_id", keep="last")
    df["serial_norm"] = df["product_serial"].map(normalise_serial)
    return df.reset_index(drop=True)


def claim_level(df, products, partners):
    d = df.merge(products, on="sku", how="left").merge(partners, on="partner_id", how="left")
    amt = d["claim_amount_inr"].astype(float)
    insp = d["partner_inspected"].eq("Y")
    d["log_amount"] = np.log1p(amt)
    d["amount_to_price"] = amt / d["list_price_inr"]
    d["near_threshold"] = amt.between(1900, AUTO_APPROVE_LIMIT - 0.01).astype(int)
    d["small_uninspected"] = ((amt < AUTO_APPROVE_LIMIT) & ~insp).astype(int)
    d["post_policy"] = (d["ts"] >= POLICY_CHANGE).astype(int)
    d["photo"] = d["photo_attached"].eq("Y").astype(int)
    d["inspected"] = insp.astype(int)
    d["note_blank"] = d["inspector_note"].isna().astype(int)
    d["warranty_used"] = d["days_since_purchase"] / (d["warranty_months"] * 30.4)
    d["prior_claims"] = d["customer_prior_claims"]
    d["pt_franchise"] = d["partner_type"].eq("franchise").astype(int)
    d["pt_freelance"] = d["partner_type"].eq("freelance_technician").astype(int)
    d["tenure_days"] = (d["ts"] - pd.to_datetime(d["onboarded_date"])).dt.days
    return d


def partner_volume(all_claims):
    """Rolling 60-day partner activity, strictly before each claim (no labels used)."""
    a = all_claims.sort_values("ts").copy()
    small_u = (a["claim_amount_inr"] < AUTO_APPROVE_LIMIT) & a["partner_inspected"].eq("N")
    near = a["claim_amount_inr"].between(1900, AUTO_APPROVE_LIMIT - 0.01)
    a["_one"], a["_su"], a["_nt"] = 1.0, small_u.astype(float), near.astype(float)
    out = []
    for _, g in a.groupby("partner_id", sort=False):
        g = g.set_index("ts")
        r60 = g[["_one", "_su", "_nt"]].rolling("60D", closed="left").sum().fillna(0)
        r365 = g["_one"].rolling("365D", closed="left").sum().fillna(0)
        res = pd.DataFrame({
            "claim_id": g["claim_id"].values,
            "p_claims_60d": r60["_one"].values,
            "p_small_uninsp_60d": r60["_su"].values,
            "p_near_thr_60d": r60["_nt"].values,
            # how much busier the last 60 days are than this partner's yearly pace
            "p_growth": r60["_one"].values / (r365.values / 6.0 + 1.0),
        })
        out.append(res)
    return pd.concat(out)


def partner_label_history(labelled, cutoff):
    """Fraud outcomes per partner, using only claims decided before `cutoff`."""
    h = labelled[(labelled["ts"] < cutoff) & labelled["is_fraud"].notna()]
    g = h.groupby("partner_id")["is_fraud"].agg(["sum", "count"])
    prior = 0.012   # base fraud rate; smooths partners with few claims
    g["p_hist_rate"] = (g["sum"] + prior * 20) / (g["count"] + 20)
    return g.rename(columns={"sum": "p_hist_frauds"})[["p_hist_frauds", "p_hist_rate"]]


def attach_partner_history(d, hist):
    d = d.merge(hist, left_on="partner_id", right_index=True, how="left")
    d["p_hist_frauds"] = d["p_hist_frauds"].fillna(0)
    d["p_hist_rate"] = d["p_hist_rate"].fillna(0.012)
    return d


def expected_saving(p, amount):
    """Rupees saved by reviewing this claim, in expectation (ops-policy §4)."""
    return p * amount - (1 - p) * GOODWILL_COST - CONTACT_COST
