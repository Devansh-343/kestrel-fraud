"""Train the claim-fraud model, backtest it month by month, write predictions.

Usage:  python src/train.py            (expects the CSVs in ./data)

Outputs
  predictions.csv              one score per claim_id in test_unlabelled.csv
  artifacts/model.joblib       fitted model + feature list
  artifacts/partner_snapshot.json  per-partner stats the service needs
  reports/backtest.csv, reports/backtest.md, reports/errors.csv
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F

ROOT = Path(__file__).resolve().parents[1]
DATA, ART, REP = ROOT / "data", ROOT / "artifacts", ROOT / "reports"
ART.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)
RNG = np.random.default_rng(7)


def load():
    tr = F.clean_claims(pd.read_csv(DATA / "train.csv"))
    te = F.clean_claims(pd.read_csv(DATA / "test_unlabelled.csv"))
    products = pd.read_csv(DATA / "products.csv")
    partners = pd.read_csv(DATA / "partners.csv")
    return tr, te, products, partners


def build(tr, te, products, partners):
    allc = pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True)
    d = F.claim_level(allc, products, partners)
    d = d.merge(F.partner_volume(allc), on="claim_id", how="left")
    return d


class Blend:
    """Tried and discarded: average of GBM and logistic. Logistic learned "small claims are safe"
    before May and collapses after the auto-approve change (AUC 0.22 in May 2026).
    GBM alone gives ~75% of claims the exact same score; the logistic part breaks those ties."""
    def __init__(self):
        m = _base_models(); self.gbm, self.lr = m["gbm"], m["logistic"]
    def fit(self, X, y):
        self.gbm.fit(X, y); self.lr.fit(X, y); return self
    def predict_proba(self, X):
        return 0.5 * self.gbm.predict_proba(X) + 0.5 * self.lr.predict_proba(X)


def make_models():
    m = _base_models(); m["blend"] = Blend(); return m


def _base_models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
        "gbm": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              random_state=0),
    }


def pick(month_df, score, k=F.REVIEWS_PER_MONTH, require_positive=False):
    """Choose up to k claims to review this month, highest score first."""
    s = month_df.assign(_s=score).sort_values("_s", ascending=False)
    if require_positive:
        s = s[s["_s"] > 0]
    return s.head(k)


def outcome(chosen):
    fr = chosen["is_fraud"] == 1
    caught = chosen.loc[fr, "claim_amount_inr"].sum()
    genuine = int((~fr).sum())
    net = caught - genuine * F.GOODWILL_COST - len(chosen) * F.CONTACT_COST
    return dict(reviewed=len(chosen), frauds_caught=int(fr.sum()), genuine_held=genuine,
                fraud_rs_stopped=round(caught), net_rs=round(net),
                net_rs_per_review=round(net / max(len(chosen), 1)))


def backtest(d, feats, months):
    rows, err_rows = [], []
    lab = d[(d.split == "train")]
    for m in months:
        start = pd.Timestamp(m); end = start + pd.offsets.MonthBegin(1)
        hist = F.partner_label_history(lab, start)
        dd = F.attach_partner_history(d, hist)
        trn = dd[(dd.split == "train") & (dd.ts < start) & dd.is_fraud.notna()]
        val = dd[(dd.split == "train") & (dd.ts >= start) & (dd.ts < end) & dd.is_fraud.notna()]
        y = val.is_fraud.values
        all_fraud_rs = val.loc[val.is_fraud == 1, "claim_amount_inr"].sum()
        base = dict(month=m, claims=len(val), frauds=int(y.sum()), fraud_rs=round(all_fraud_rs),
                    accuracy_if_all_genuine=round(1 - y.mean(), 4))
        for name, mdl in make_models().items():
            mdl.fit(trn[feats], trn.is_fraud)
            p = mdl.predict_proba(val[feats])[:, 1]
            r = dict(base, strategy=f"{name}: rank by expected rupees",
                     model_accuracy_at_0_5=round(((p >= 0.5) == y).mean(), 4),
                     auc=round(roc_auc_score(y, p), 3), pr_auc=round(average_precision_score(y, p), 3),
                     **outcome(pick(val, F.expected_saving(p, val.claim_amount_inr), require_positive=True)))
            rows.append(r)
            rows.append(dict(base, strategy=f"{name}: rank by probability",
                             **outcome(pick(val, p))))
            if name == "gbm":
                v = val.assign(p=p, ev=F.expected_saving(p, val.claim_amount_inr))
                v["flagged"] = v.index.isin(pick(val, v.ev.values, require_positive=True).index)
                err_rows.append(v)
        rows.append(dict(base, strategy="baseline: 40 largest claims",
                         **outcome(pick(val, val.claim_amount_inr.values))))
        new = val[val.tenure_days < 365]
        sims = [outcome(new.sample(min(40, len(new)), random_state=int(s))) for s in RNG.integers(0, 1e6, 50)]
        rows.append(dict(base, strategy="baseline: 40 random claims from new partners",
                         **{k: round(np.mean([s[k] for s in sims])) for k in sims[0]}))
        sims = [outcome(val.sample(40, random_state=int(s))) for s in RNG.integers(0, 1e6, 50)]
        rows.append(dict(base, strategy="baseline: 40 random claims",
                         **{k: round(np.mean([s[k] for s in sims])) for k in sims[0]}))
    return pd.DataFrame(rows), pd.concat(err_rows)


def main():
    tr, te, products, partners = load()
    d = build(tr, te, products, partners)
    feats = F.FEATURES
    months = [f"2026-{m:02d}-01" for m in range(1, 7)]

    bt, errs = backtest(d, feats, months)
    bt.to_csv(REP / "backtest.csv", index=False)

    # Ablation: does adding partner tenure ("new partner") help?
    bt_t, _ = backtest(d, feats + ["tenure_days"], months)
    bt_t.to_csv(REP / "backtest_with_tenure.csv", index=False)

    keep = ["claim_id", "ts", "partner_id", "sku", "claim_amount_inr", "partner_inspected",
            "customer_prior_claims", "tenure_days", "is_fraud", "p", "ev", "flagged"]
    errs[keep].to_csv(REP / "errors.csv", index=False)

    # ---- final model: all decided training claims, partner history to 1 Jul 2026
    cutoff = pd.Timestamp("2026-07-01")
    hist = F.partner_label_history(d[d.split == "train"], cutoff)
    dd = F.attach_partner_history(d, hist)
    trn = dd[(dd.split == "train") & dd.is_fraud.notna()]
    model = make_models()["gbm"].fit(trn[feats], trn.is_fraud)
    test = dd[dd.split == "test"].copy()
    test["score"] = model.predict_proba(test[feats])[:, 1]
    # ~85% of claims land on the GBM's floor probability. Break those ties with the logistic
    # model's ranking. The weight is tiny, so the order of above-floor claims is unchanged.
    fine = make_models()["logistic"].fit(trn[feats], trn.is_fraud)
    test["submit_score"] = test["score"] + 1e-5 * pd.Series(
        fine.predict_proba(test[feats])[:, 1], index=test.index).rank(pct=True)
    test["expected_saving_rs"] = F.expected_saving(test.score, test.claim_amount_inr)

    sub = pd.read_csv(DATA / "sample_submission.csv")[["claim_id"]]
    sub = sub.merge(test[["claim_id", "submit_score"]].rename(columns={"submit_score": "score"}),
                    on="claim_id", how="left")
    assert sub.score.notna().all() and len(sub) == sub.claim_id.nunique()
    sub["score"] = sub.score.round(9)
    sub.to_csv(ROOT / "predictions.csv", index=False)

    # Review queue for Jul-Sep: 40 per month by expected rupees
    test["month"] = test.ts.dt.to_period("M").astype(str)
    q = (test[test.expected_saving_rs > 0].sort_values("expected_saving_rs", ascending=False)
         .groupby("month").head(F.REVIEWS_PER_MONTH))
    q[["month", "claim_id", "partner_id", "claim_amount_inr", "score", "expected_saving_rs"]] \
        .to_csv(REP / "review_queue_jul_sep_2026.csv", index=False)

    # ---- snapshot for the service: per-partner state as of the end of the data
    end = d.ts.max()
    recent = d[d.ts > end - pd.Timedelta(days=60)]
    yr = d[d.ts > end - pd.Timedelta(days=365)].groupby("partner_id").size()
    small_u = (recent.claim_amount_inr < F.AUTO_APPROVE_LIMIT) & recent.partner_inspected.eq("N")
    snap = {}
    for pid, row in partners.set_index("partner_id").iterrows():
        r = recent[recent.partner_id == pid]
        n60 = len(r)
        snap[pid] = dict(
            city=row.city, partner_type=row.partner_type, onboarded_date=row.onboarded_date,
            p_claims_60d=n60,
            p_small_uninsp_60d=int(small_u[recent.partner_id == pid].sum()),
            p_near_thr_60d=int(r.claim_amount_inr.between(1900, 1999.99).sum()),
            p_growth=n60 / (yr.get(pid, 0) / 6 + 1),
            p_hist_frauds=float(hist.p_hist_frauds.get(pid, 0)),
            p_hist_rate=float(hist.p_hist_rate.get(pid, 0.012)),
        )
    json.dump(dict(as_of=str(end), partners=snap), open(ART / "partner_snapshot.json", "w"), indent=1)
    imp_base = trn[feats].median().to_dict()
    joblib.dump(dict(model=model, features=feats, medians=imp_base,
                     train_fraud_rate=float(trn.is_fraud.mean())), ART / "model.joblib")

    write_report(bt, bt_t, test, q)
    print(open(REP / "backtest.md").read())


def summarise(bt, months_label, mask):
    g = bt[mask].groupby("strategy")[["reviewed", "frauds_caught", "genuine_held",
                                        "fraud_rs_stopped", "net_rs"]].sum()
    g["net_rs_per_review"] = (g.net_rs / g.reviewed.clip(lower=1)).round()
    g["frauds_per_40"] = (g.frauds_caught / g.reviewed.clip(lower=1) * 40).round(1)
    return f"### {months_label}\n\n" + g.sort_values("net_rs", ascending=False).to_markdown() + "\n"


def write_report(bt, bt_t, test, q):
    one = bt.drop_duplicates("month")
    lines = ["# Backtest: train on everything before a month, pick 40 claims in that month\n",
             "Validation months Jan-Jun 2026, expanding window. Only decided claims are scored "
             "(undecided cases have no outcome). Net rupees = fraud stopped - Rs 380 x genuine "
             "customers held - Rs 260 x reviews (ops-policy section 4).\n",
             "## Why accuracy is the wrong KPI\n",
             one[["month", "claims", "frauds", "fraud_rs", "accuracy_if_all_genuine"]].to_markdown(index=False),
             "\nA model that never flags anything already scores ~98-99% accuracy.\n",
             "## Ranking quality per month\n",
             bt[bt.strategy.str.endswith("rank by expected rupees")][["month", "strategy", "auc", "pr_auc", "model_accuracy_at_0_5"]].to_markdown(index=False),
             "\n## Rupees, all six months\n",
             summarise(bt, "Jan-Jun 2026", bt.month.notna()),
             summarise(bt, "After the auto-approve change (May-Jun 2026)", bt.month >= "2026-05-01"),
             "## Ablation: adding partner tenure ('new partner')\n",
             summarise(bt_t, "Jan-Jun 2026, with tenure feature", bt_t.month.notna()),
             "## Test period (Jul-Sep 2026)\n",
             f"Claims scored: {len(test)}. Mean predicted fraud probability: {test.score.mean():.3f} "
             f"(expected frauds ~{test.score.sum():.0f}). Claims with positive expected saving: "
             f"{(test.expected_saving_rs > 0).sum()}. Review queue (<=40/month): {len(q)} claims, "
             f"expected net saving Rs {q.expected_saving_rs.sum():,.0f}.\n"]
    (REP / "backtest.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
