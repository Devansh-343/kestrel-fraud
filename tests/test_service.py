"""Smoke tests for the scoring service. Run after `python src/train.py`:  python -m pytest -q"""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from main import app  # noqa: E402

client = TestClient(app)
BASE = dict(partner_id="SP3201", sku="KH-WP-01", claim_amount_inr=4225, days_since_purchase=253,
            photo_attached="Y", partner_inspected="Y", customer_prior_claims=1,
            submitted_at="2026-09-11 20:19")


def test_routine_claim_is_paid():
    r = client.post("/score", json=BASE).json()
    assert r["decision"] == "pay" and r["fraud_probability"] < 0.05


def test_flagged_outlet_small_claim_is_reviewed_with_reasons():
    claim = dict(BASE, partner_id="SP3129", sku="KH-CF-03", claim_amount_inr=1995,
                 days_since_purchase=12, partner_inspected="N", customer_prior_claims=0)
    r = client.post("/score", json=claim).json()
    assert r["decision"] == "review" and r["expected_saving_rs"] > 0
    assert len(r["reasons"]) >= 1 and all("reason" in x for x in r["reasons"])


def test_unknown_sku_is_a_readable_422():
    r = client.post("/score", json=dict(BASE, sku="NOPE"))
    assert r.status_code == 422 and "Unknown SKU" in r.json()["detail"]


def test_unknown_partner_still_scores_with_a_note():
    r = client.post("/score", json=dict(BASE, partner_id="SP9999")).json()
    assert r["notes"] and "not in the partner list" in r["notes"][0]


def test_missing_fields_rejected():
    assert client.post("/score", json={"partner_id": "SP3129"}).status_code == 422


def test_screen_and_health():
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
