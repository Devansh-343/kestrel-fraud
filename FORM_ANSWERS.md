# Submission form: draft answers

Items in [brackets] are yours to fill in or confirm. Edit anything that doesn't match what you actually did.

---

**What did you build, and what business outcome does it move? State the number and the money.**

A claim-review flag that scores every warranty claim before payout and sends it to the investigation desk only
when the expected fraud stopped outweighs the cost of reviewing it (Rs 380 goodwill if genuine + Rs 260 per
contact, ops-policy §4). Walk-forward backtest, Jan–Jun 2026: ~9 reviews a month, 25 frauds caught, **net Rs 1.27
lakh over six months (~Rs 21,000/month, Rs 2,307 saved per claim checked)**. Filling all 40 desk slots by
probability would have netted Rs 25,600; checking 40 random new-partner claims would have lost Rs 1.31 lakh.
Second outcome: seven specific outlets produced 33 of the 35 small-claim frauds after the May auto-approve change
(33 of their 51 small uninspected claims were fraud). Putting them back on inspection targets roughly
Rs 20,000–25,000 a month. Plus a FastAPI service and screen that return the decision with plain-English reasons.

**What does one run cost, and what would a month cost at Vireo's volume (roughly 650 tickets a week)?**

This question names Vireo and tickets, which look carried over from a different brief; I answered for Kestrel
below. For completeness: no paid calls, so 650 × 4.33 ≈ 2,815 records/month × Rs 0 = Rs 0 in API cost. Only
hosting applies (see the Kestrel answer).

**What does one prediction cost, and what would a month cost at Kestrel's volume (about 750 claims a month)?**

No paid API calls anywhere. One prediction is a local scikit-learn model call: about 10–20 ms of CPU (11 re-scores
for the reasons included). 750 claims × Rs 0 = **Rs 0/month in model cost**. Hosting: the service fits on the
smallest cloud VM (1 vCPU, 1 GB), roughly Rs 500–900/month, or on an existing Kestrel server for nothing extra.
Retraining takes ~20 seconds on a laptop.

**How do you know it works?**

Walk-forward backtest: for each month Jan–Jun 2026, trained only on earlier claims and earlier outcomes, then
picked that month's reviews. 4,295 decided claims, 67 frauds (Rs 2.58 lakh). Compared against four baselines on
the same months. Results: 45% of reviewed claims were fraud (base rate 1.2%); caught 25 of 67 frauds (37%) but
59% of fraud rupees. Errors: 63% of frauds missed, mostly (a) small claims from new bad outlets before they had any
confirmed history and (b) large claims from partners with a clean record. 55% of reviews were genuine customers,
typically mid-size claims from partners with past fraud. Monthly AUC 0.64–1.00, median 0.84; with only 5–22 frauds
a month, single months are noisy. Details in EVIDENCE.md. Service has 6 automated tests.

**Did you change, narrow, or push back on the client's ask?**

Yes, three times.
1. *Accuracy KPI.* At the start, after seeing that a model that flags nothing scores 96.9–99.3% every month. I
   still report accuracy because the board asked, but I optimised and report rupees saved per claim checked,
   which is what Farhan asked for and what the 40-claim capacity makes the real question.
2. *"New partners are the problem."* Tested it rather than encoding it. New partners do have ~7× the fraud rate,
   but it comes from 7 of 60 outlets. I left partner age out of the model (it added one caught claim over six
   months) so the flag targets behaviour, not onboarding date, as Meenal asked.
3. *Using all 40 slots.* The desk shouldn't fill all 40 slots: reviews beyond ~10–15 a month cost more in
   goodwill than they recover. The memo recommends using spare capacity to audit the seven outlets instead.
I also kept the client data out of the public repo (ops-policy §10). Only code, aggregate reports and
`predictions.csv` (claim ID + score, which the brief requires) are committed.

**What is wrong with what you are handing us?**

- The service uses a **static partner snapshot** (as of 30 Sep 2026). In production, partner activity counts and
  fraud history should be computed live from the CRM; right now they go stale.
- **Zoho labels**: ~3% of pre-Oct-2025 "not fraud" labels are probably undecided cases. I kept them (expected 1–2
  hidden frauds) rather than throw away six months of data.
- **Probabilities are not well calibrated.** ~85% of claims sit at the same floor value, and the expected-saving
  rule depends on the probability being right. I didn't calibrate (too few frauds to do it reliably).
- **Reasons are approximate:** each feature is reset to a typical value and the claim re-scored. Correlated
  features (partner fraud count and rate) can both claim credit.
- The Rs 1,900–2,000 "near threshold" band is my choice, not tuned.
- The out-of-warranty note ignores Kestrel Shield, which isn't in the data.
- `predictions.csv` in a public repo is still claim-level data; strictly, §10 would keep it private too.
- No authentication on the endpoint; the screen loads a Google font (falls back to system fonts offline).
- No retraining schedule or drift monitoring.

**What did you deliberately leave out, and why that rather than something else?**

- A language model in the product. Reasons come straight from the model, cost nothing and can't make things up.
  An LLM would add cost, a key dependency and a new failure mode for no better decision.
- Text features from fault descriptions and inspector notes: tested, near-zero signal.
- Hyperparameter search and fancier models: with 145 frauds, gains would be within noise. The time went into an
  honest backtest and the rupee rule instead.
- Partner age as a feature, for the reasons above.

**Anything you built or found that nobody asked for?**

- The seven-outlet finding, including claims priced at exactly Rs 1,995 just under the auto-approve limit, and the
  recommendation to fix it with inspection rather than investigation.
- The "review only when it pays" rule, which shows the desk should do ~10–15 checks a month, not 40.
- A Jul–Sep review queue (`reports/review_queue_jul_sep_2026.csv`, git-ignored, regenerated by train.py).
- The Vireo/tickets question in the form looks like it came from another brief.

**What did you use AI for?** [Edit to match what you actually did.]

Claude (Anthropic, Claude Opus 5.5 in claude.ai) wrote most of the analysis and code from my brief: data audit,
features, backtest, FastAPI service, screen, and drafts of the memo and this form. I [reviewed the backtest
numbers / reran everything locally / changed X / rewrote Y]. Discarded along the way: logistic and blended models,
a class-weighted model that broke the rupee rule, partner-age and serial-reuse features. Cost: [your plan's
subscription; no API credits used]. Where it wasted time: [e.g. tie-breaking scores took three attempts].
Screen recording: [link]

**Someone picks this up on Monday and you are unreachable. The three things they need to know.**

1. Run `python src/train.py` with the five CSVs in `data/` first; it builds the model, `predictions.csv` and the
   partner snapshot the service needs. Then `uvicorn app.main:app`.
2. The decision rule is expected rupees, not probability. Costs live at the top of `src/features.py`; if Kestrel
   changes the goodwill amount, the Rs 2,000 limit or desk capacity, change them there and retrain.
3. Partner features are the backbone and they go stale. Retrain monthly as investigations close, and replace the
   snapshot with a live CRM query before anyone relies on it.

**Honest hours spent.** [your number]

**GitHub repo link.** [link]

**Public Google Drive link.** [link]
