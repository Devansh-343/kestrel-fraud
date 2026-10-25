# Screen recording outline (max 3 minutes, no slides)

**0:00–0:30 — The trap.** Open `reports/backtest.md`, "Why accuracy is the wrong KPI" table. "Flagging nothing
scores 97–99% every month, so I optimised rupees per claim checked under the 40-a-month limit."

**0:30–1:10 — What I tried and threw away.** Scroll the rupee tables. "Filling all 40 slots by probability: Rs 25k
over six months. Reviewing only when the expected saving is positive: Rs 1.27 lakh with ~9 reviews a month.
Logistic regression collapsed after the May rule change (AUC 0.22), so I dropped it. Partner age added one claim,
so I left it out. Serial reuse and text fields had no signal."

**1:10–1:50 — The finding.** Show the seven outlets in `memo.md`. "33 of their 51 small uninspected claims were
fraud, many priced at exactly Rs 1,995. Most new partners are fine. The fix is inspection for these seven, not the
desk."

**1:50–2:40 — The product.** Run `uvicorn app.main:app`, open localhost:8000. Click "Small claim, flagged outlet"
→ Review, 62%, reasons. Click "Routine inspected claim" → Pay. Type a bad SKU → readable error.

**2:40–3:00 — What's wrong.** "Partner stats are a static snapshot, probabilities aren't calibrated, and it misses
large frauds from clean partners. All listed in EVIDENCE.md and the form."
