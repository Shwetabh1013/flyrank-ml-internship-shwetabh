# Adding the Next Case Study

## How to add a new case study (repeat this every time)

1. Finish the work and get a real number out of it — a metric, a before/after comparison, not a vibe.
2. Draft in the same three-beat shape as the FlyRank case: **problem → what I did → what came of it**. If the result isn't clean, keep the "not a win — evidence" framing rather than tuning it away.
3. Paste the raw findings into the Claude Project — it already has the voice card, repo structure, and W01–W04 context loaded, so this is a short conversation, not a rebuild.
4. Drop the new case into `docs/index.html`, replacing the relevant TODO placeholder, and commit.
5. Update `case_study_flyrank.md` (or add a new file, `case_study_<name>.md`) with the same structure so the raw text exists outside the HTML too.

## Next case study, named

**W05: trained ranking model vs. baseline.**

Replace the hand-weighted scoring rule from W02/W04 with a trained supervised ranking model. Validate with precision@20 against:
- the staleness-only baseline (0.31)
- the hand-weighted heuristic (0.29)

The case study is whether learning from real outcomes beats both.

## Reminder set

Google Calendar event created:

- **Title:** Add W05 case study to portfolio (trained model vs. baseline)
- **When:** Monday, Sept 14, 2026, 9:00–9:30 AM IST
- **Alert:** 30-minute popup reminder
- **Link:** https://www.google.com/calendar/event?eid=dm81ODU4ZHVrODdpMnFwNWZvc3M0dTBmcG8gc2h3ZXRhYmguc2tzcy4xMmFAbQ
