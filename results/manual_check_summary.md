# Manual check of the fixed rule (23 Nov 2026)

**Labels put by Claude (AI), not yet reviewed by a person.** Before using these numbers in the pitch or on
the website, a team member must review the labels (column `REVIEWED BY (name) / CHANGED?` in the Excel file)
and the text below must say who reviewed them.

File with all the labels: `data/manual_check_CLAUDE_labels.xlsx` (not on GitHub: it contains copied reviews).
Created by `src/check_pieces.py`; settings of the rule: thresholds 0.50 / 0.50, contrast word removed.

## 1. citizenM Zürich: 30 real Booking reviews, each as one text (sheet `google_pieces`)

| What | Result |
|---|---|
| Pieces | 118 |
| Cut in the right place (one judgement per piece) | 110 / 118 (93.2%) |
| Pieces without a judgement (neutral) | 7 (5.9%): 6 sent to negative, 1 to positive |
| Right side, pieces with a judgement | 103 / 111 (92.8%) |
| Criticisms on the negative side | 46 / 46 (100%) |
| Praise on the positive side | 57 / 65 (87.7%) |

- **All the errors go in one direction: praise or neutral facts sent to the negative side** (14 of the 66
  pieces sent to negative, about 21%). No criticism was lost. So the report does not miss problems, but it
  can show problems that are not there.
- Example that reaches the report: "didn't hear a thing from inside our room" (praise of the soundproofing)
  is counted as a Noise criticism and shown as an example under Noise in Fix first. Model 2 reads "didn't" as
  negative.
- Other praise sent to negative: short or unusual words ("Technology", "Funky lights"), "not typical of where
  we usually stay", "the issue was resolved".
- Bad cuts (8): commas that cut one judgement in two ("Staff, location, bed were great"), and a list number
  ("3") taken as a piece.

**Fix first of the hotel (Room, Noise, Bed, Comfort, Facilities) against the reviews.** The main problems
that a reader sees are there: small rooms (many reviews) and the bed pushed against the wall/window
(4 reviews, found as Bed). Noise is real (construction, night club, busy street) but one of its 6 reviews is
the soundproofing praise above. The heat / air conditioning complaints (4 reviews) have no topic of their own
among the 21 and are spread over Facilities and Comfort.

## 2. 515K, 30 pieces sent to the "wrong" side (train hotels, sheet `wrong_side`)

| What | Result |
|---|---|
| Real errors of the rule (the Booking field was right) | 14 (47%) |
| The guest wrote it in the wrong field (the rule was right) | 11 (37%) |
| Neutral, no judgement | 5 (17%) |

So about half of the "wrong side" counted by `verify_full_texts.py` are not errors of the rule: the automatic
numbers underestimate the rule. Several real errors are long pieces without punctuation (the 515K has none).

## 3. HRAST, 30 sentences with a contrast word (sheet `contrast_sentences`)

23 / 30 (76.7%) cut and sent to the right sides. Errors: mitigations sent to negative ("but it is always so in
old city centers", "but not sure that would put me off staying again"), a criticism sent to positive ("but a
bit far from the city center"), a praise left inside a negative piece ("the staff was helpful").

## Limits of this check

- One person labeled, and it was an AI: no agreement between two people can be measured yet.
- One Swiss hotel, 30 reviews, all in English, copied from Booking (not Google).
- 30 + 30 pieces in sections 2 and 3: the percentages have a wide margin of error.
