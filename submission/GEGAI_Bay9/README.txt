GEGAI_Bay9 — DockSense
======================

Team              Bay 9
SPOC              Anirudh Badampudi — anirudhbadampudi@gmail.com
Second member     Akshat Agrawal — akshatagrawal.work@gmail.com
Submitted         11 September 2026


WHAT IS IN THIS FOLDER
----------------------

DockSense_Concept_Deck.pdf    Six-slide concept deck. Open this first.
DockSense_Concept_Deck.pptx   Same deck, editable in PowerPoint.
DockSense_Demo.mp4            Demo video, under 3 minutes, no narration —
                              captions are burned in.
figures/                      Real session-1 frames with every detection drawn
                              and labelled, plus a screenshot of the running
                              review console showing real incidents.

Live interactive deck   https://claude.ai/code/artifact/3003a68e-4119-44de-9bda-7364e4211acc
Source code             https://github.com/bnssaanirudh/DockSense


THE ONE-LINE VERSION
--------------------

Warehouses already record everything and watch none of it. A single frame cannot
tell you whether a box was placed or dropped — that distinction lives in a
sequence. DockSense tracks entities through loading-bay video and reasons over
their short histories, producing reviewable incidents with an evidence clip, an
explanation, a risk score, a separate confidence, and the SOP that applies.


HOW TO READ OUR NUMBERS
-----------------------

Two are worth your attention, and they point in opposite directions.

  Removing tracking drops reasoning F1 to 0.000 on both evaluation splits.
  Nothing fires at all without persistent identity. That is the evidence that
  this is a temporal system rather than a frame classifier.

  On seven clips of real loading-bay footage, we reported 1 of 9 expected
  behaviours. That is a poor result and we lead with it, because everything
  measured on synthetic clips was measured on clips that could not fail in the
  ways real footage does.

Precision on real footage is currently UNMEASURABLE, not unmeasured: every clip
we have shows a violation, so there is no footage of ordinary handling that the
system must stay silent on. A detector that fired constantly would score
perfectly. We say so rather than quoting a number that would look better.

Slide 5 of the deck lists the four defects that one afternoon of real footage
exposed — three fixed, one that tuning cannot fix.

Every figure is traceable: STATE.md in the repository carries a claims ledger
naming each number, where it was measured, and which are still marked NOT
MEASURED.


WHAT WE ARE NOT CLAIMING
------------------------

Real-time inference. Multi-camera tracking. Worker identification. Confirmed
product damage. Any trained model. Absolute distances or speeds — there is no
camera calibration, so every quantity is object-relative by design.


RESPONSIBLE AI
--------------

Behaviour, not identity: no face recognition, no re-identification, no worker
names, no ranking. Stored evidence clips blur the upper quarter of every detected
person box — deliberately not face detection, because running a face detector to
decide what to blur would build the exact capability we say we do not have. It
reduces identifiability; it is not anonymisation.

Observed, inferred and confirmed are kept distinct. The system reports what it
observed and what risk it inferred. It never claims damage occurred. Alerts are
decision support; the system produces no disciplinary output.

Datasets used for detector validation are CC BY 4.0 and attributed in the
repository README (Unsafe-Net, Önal & Dandıl 2024; LOCO, Mayershofer et al., TUM).
