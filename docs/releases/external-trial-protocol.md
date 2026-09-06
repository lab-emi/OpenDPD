# External trial and onboarding protocol (S14, human-run)

The plan's S14 gate needs evidence from people who did not build the Studio:
three independent external groups completing real tasks, and an onboarding
session with at least five target users of whom at least four complete the
example flow without a maintainer changing code. This protocol fixes *how*
that is measured so the report is comparable and honest. Status: **pending**
until the sessions have happened; agents do not run or simulate them.

## Candidate to test

One release candidate (wheel + checksum from `docs/releases/studio-progress.md`),
installed by the participant from the written instructions only. Record the
exact commit, OS, Python and browser.

## Group trials (≥ 3 groups)

| Group task | What "complete" means | Recorded |
|---|---|---|
| Own data | import a capture they own, run the doctor, create a preprocessing version, train the smoke PA recipe, read the result page | data size and format, every blocking problem, time to first result |
| New method | add a model through `docs/tutorials/adding-a-model.md` and run it through GUI and CLI | what had to be asked, what the docs missed |
| Independent reproduction | import a share package produced elsewhere and re-evaluate it; compare with the stored result | agreement within the frozen tolerance or the reason it differs |

Attempts that were abandoned count as results; the report lists them with
the reason. Fixes made during a session are noted and re-tested in a later
candidate, never applied live to make the session succeed.

## Onboarding session (≥ 5 users)

Task: install, register the built-in example, train the smoke recipe, open the
result, export a share package. Two clocks per participant:

- **operation time**: while the participant is reading, clicking or typing;
- **waiting time**: install downloads and training, not attributable to the
  interface.

Pass = the task completed without a maintainer touching code or
configuration. Hints given by the observer are recorded verbatim; a hint that
changes a setting counts as a failure of the documentation, not a pass.

## Report

`docs/releases/external-trial-report.md` with: participants (role, prior DPD
experience, machine), per-task outcome, both clocks, every blocking problem
with its fix or documented degradation path, and the prioritised backlog that
came out of it. The S14 acceptance items in `docs/releases/studio-progress.md`
are ticked only from that report.
