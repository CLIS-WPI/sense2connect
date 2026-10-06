GOAL MODE. Save this message as GOAL_TVT.md on branch `tvt` (commit it
first), then work autonomously until the stop condition below.

Goal: complete milestones T0-T6 of ROADMAP_TVT.md on DEVELOPMENT seeds,
in order, and leave the branch ready for a freeze. Success means every
milestone is implemented, tested and honestly reported, NOT that the
proposed method wins. Null or negative results are valid outcomes.

Overrides of the STOP points in ROADMAP_TVT.md (decided by the humans):
- T0: use the paper-1 service model (best-beam SNR) for all journal
  results. Still implement the switch and report how both papers'
  headline numbers change under each model, then continue.
- T4 decision point: write the evidence for both narratives ("closes
  the gap" vs "requirements and achievable region") in the T4 report,
  do not choose, and continue with T5.
- T6 second deployment: write the geometry, O-RU placement, traffic and
  the reason in results/TVT/T6/scene_design.md and commit it BEFORE
  tracing; never change it after seeing results.
- Testbed work (OAI, FlexRIC, USRPs) stays out of scope.

Working rules:
- All rules of AGENTS.md and the research-integrity section of
  ROADMAP_TVT.md apply. Held-out seeds 4001-4010 must never be opened;
  never create or move the tag tvt-freeze (humans only).
- Never weaken an acceptance criterion, drop seeds or runs, or change
  the scenario, traffic or impairments to make a method look better.
  Baselines get the same tuning budget as the proposed method.
- Reuse existing caches in results/cache and the paper-2 caches
  READ-ONLY; check provenance; re-trace only when a milestone needs new
  geometry and say why. GPU 1 only, peak memory < 60 GB.
- Per milestone: unit tests pass, results/TVT/<T>/report.md written
  (what was built, how to run, tests, metrics with seed-level stats,
  deviations, open issues), then commit and push to `tvt` only (never
  results/, never main, never existing tags).
- Keep PROGRESS_TVT.md at the repo root updated after each milestone:
  status, GPU hours used, key numbers, open questions for the humans.
- If something is infeasible after three serious attempts, document
  it, mark the milestone BLOCKED with options, and continue with the
  next milestone that does not depend on it.
- If one experiment would need more than 24 GPU hours, write the
  estimate in PROGRESS_TVT.md and run a reduced version first.

Stop condition: T0-T6 are DONE or BLOCKED, the code is clean and
tested, and PROGRESS_TVT.md contains a freeze-readiness checklist and a
one-page summary of hypotheses J1-J5 on development seeds (supported /
not supported / mixed). Then STOP for human review. Do not start T7.
