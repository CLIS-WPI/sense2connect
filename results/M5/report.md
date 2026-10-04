# M5 — Experiments and figures (in progress)

M5 started after the ROADMAP update (commit 24197f2). Status: the figures and the table are built; `paper/numbers.tex` is waiting for `paper/main.tex`, which you will add. Nothing is committed.

## Outputs (one script each, run in the container)

| Output | Command | Source data |
|---|---|---|
| `paper/tables/tab_blockage.tex` (+ `results/M5/blockage_table.json`) | `python scripts/fig_blockage_table.py` | cached comm traces `results/cache/*/*/seed_*/comm_trace.json`, evaluation seeds (no re-trace) |
| `paper/figs/fig_leadtime.pdf` (+ `results/M5/leadtime.json`) | `python scripts/fig_leadtime.py` | `results/M2/followup.md`, lead-time table (evaluation seeds) |
| `paper/figs/fig_onset.pdf` | `python scripts/fig_onset.py` | `results/M3/onset/onset_events.json` + `m3_timeline` caches |
| `paper/figs/fig_outage_margin.pdf` | `python scripts/fig_outage_margin.py` | `results/M3/metrics.json`, `hybrid.json`, `genie.json` |
| `paper/figs/fig_headroom.pdf` | `python scripts/fig_headroom.py` | `results/M3/genie.json` (A3 decomposition, tau_HO 20 ms and 0), cross-checked against `genie2.json` |

Style (`scripts/figstyle.py`): vector PDF, IEEE single-column width 3.5 in, 8 pt serif (STIX), TrueType fonts embedded (fonttype 42), no titles, PDF creation metadata cleared.

## Notes per item

1. **Blockage table.** This re-runs M1.5 part 2 (two O-RUs: serving-link 10 dB LoS events, other-cell availability, other-cell power, time with both cells blocked, and the fixed vs. oracle event rate) on the evaluation seeds. It calls the unchanged `scripts/diversity_study.py` functions on the cached comm traces. Raw UE events are rebuilt from the cached per-link trace with the same `_ue_events` call the trace loop used.
   - Check: the same code on the tuning seeds reproduces `results/M1.5/diversity.json` exactly (other-cell counts per mount and class, both-blocked counts, fixed/oracle rates). The script stops if it does not.
   - Evaluation seeds:
     - bus/truck: other cell clear in 68 % (lamppost) / 71 % (facade) of events;
     - pedestrian: 27 % / 34 %;
     - both cells ≥ 10 dB: 4.0 % / 3.5 % of the time.
2. **Lead time.** Map-constrained vs. unconstrained tracker with the same detector (budget 4 FA/CPI, blind clutter, no ghost handling), pooled over mounts, with 95 % Wilson intervals.
   - The counts are parsed from the table that `scripts/run_m2_followup.py` wrote, not recomputed. Re-running the tracker replay would load the M2 detection caches, and their provenance (commit 216eb53) is refused by the current tree.
   - Events here are the M2 definition (10 dB LoS on oru-0, 0.1 s), not the M3 fixed-cell events.
3. **Onset figure.** A restyled version of the M3 figure. Same events (rule: duration closest to the class median), same handovers. Solid lines are the 3GPP reference margin, dashed lines 10 dB. Genie markers are drawn wide and light under the A3 markers, because some switch times coincide.
4. **Outage vs. margin.**
   - Schemes: A3 (main tuned), hybrid (jointly tuned), genie + A3 (first-round policy, **without** sensing overhead, i.e. the stronger bound), and the oracle.
   - Mean and 95 % CI over 40 evaluation jobs, log scale. A small horizontal offset keeps the CI bars apart.
   - The 3GPP reference (35.9 dB) is marked. The v1-radio point is not plotted.
5. **Headroom.** Stacked shares of A3 outage per margin: both cells unusable, handover interruption, foresight bound (wrong cell inside 10 dB events), and wrong cell outside events.
   - The fourth component is needed for the stack to sum to the A3 outage; the script checks the sum.
   - Line: A3 outage at tau_HO = 0 (retuned) relative to tau_HO = 20 ms.
   - Shares rather than absolute values, because "both unusable" spans 0.02–35 s/UE-min.

## Open items for you

- `paper/numbers.tex` will be generated once `paper/main.tex` (with the macro list in its header) exists.
- The M1 single-O-RU statistics (`blockage_stats.py`, M1.5 part 1) use `configs/m1_scenario.yaml`, which is not cached. Re-running them on evaluation seeds needs a short re-trace (comm only). Not done; your decision.
- Lead time from recomputation instead of parsing would need the M2 detection caches restamped (`scripts/restamp_cache.py`) or the provenance check relaxed for that replay. Not done; your decision.
- The ROADMAP acceptance asks for one command for everything. A wrapper (e.g. `scripts/make_paper.sh` running all `fig_*.py`) is not written yet.
