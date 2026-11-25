# Apple M6 paired timing records

`timing_pairs.csv`, `timing_summary.csv`, and `timing_audit.json` contain the saved five-round results and endpoint checks used in the presentation and report. Verify the arithmetic with `python3 verify_recorded.py`.

To run the benchmark protocol again, follow the full-data command in the repository root README. The script builds an instrumented author solver from a pinned source revision, obtains and verifies the official data, runs both modes in alternating order, and writes a timestamped set of logs and summaries. Elapsed times will vary with the machine and build environment; the archived Apple M6 table and any new run are separate sessions.
