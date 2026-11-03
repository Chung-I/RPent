# Help-ladder smoke test (PILOT ONLY, not in the paper): operational outputs

Gemma 4 31B FP8 and Qwen3.8-27B FP8, the 12 smoke cases x rungs A-D (48 rung runs per model), on ladder-pilot 9a31ab5.
`ladder ops` read only operational fields: replies, tool calls, tokens, seconds. The pilot's rows (`rows.jsonl`, with
the choices and costs) stay local and unread in the worktree's `logs/ladder/pilot_<model>/`. They are not committed.

Server start (the sbatch script's START line to vLLM's "Available routes"): Gemma 28.2 min (job 693822), Qwen 3.4 min
(job 694690). wandb (rpent-pick-place, ladder_pilot_ops): Gemma guxk4ucg, Qwen m102sle8.
