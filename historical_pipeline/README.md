# Custom historical experiment and integration source

This directory preserves the broader self-written source from the course project under the repository's MIT License. It is optional relative to the standalone root package. Several paths require excluded upstream course software; no claim is made that the whole original training-to-report environment is runnable from this repository alone.

## Coverage and dependencies

| Functionality | Source under `src/` | Dependency/status |
|---|---|---|
| Trie, greedy, minimum-token DP, strict decoding | `tokenizer_core.py` | Full historical core; standard library only. |
| Comparison integration | `adapter_greedy.py`, `adapter_dp.py` | Custom wrappers import omitted `baseline_bpe.train`; direct script entry imports omitted `project1_check.main`. |
| Final submission integration | `student_adapter.py` | Implementation and signatures retained; exercise/setup/contract prose omitted. Training and checker behavior remain upstream dependencies. |
| Controlled same-vocabulary comparison and synthetic length sweeps | `experiments/compare_encoders.py` | Full custom utility; does not train or use the course harness. Needs vocabulary and JSONL input, and uses the Unix `resource` module. |
| Token/embedding trace tables and UTF-8 byte spans | `experiments/inspect_tokens.py` | Full custom utility; needs vocabulary and JSONL input, with no baseline dependency. |
| Selected-record BPE/greedy/DP traces | `experiments/inspect_cases.py` | Requires omitted `baseline_bpe.prepare` and `baseline_bpe.encode`, plus a compatible saved BPE model. |
| Historical result/context audit | `experiments/summarize_results.py` | Requires omitted `benchmark.read_records`, `compare_results.compare` and compatible result files. Does not replace course scoring. |
| Full original tests | `tests/` | Six files and 67 test methods. The full suite needs omitted dependencies; only the independent root suite is executed in release validation. |

The historical `train()` function is an interface forwarding to the missing course trainer, not a substitute implementation. The separate encoder implementations and custom experiments are fully present. Missing course datasets, model and logs prevent a fresh reproduction of the historical report.

## Optional utilities after lightweight validation

Python 3.11+ is recommended. The comparison runner assumes Linux/macOS or Windows through WSL because it uses `resource`. From the repository root, the custom trace utility can consume the manual example files:

```sh
PYTHONPATH=historical_pipeline/src python3 -m experiments.inspect_tokens \
  --model examples/toy-vocabulary.json \
  --evaluation examples/review-records.jsonl \
  --strategy both --output toy-traces.md
```

This historical utility is preserved but was not executed during release preparation. A controlled timing run is also available as source, but is intentionally not part of the lightweight verification:

```sh
PYTHONPATH=historical_pipeline/src python3 -m experiments.compare_encoders \
  --model examples/toy-vocabulary.json \
  --evaluation examples/review-records.jsonl \
  --repeats 1 --output toy-comparison.json
```

The timing command includes the script's synthetic length sweep. Its output would describe the toy inputs, not reproduce the report. Utilities may save input text, machine metadata or filesystem paths in generated outputs; review those artifacts before sharing them.

For upstream-dependent paths, obtain the exact compatible course modules only from an authorized course distribution or rights holder. No public download location has been established. The repository does not copy those modules, and its MIT License does not cover them. Keep the existing course and AI-assistance attribution in [ATTRIBUTION.md](../ATTRIBUTION.md).
