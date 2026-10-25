# Byte-level tokenization: Trie, greedy and minimum-token DP

A Python implementation of byte-level token matching and segmentation by **Fengrui Zhang**. It compares longest-match greedy encoding with dynamic programming that minimizes the number of tokens under a fixed vocabulary. Both policies share an immutable Trie and strict UTF-8 decoder.

This repository includes a runnable standalone encoder package and the fuller custom experiment, analysis and integration source from the original course project. It does **not** include the course BPE trainer, checker, benchmark/comparator, assignment, original course data, final trained model or raw experiment logs. The historical training-to-report pipeline therefore has unresolved external dependencies; it is not presented as independently reproducible from this repository.

## Quick start

Use **Python 3.11 or newer**. The standalone package uses only the standard library; no installation is required. From the repository root:

```sh
python3 demo.py
python3 demo.py --text '你好, tokenizer 😀' --strategy dp
python3 -B -m unittest discover -s tests -v
```

The default example loads `examples/toy-vocabulary.json`: all 256 individual bytes plus the manually specified tokens `ab` and `bcd`. No training or download occurs.

| Policy on `abcd` | Segmentation | Token IDs | Count |
|---|---|---|---:|
| Greedy | `ab` · `c` · `d` | `[256, 99, 100]` | 3 |
| Minimum-token DP | `a` · `bcd` | `[97, 257]` | 2 |

Supply an explicit compatible vocabulary with `--model PATH`, choose `--strategy greedy`, `dp` or `both`, and use `--text` for the input. The model stores token byte sequences as hex strings and keeps the historical format identifier `msdm5051-tokenizer-v1`. This identifier is retained for compatibility, not as a course endorsement. IDs 0–255 must match their byte values; other tokens must be nonempty and distinct.

## What the implementation does

- A byte Trie shares vocabulary prefixes and returns every complete matching token, including shorter prefix tokens.
- Greedy selects the longest complete token at the current byte position. This is a local choice, not an optimal token-count guarantee.
- DP computes the minimum count from every byte position and reconstructs the chosen segmentation iteratively. Ties favor a longer current token, then a smaller token ID.
- The decoder joins all token bytes before strict UTF-8 decoding. Individual tokens may contain only part of a character. No Unicode normalization or whitespace rewriting is performed.
- Model preparation creates immutable runtime data and retains the original 1 MiB canonical JSON limit.

For `n` input bytes and maximum token length `L`, DP matching takes `O(nL)` time with `O(n)` DP state, in addition to the shared Trie. Its guarantee is for a **fixed vocabulary**; it does not establish better vocabulary learning, language-model quality or speed.

## Code layout and completeness

```text
tokenizer_core.py           standalone Trie, encoders and decoder
demo.py                     explicit-vocabulary command-line demo
tests/                      26 independent encoder/model test methods
examples/                   manually specified teaching fixtures
historical_pipeline/src/    full custom experiment/analysis source,
                            integration implementations and original tests
```

The [historical pipeline guide](historical_pipeline/README.md) maps all four custom experiment/analysis modules, three integration implementations and six original test files to their dependencies. The original suite has 67 methods; the root suite contains 26 selected methods from it. These counts overlap and must not be added together.

Full custom source coverage does not mean the excluded course components have been replaced. The root demo needs no trainer. The historical `train()` interface delegates to an omitted course baseline, and some historical analyses import the omitted course harness/comparator. Obtain upstream components only from an authorized course distribution or the rights holder; no public distribution location is asserted here.

## Verification and historical results

The release validation runs the standalone import, default and Unicode demos, and all 26 root tests. The recorded environment and results are in [release-validation.json](release-validation.json). Historical training, benchmarks, expensive experiments and the full upstream-dependent suite are not run as part of this release preparation.

The supplied historical report records 507 greedy tokens versus 503 DP tokens on a small smoke evaluation: approximately 0.79% fewer tokens and approximately 5.76× encoding time for DP. Those figures are report evidence, **not measurements from this standalone package or production benchmarks**. The original trained model and raw experiment logs were not supplied. The toy examples in this repository cannot reproduce those historical figures.

## Attribution and license

The original vocabulary training reused a course-provided BPE baseline; that implementation is excluded. The project report disclosed OpenAI Codex and DeepSeek assistance with implementation, tests, experiment design, analysis and writing. Release preparation also used Codex to separate the standalone package, preserve the custom historical code and validate the lightweight entry points. See [ATTRIBUTION.md](ATTRIBUTION.md).

The included code is licensed under the [MIT License](LICENSE), copyright 2026 Fengrui Zhang. This license does not grant rights over the excluded course software, datasets or other third-party materials.
