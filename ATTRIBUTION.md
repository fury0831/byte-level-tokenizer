# Attribution and scope

Author and copyright holder for the included project code: **Fengrui Zhang**.

The project originated as tokenizer coursework. The custom work includes shared byte Trie matching, greedy/minimum-token segmentation, strict decoding, experiment and analysis utilities, integration implementations and tests. The author has confirmed permission to publish the self-written code and has selected MIT for the included release.

The original experiment reused a course-provided BPE training baseline. The baseline, checker, benchmark/comparator implementation, original assignment, course datasets and report are not redistributed here. References or imports naming those upstream components describe dependencies and do not provide their code or grant their license.

The original report disclosed OpenAI Codex assistance with implementation, tests, experiments, drafting and audit, and DeepSeek assistance with DP selection, reruns and analysis. Exact model variants were not independently verified. Release preparation used Codex to separate the runnable core, preserve the custom historical pipeline, add the standalone CLI/manual fixtures and perform lightweight validation. This disclosure is retained rather than claiming every line was written without assistance.

The original report also identified the Hugging Face BPE tutorial and GPT-2 documentation as conceptual references. No pretrained vocabulary, Transformers dependency or language-model call is included.

The historical core, comparison wrappers, experiment modules and full tests are preserved from the supplied source. The historical final submission adapter retains its implementation while omitting exercise/setup/contract docstrings. The standalone core changes explanatory wording only; its algorithm implementation is preserved. The root tests select the 26 methods that do not need upstream course components.

The new example vocabulary and JSONL are small, manually specified teaching fixtures. They are not course data or trained model artifacts. Original raw logs and learned models are not supplied, and historical report metrics have not been reproduced for this release.
