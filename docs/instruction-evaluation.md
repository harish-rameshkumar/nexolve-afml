# Instruction evaluation

We tested the same 50 prompts at four training budgets: 0, 33,000, 67,000 and 100,000 applied assistant-response tokens. The zero-token checkpoint is the pretrained model. The [frozen benchmark](../evaluation/benchmarks/instruction_constraints_v1/) uses greedy generation with a 64-token limit and checks one constraint per prompt.

## Run the evaluation

1. Clone this repository and install the dependencies with `python -m pip install -r requirements.txt`.
2. Download and extract the [saved models, tokenizer, and training records](https://drive.google.com/drive/folders/1JhTpxHqsIH7sBsVe907WTf2UmF09CBut). Keep the subfolders as downloaded. Pass the parent folder containing `01_TRAINING_RECORDS`, `02_PRETRAINED_CHECKPOINTS`, `03_SFT_CHECKPOINTS`, and `04_DATA_AND_TOKENIZER` as `--asset-root`.
3. From the repository root, run this quick check with one real checkpoint per architecture:

```sh
python -m evaluation.evaluate_instruction_checkpoints --asset-root "PATH_TO_DOWNLOADED_FOLDER" --protocol evaluation/benchmarks/instruction_constraints_v1/protocol.json --out outputs/checkpoint-check.json --quick-check
```

For the full paper benchmark, use all 24 checkpoints and all 50 prompts:

```sh
python -m evaluation.evaluate_instruction_checkpoints --asset-root "PATH_TO_DOWNLOADED_FOLDER" --protocol evaluation/benchmarks/instruction_constraints_v1/protocol.json --out outputs/instruction-scores.json
```

The full run writes individual predictions, pass rates, generation-limit counts, and learning-curve area. It runs on CPU and took about 15 minutes of generation in the recorded environment. The evaluator verifies file hashes and token counts. Compare the output with the [published scores](benchmark-scores.csv) and read the [results and limitations](results.md).

For a quick illustration without saved models, run `python -m evaluation.demo --out outputs/evaluation-demo` and open the generated report.
