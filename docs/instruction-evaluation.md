# Instruction evaluation

This stage measures how well the models follow instructions at four points in supervised fine-tuning: **0, 33,000, 67,000 and 100,000 applied response tokens**. The zero-token point uses pretrained weights. Each point uses the same [frozen 50-prompt benchmark](../evaluation/benchmarks/instruction_constraints_v1/) and greedy decoding with a 64-token limit. The scorer checks a single explicit constraint per prompt.

[Results and limitations](results.md) explains what the comparison showed. [Benchmark scores](benchmark-scores.csv) contains one row per model, seed and budget.

## Run the evaluation

Run `python -m evaluation.demo --out outputs/evaluation-demo` for a fixture-based example. For real checkpoint evaluation, obtain the experiment assets separately. The evaluator expects this layout under the directory passed to `--asset-root`:

```text
work/handoff/sft/tokenizer.json
work/handoff/parent_<model>_s<seed>/final.pt
work/ws3-final/training_record.json
work/ws3-final/<model>_s<seed>/sft_<budget>.pt
```

The model names are T, D and P. The seeds are 0 and 1. The snapshot budgets are 33000, 67000 and 100000.

```sh
python -m evaluation.evaluate_instruction_checkpoints --asset-root PATH_TO_ASSETS --protocol evaluation/benchmarks/instruction_constraints_v1/protocol.json --out outputs/instruction-scores.json
```

The evaluator checks checkpoint hashes and applied-token counters before scoring. The tokenizer, model weights and training data are managed separately from this repository.
