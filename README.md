# Nexolve AFML

**Does Differential Transformer V2 learn to follow instructions with less supervision?**

We compare three small language models trained from scratch on FineWeb-Edu and then fine-tuned on instruction-response examples. Each architecture has two runs. Within each seed, the architectures use the same training data order and settings.

| Model | Attention | Parameters |
|---|---|---:|
| T | Standard causal attention | 27,662,848 |
| D | Differential Transformer V2 | 29,260,288 |
| P | Standard causal attention with a wider feed-forward network | 29,284,864 |

P controls for total parameter count: it is within 0.084% of D. The [architecture notes](docs/design_rationale.md) explain the shared backbone and what changes between models.

## Results

Differential Transformer V2 showed no consistent advantage on the 50-prompt instruction benchmark. It led on the synthetic two-fact retrieval learning curve. The parameter-matched control reached the accuracy target sooner on the familiar instruction-conflict task. Rephrasing the instruction sharply reduced accuracy across all three models.

[Published benchmark scores](docs/benchmark-scores.csv) · [Learning curves](docs/figures/instruction-learning-curves.png) · [Results and limitations](docs/results.md)

## Reproduce the evaluation

Use Python 3.12. Run the commands from the repository root unless a step changes directories.

### 1. Set up and run the demo

```sh
git clone https://github.com/harish-rameshkumar/nexolve-afml.git
cd nexolve-afml
python -m pip install -r requirements.txt
python -m evaluation.demo --out outputs/evaluation-demo
python -m pytest tests sft/test_masked_loss.py -q
```

Open `outputs/evaluation-demo/report.html` to see how example answers are scored. The demo and tests use small fixtures and require no downloaded checkpoints.

### 2. Run the paper benchmark from saved checkpoints

Download the [saved checkpoints, tokenizer, and training records](https://drive.google.com/drive/folders/1JhTpxHqsIH7sBsVe907WTf2UmF09CBut). Extract the download and keep its folder structure. Replace `PATH_TO_DOWNLOADED_FOLDER` below with the path to the folder that directly contains:

```text
01_TRAINING_RECORDS/
02_PRETRAINED_CHECKPOINTS/
03_SFT_CHECKPOINTS/
04_DATA_AND_TOKENIZER/
```

First check that the evaluator can load real checkpoints and score responses:

```sh
python -m evaluation.evaluate_instruction_checkpoints --asset-root "PATH_TO_DOWNLOADED_FOLDER" --protocol evaluation/benchmarks/instruction_constraints_v1/protocol.json --out outputs/checkpoint-check.json --quick-check
```

Then run all 50 prompts on all 24 checkpoints. The four budgets are 0, 33,000, 67,000, and 100,000 applied assistant-response tokens. The zero-token checkpoint is the pretrained starting model.

```sh
python -m evaluation.evaluate_instruction_checkpoints --asset-root "PATH_TO_DOWNLOADED_FOLDER" --protocol evaluation/benchmarks/instruction_constraints_v1/protocol.json --out outputs/instruction-scores.json
```

The evaluator checks checkpoint and tokenizer hashes and uses greedy decoding with a 64-new-token limit. The output JSON contains generated answers, pass rates, generation-limit counts, and learning-curve areas. Compare the pass rates with the [published checkpoint scores](docs/benchmark-scores.csv). The full run took about 15 minutes of CPU generation in the recorded environment.

### 3. Recalculate the synthetic follow-up results

Run these commands from the repository root. They recalculate the two-fact retrieval, instruction-conflict, and wording-transfer results from saved predictions in the [evaluation evidence archive](docs/evaluation-evidence.zip).

```sh
python -m zipfile -e docs/evaluation-evidence.zip outputs/evaluation-evidence
cd outputs/evaluation-evidence
python work/summarize_ws4_two_fact.py
python work/summarize_ws4_instruction_conflict.py
python work/summarize_ws4_conflict_transfer.py
```

The scripts print the scores and comparisons. This step requires only Python and the archive.

## Training and project files

- [Model implementation](models/) and [architecture notes](docs/design_rationale.md)
- [Pretraining guide](docs/pretraining.md) and [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/sample-10BT)
- [Supervised fine-tuning guide](docs/supervised-fine-tuning.md) and [SmolTalk dataset](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/smol-magpie-ultra)
- [Reproduction notes](docs/reproduction.md) for the recorded environment, prepared data, and limits on recreating the original training runs

The commands above reproduce evaluation from existing artifacts. Recreating the paper's fine-tuned checkpoints requires the exact trainer used for those runs.
