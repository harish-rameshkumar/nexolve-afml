# Nexolve AFML

**Does Differential Transformer V2 learn to follow instructions with less supervision?**

We compare three small language models trained from scratch on FineWeb-Edu and then fine-tuned on instruction-response examples. Each architecture has two runs. Within each seed, the architectures use the same training data order and settings.

| Model | Attention | Parameters |
|---|---|---:|
| T | Standard causal attention | 27,662,848 |
| D | Differential Transformer V2 | 29,260,288 |
| P | Standard causal attention with a wider feed-forward network | 29,284,864 |

P controls for total parameter count: it is within 0.084% of D. The [architecture notes](docs/design_rationale.md) explain the shared backbone and what changes between models.

## How the project fits together

1. **Build the models.** The shared model and checkpoint interfaces are in [models/](models/).
2. **Pretrain on text.** Train each model to predict the next token and save its weights. See [Pretraining](docs/pretraining.md).
3. **Fine-tune on instructions.** Start from the pretrained weights and train on assistant responses. Save weights at fixed token budgets. See [Supervised fine-tuning](docs/supervised-fine-tuning.md).
4. **Evaluate generated answers.** Load the saved weights and compare instruction scores at the same budgets. See [Instruction evaluation](docs/instruction-evaluation.md).

## What we found

Differential Transformer V2 showed no consistent advantage on the original instruction benchmark. It led on a small synthetic retrieval learning curve. The parameter-matched standard-attention model reached the accuracy target sooner on the familiar instruction-conflict task. Rephrasing that instruction sharply reduced accuracy across architectures. [Results and limitations](docs/results.md)

## Run the demo and tests

From the repository root, run the demo with Python 3.12:

```sh
python -m evaluation.demo --out outputs/evaluation-demo
```

Open `outputs/evaluation-demo/report.html` to see how the scorer handles example responses. The demo uses fixtures, so it runs without checkpoints or training data.

To run the test suite, install the dependencies first:

```sh
python -m pip install -r requirements.txt
python -m pytest tests sft/test_masked_loss.py -q
```

## Datasets

- [FineWeb-Edu, sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/sample-10BT) for pretraining. See [data preparation and validation split](docs/pretraining.md).
- [SmolTalk, smol-magpie-ultra](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/smol-magpie-ultra) for supervised fine-tuning. See [fine-tuning data and token accounting](docs/supervised-fine-tuning.md).

These links point to the upstream datasets. Use the exact prepared artifacts from the team to reproduce the reported runs.

## Reproduce the evaluation

GitHub contains the [frozen prompts and scorer](evaluation/benchmarks/instruction_constraints_v1/), the [published scores](docs/benchmark-scores.csv), and the [learning curves](docs/figures/instruction-learning-curves.png). To rerun the original 50-prompt benchmark, obtain the six pretrained checkpoints, the 18 fine-tuning snapshots, the exact tokenizer, and the training record from the project team. The [evaluation guide](docs/instruction-evaluation.md#run-the-evaluation) shows the folder layout and command. The evaluator checks artifact hashes and applied token counts before scoring.

The fine-tuning guide describes the final snapshot budgets. The trainer checked into this repository is the pilot version. Reproducing the final fine-tuning runs from scratch requires the snapshot-producing trainer and data from the team handoff. The follow-up protocols, scripts, and predictions are in the team's separate evidence archive.

See the [reproduction checklist](docs/reproduction.md) for the tested environment and the files needed to audit each result.
