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

## Run the checks

Install the project dependencies, then run the tests and a small evaluation demo from the repository root:

```sh
python -m pip install -r requirements.txt
python -m pytest tests sft/test_masked_loss.py -q
python -m evaluation.demo --out outputs/evaluation-demo
```

The demo uses fixture responses. See the evaluation guide for how to run the benchmark with trained weights.
