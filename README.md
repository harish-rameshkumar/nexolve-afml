# Nexolve AFML

**Does Differential Transformer V2 learn to follow instructions with fewer supervised response tokens?**

We trained three small language-model designs from scratch, fine-tuned them on the same instruction data, and compared their answers at fixed training budgets. The original instruction benchmark showed no consistent advantage for Differential Transformer V2. Follow-up experiments found that the ranking depended on the task and that all three models struggled when a familiar instruction was rephrased.

## Models and experiment design

| Model | Attention | Parameters | Role |
|---|---|---:|---|
| T | Standard causal attention | 27,662,848 | Standard baseline with the same feed-forward width as D |
| D | Differential Transformer V2 | 29,260,288 | Architecture under study |
| P | Standard causal attention with a wider feed-forward network | 29,284,864 | Control for total parameter count |

P is within 0.084% of D in total size. D puts its additional capacity in attention, while P puts it in the feed-forward network. Each design was run with two seeds. Within each seed, the models used the same data order and training recipe. See the [architecture rationale](docs/design_rationale.md).

1. **Pretrain on text.** Each model learned next-token prediction on about 301.5 million FineWeb-Edu tokens using the same tokenizer. See [Pretraining](docs/pretraining.md).
2. **Fine-tune on instructions.** Starting from the pretrained weights, each model learned from assistant responses. We saved weights at 33,000, 67,000, and 100,000 applied response tokens. An applied token contributes to a successful optimizer update. See [Supervised fine-tuning](docs/supervised-fine-tuning.md).
3. **Evaluate generated answers.** The pretrained model gives the zero-token baseline. At each budget, we generated answers to the same 50 frozen prompts and checked explicit constraints such as a required keyword or output length. We compared the entire learning curve across paired seeds. See [Instruction evaluation](docs/instruction-evaluation.md).

## Results at a glance

- **Original instruction benchmark.** D had a lower learning-curve score than T in both seeds, and its ranking against P changed by seed. All 1,200 responses reached the generation limit. Most passes came from keyword inclusion, which limits what this benchmark says about completed answers.
- **Two-fact retrieval.** D had the highest score across the tested budgets in both seeds, averaging 2.4 percentage points above T and 5.4 above P. T scored higher at the final checkpoint in the original wording.
- **Conflicting examples.** On the familiar prompt format, P reached the 50% accuracy target at 192 supervised token exposures in both seeds. D reached it at 256 and 288. These are measured checkpoints, and the counts include repeated exposures.
- **Wording transfer.** At the final conflict checkpoint, all six runs got every pair right in two example layouts with the familiar wording. Across two equivalent rewordings, they passed 23 of 576 pair evaluations.

The results support task-specific comparisons at this scale. The two paired seeds and sensitivity to wording leave the broader efficiency question open. See [Results and limitations](docs/results.md), the [benchmark scores](docs/benchmark-scores.csv), and the [learning curves](docs/figures/instruction-learning-curves.png).

## Try it from a fresh checkout

From the repository root, run this example without model weights or training data:

```sh
python -m evaluation.demo --out outputs/evaluation-demo
```

Open `outputs/evaluation-demo/report.html`. The demo scores fixture responses to show how the evaluation works. For the project tests, install the dependencies first:

```sh
python -m pip install -r requirements.txt
python -m pytest tests sft/test_masked_loss.py -q
```

## Reproduce the results

The repository includes the [frozen benchmark](evaluation/benchmarks/instruction_constraints_v1/), scorer, evaluation code, and summary scores. For the original checkpoint evaluation, obtain the six pretrained checkpoints, the fine-tuning snapshots, the exact tokenizer, and the training record from the project team. The [evaluation guide](docs/instruction-evaluation.md#run-the-evaluation) shows where to place the files and how to run the evaluator. It verifies artifact hashes and checkpoint token counts.

Reproducing training or validation loss also requires the corresponding datasets. The follow-up results are summarized here. Their frozen protocols, scripts, and raw predictions are in the team's separate evidence archive.

## Repository map

- [models/](models/) and [configs/](configs/) define the three architectures.
- [pretraining/](pretraining/) prepares text and trains the base models.
- [sft/](sft/) prepares instructions and fine-tunes the models.
- [evaluation/](evaluation/) contains the benchmark, scoring, and analysis.
- [docs/](docs/) contains the guides, results, and figures.
