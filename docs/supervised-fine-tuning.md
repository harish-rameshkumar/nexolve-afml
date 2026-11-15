# Supervised fine-tuning

Supervised fine-tuning starts with each pretrained model and teaches it to answer user instructions. The [training code](../sft/) uses a shared tokenizer and instruction dataset. It computes loss on assistant response tokens and the END marker.

The first 100,000-token pilots saved final model weights and intermediate validation loss. For the final experiment, all six runs saved weights at exactly **33,000, 67,000 and 100,000 applied response tokens**. An applied token contributes to an optimizer update that succeeded. Tokens from skipped updates are tracked separately.

The saved weights are the input to [Instruction evaluation](instruction-evaluation.md). Evaluation also loads the original pretrained weights for the zero-token baseline. Validation loss measures how well the model predicts reference answers. The evaluation generates and scores new answers to measure whether the model follows the instruction.

## Run the pilot pipeline

From the repository root, install `requirements.txt` and prepare the tokenizer and a pretrained checkpoint using the [pretraining guide](pretraining.md). The pilot trainer requires CUDA.

```sh
python -m sft.prepare_data
python -m sft.split_pilot
python -m sft.train --arm T --seed 0 --budget 100000
```

By default, the commands use `data/pretrain/tokenizer.json`, `data/sft/`, `outputs/pretrain/` and `outputs/sft/`. To use assets stored elsewhere, set `NEXOLVE_TOKENIZER`, `NEXOLVE_SFT_DATA`, `NEXOLVE_PRETRAINED`, or `NEXOLVE_SFT_OUTPUT` before running the commands. Point `NEXOLVE_PRETRAINED` to the folder containing run directories such as `T_s0/`.

These commands run the pilot recipe. Use the [saved snapshots](instruction-evaluation.md#run-the-evaluation) to reproduce the paper's benchmark.
