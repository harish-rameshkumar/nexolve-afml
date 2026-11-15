# Reproduction checklist

## Run the demo and tests

Follow the commands in the [README](../README.md#run-the-demo-and-tests). The fixture demo runs without trained weights. The verified Windows CPU evaluation environment used Python 3.12.10, PyTorch 2.14.1+cpu, NumPy 2.5.4, and tokenizers 0.23.3. The test run used pytest 9.2.0, and plot generation used Matplotlib 3.11.2. These are recorded versions from the verification environment. GPU training was not rerun in this environment.

## Recalculate the reported scores

- **50-prompt benchmark.** Use the [published checkpoint scores](benchmark-scores.csv) to inspect pass rates. To regenerate the answers, follow the [evaluation guide](instruction-evaluation.md#run-the-evaluation) with the [saved models, tokenizer, and training records](https://drive.google.com/drive/folders/1JhTpxHqsIH7sBsVe907WTf2UmF09CBut). The evaluator verifies artifact hashes and token counters.
- **Synthetic follow-ups.** Download the [evaluation evidence archive](evaluation-evidence.zip) for the frozen prompts, scoring rules, predictions, and scripts for two-fact retrieval, instruction conflict, and wording transfer. The archive includes commands to recalculate scores from saved predictions with Python. The [results summary](results.md) describes each comparison.

## Reproduce training

The [pretraining guide](pretraining.md) explains how to prepare the data and launch pretraining. For exact reproduction, obtain the [prepared pretraining data and tokenizer](https://drive.google.com/drive/folders/1dVN0q7oC_Iq3Foyb72_MtKrz_nwfmmFA) and compare hashes against the recorded manifest.

The repository contains the pilot fine-tuning trainer. The exact trainer that produced the final 33K, 67K, and 100K snapshots is needed to recreate them from scratch. The published evaluation uses the saved snapshots above.
