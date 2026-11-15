# Reproduction checklist

## Run the demo and tests

Follow the commands in the [README](../README.md#run-the-demo-and-tests). The fixture demo runs without trained weights. The verified Windows CPU evaluation environment used Python 3.12.10, PyTorch 2.14.1+cpu, NumPy 2.5.4, and tokenizers 0.23.3. The test run used pytest 9.2.0, and plot generation used Matplotlib 3.11.2. These are recorded versions from the verification environment. GPU training was not rerun in this environment.

## Recalculate the reported scores

- **50-prompt benchmark.** Use the [published checkpoint scores](benchmark-scores.csv) to inspect pass rates. To regenerate the answers, follow the [evaluation guide](instruction-evaluation.md#run-the-evaluation) with the six pretrained weights, the 18 fine-tuning snapshots, the tokenizer, and the training record from the team. The evaluator verifies artifact hashes and token counters.
- **Synthetic follow-ups.** Request the team's evaluation evidence archive for the frozen prompts, scoring rules, predictions, and scripts for two-fact retrieval, instruction conflict, and wording transfer. The [results summary](results.md) describes each comparison.

## Reproduce training

The [pretraining guide](pretraining.md) explains how to prepare the data and launch pretraining. For exact reproduction, use the team's prepared data and tokenizer and compare hashes against the recorded manifest.

The checked-in fine-tuning trainer is the pilot implementation. The final training record identifies source revision `3de3f28c01c993a32c121b522250d22a3fd15be5` for the rerun that produced the 33K, 67K, and 100K applied-token snapshots. That revision is unavailable in this repository. Obtain the archived trainer, run configuration, and prepared instruction data from the team for a from-scratch rerun. The saved snapshots are sufficient to rerun the original evaluation.
