# Supervised fine-tuning

Supervised fine-tuning starts with each pretrained model and teaches it to answer user instructions. The [training code](../sft/) uses a shared tokenizer and instruction dataset. It computes loss on assistant response tokens and the END marker.

The first 100,000-token pilots saved final model weights and intermediate validation loss. For the final experiment, all six runs saved weights at exactly **33,000, 67,000 and 100,000 applied response tokens**. An applied token contributes to an optimizer update that succeeded. Tokens from skipped updates are tracked separately.

The saved weights are the input to [Instruction evaluation](instruction-evaluation.md). Evaluation also loads the original pretrained weights for the zero-token baseline. Validation loss measures how well the model predicts reference answers. The evaluation generates and scores new answers to measure whether the model follows the instruction.
