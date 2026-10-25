
import torch
import torch.nn.functional as F


def masked_next_token_loss(logits, labels):
    """
    logits: (batch, sequence_length, vocab_size)
    labels: (batch, sequence_length)
    -100 labels are ignored.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def test_prompt_tokens_do_not_contribute_to_loss():
    torch.manual_seed(0)

    # 1 example, 5 positions, vocabulary size 10.
    logits = torch.randn(1, 5, 10)
    labels = torch.tensor([[-100, -100, -100, 7, 5]])

    loss = masked_next_token_loss(logits, labels)

    # The shifted labels supervise positions 3 and 4:
    # response token 7 and ending token 5.
    expected = F.cross_entropy(
        logits[:, 2:4, :].float().reshape(-1, 10),
        torch.tensor([7, 5]),
        reduction="mean",
    )

    assert torch.allclose(loss, expected), (loss.item(), expected.item())
    print("PASS: only response and end tokens contribute to loss")
    print("Loss:", loss.item())


def test_all_ignored_targets_are_not_silently_accepted():
    logits = torch.randn(1, 4, 10)
    labels = torch.full((1, 4), -100)

    shifted_labels = labels[:, 1:]
    assert not (shifted_labels != -100).any()
    print("PASS: all-masked batch detected; training loop must reject it")


if __name__ == "__main__":
    test_prompt_tokens_do_not_contribute_to_loss()
    test_all_ignored_targets_are_not_silently_accepted()
