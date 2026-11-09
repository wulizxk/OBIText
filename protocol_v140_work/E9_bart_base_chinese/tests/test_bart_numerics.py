"""Independent native-forward oracles for BART-only numerical contracts."""
import copy
import torch

from ed3.data import training_batch
from ed3.model import restricted_logits, resize_embeddings
from tiny_checkpoint import load_tiny_checkpoint


def test_right_shift_matches_native_labels_helper_including_padding(tiny):
    samples = [{'input_ids': [101, 5, 102], 'labels': [5, 8]},
               {'input_ids': [101, 6, 7, 102], 'labels': [7, 6, 8]}]
    batch = training_batch(samples, {5: 0, 6: 1, 7: 2, 8: 3}, 0, 8, 'cpu')
    labels = torch.full_like(batch['decoder_input_ids'], -100)
    for index, sample in enumerate(samples):
        labels[index, :len(sample['labels'])] = torch.tensor(sample['labels'])
    native = tiny.prepare_decoder_input_ids_from_labels(labels)
    # The task excludes END_GAP at padded query slots; these slots are masked.
    active = batch['decoder_attention_mask'].bool()
    assert torch.equal(native[active], batch['decoder_input_ids'][active])
    assert batch['decoder_input_ids'][:, 0].tolist() == [102, 102]
    assert batch['decoder_attention_mask'].sum(1).tolist() == [2, 3]


def test_restricted_projection_includes_nonzero_bias_and_has_no_t5_scale(tiny):
    tiny.eval()
    with torch.no_grad():
        tiny.final_logits_bias.copy_(torch.linspace(-2., 2., 128).unsqueeze(0))
    batch = training_batch([{'input_ids': [101, 5, 102], 'labels': [5, 8]}],
                           {5: 0, 8: 1}, 0, 8, 'cpu')
    candidates = torch.tensor([8, 5])  # Noncontiguous, noncanonical native IDs.
    with torch.no_grad():
        native = tiny(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                      decoder_input_ids=batch['decoder_input_ids'],
                      decoder_attention_mask=batch['decoder_attention_mask'], use_cache=False).logits
        expected = native[batch['owners'], batch['positions']][:, candidates]
        actual = restricted_logits(tiny, batch['input_ids'], batch['attention_mask'],
            batch['owners'], batch['positions'], candidates,
            batch['decoder_input_ids'], batch['decoder_attention_mask'])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_resize_preserves_nonzero_bias_and_checkpoint_ties(tiny):
    with torch.no_grad():
        tiny.final_logits_bias.copy_(torch.arange(128).unsqueeze(0))
    resize_embeddings(tiny, 138)
    assert torch.equal(tiny.final_logits_bias[:, :128], torch.arange(128).unsqueeze(0))
    assert not torch.count_nonzero(tiny.final_logits_bias[:, 128:])
    checkpoint = {'model_config': tiny.config.to_dict(), 'model': copy.deepcopy(tiny.state_dict())}
    restored = load_tiny_checkpoint(checkpoint, 'cpu')
    assert restored.model.shared.weight is restored.lm_head.weight
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in tiny.state_dict().items())
