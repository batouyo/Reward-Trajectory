import pytest
import torch

from diffusers.pipelines.rewardflow.kontext_text_steering import (
    apply_kontext_pooled_steering,
    apply_kontext_text_steering,
    build_single_pair_clip_direction,
    build_single_pair_t5_direction,
    find_t5_phrase_indices,
    reversion_fraction_to_alpha,
)


class _MockTokenizer:
    def __init__(self):
        self._ids = {"The": 1, "weighted": 2, "training": 3, "ball": 4, "is": 5, "black": 6, "blue": 7, "very": 8}
        self._tokens = {value: key for key, value in self._ids.items()}

    def __call__(self, texts, *, max_length, return_offsets_mapping, **kwargs):
        assert kwargs["padding"] == "max_length"
        assert kwargs["truncation"] is True
        text = texts[0]
        ids, offsets = [], []
        for word in text.split(" "):
            start = text.index(word, offsets[-1][1] if offsets else 0)
            ids.append(self._ids[word.rstrip(".")])
            offsets.append((start, start + len(word.rstrip("."))))
        ids = ids[:max_length]
        offsets = offsets[:max_length]
        ids.extend([0] * (max_length - len(ids)))
        offsets.extend([(0, 0)] * (max_length - len(offsets)))
        return {"input_ids": torch.tensor([ids]), "offset_mapping": torch.tensor([offsets])}

    def convert_ids_to_tokens(self, ids):
        return [self._tokens.get(int(value), "<pad>") for value in ids]


class _MockPipe:
    def __init__(self, same=False):
        self.tokenizer_2 = _MockTokenizer()
        self.text_encoder_2 = type("Encoder", (), {"dtype": torch.float32, "device": torch.device("cpu")})()
        self._execution_device = torch.device("cpu")
        self.same = same

    def _get_t5_prompt_embeds(self, prompt, num_images_per_prompt, max_sequence_length, device, dtype):
        values = torch.zeros(1, max_sequence_length, 4, dtype=torch.float32)
        if not self.same and "blue" in prompt:
            values[0, 5] = torch.tensor([2.0, 0.0, 0.0, 0.0])
        else:
            values[0, 5] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        return values

    def _get_clip_prompt_embeds(self, prompt, num_images_per_prompt, device):
        assert num_images_per_prompt == 1
        return torch.tensor([[1.0 if self.same or "black" in prompt else 4.0, 0.0]])


def test_factor_zero_is_exact_and_does_not_mutate_base():
    base = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    before = base.clone()
    output = apply_kontext_text_steering(base, [1, 4], torch.ones(4), 0.0)
    assert torch.equal(output, base)
    assert torch.equal(base, before)


def test_only_selected_rows_change_and_multiple_indices_are_supported():
    base = torch.zeros(2, 6, 4)
    output = apply_kontext_text_steering(base, [1, 4], torch.tensor([1.0, 2.0, 3.0, 4.0]), -0.5)
    assert torch.equal(output[:, [0, 2, 3, 5]], base[:, [0, 2, 3, 5]])
    expected = (-0.5 * torch.arange(1.0, 5.0)).expand(2, -1)
    torch.testing.assert_close(output[:, 1], expected)
    torch.testing.assert_close(output[:, 4], expected)


def test_phrase_alignment_supports_single_and_multi_token_spans():
    tokenizer = _MockTokenizer()
    assert find_t5_phrase_indices(tokenizer, "The blue ball is", "blue", 8) == [1]
    assert find_t5_phrase_indices(tokenizer, "The very blue ball is", "blue ball", 8) == [2, 3]


def test_phrase_alignment_covers_multiple_subtokens_of_one_word():
    class SubtokenTokenizer(_MockTokenizer):
        def __call__(self, texts, *, max_length, return_offsets_mapping, **kwargs):
            assert texts == ["The ultramarine ball"]
            ids = [1, 9, 10, 4] + [0] * (max_length - 4)
            offsets = [(0, 3), (4, 9), (9, 15), (16, 20)] + [(0, 0)] * (max_length - 4)
            return {"input_ids": torch.tensor([ids]), "offset_mapping": torch.tensor([offsets])}

    assert find_t5_phrase_indices(SubtokenTokenizer(), "The ultramarine ball", "ultramarine", 8) == [1, 2]


def test_alignment_failure_reports_phrase_context():
    with pytest.raises(ValueError, match="original text"):
        find_t5_phrase_indices(_MockTokenizer(), "The blue ball is", "green", 8)


def test_direction_is_normalized_and_tracks_source_target_indices():
    result = build_single_pair_t5_direction(
        _MockPipe(),
        "The weighted training ball is black.",
        "The weighted training ball is blue.",
        "black",
        "blue",
        8,
    )
    assert result.source_token_indices == (5,)
    assert result.target_token_indices == (5,)
    torch.testing.assert_close(torch.linalg.vector_norm(result.direction), torch.tensor(1.0))
    assert result.raw_direction_norm > 0


def test_zero_raw_direction_fails():
    with pytest.raises(ValueError, match="zero"):
        build_single_pair_t5_direction(
            _MockPipe(same=True),
            "The weighted training ball is black.",
            "The weighted training ball is blue.",
            "black",
            "blue",
            8,
        )


def test_direction_dimension_must_match_base_prompt_embeds():
    with pytest.raises(ValueError, match="dimension"):
        build_single_pair_t5_direction(
            _MockPipe(),
            "The weighted training ball is black.",
            "The weighted training ball is blue.",
            "black",
            "blue",
            8,
            torch.zeros(1, 8, 5),
        )


def test_nonfinite_and_dimension_mismatch_fail():
    base = torch.zeros(1, 4, 3)
    with pytest.raises(ValueError, match="finite"):
        apply_kontext_text_steering(base, [1], torch.tensor([float("nan"), 0.0, 0.0]), 1.0)
    with pytest.raises(ValueError, match="dimension"):
        apply_kontext_text_steering(base, [1], torch.zeros(4), 1.0)


def test_out_of_range_and_duplicate_indices_fail():
    base = torch.zeros(1, 4, 3)
    with pytest.raises(ValueError, match="unique"):
        apply_kontext_text_steering(base, [1, 1], torch.ones(3), 1.0)
    with pytest.raises(ValueError, match="out-of-range"):
        apply_kontext_text_steering(base, [4], torch.ones(3), 1.0)


def test_reversion_fraction_is_text_space_displacement_only():
    assert reversion_fraction_to_alpha(0, 4.5) == 0
    assert reversion_fraction_to_alpha(0.5, 4.5) == -2.25
    assert reversion_fraction_to_alpha(1, 4.5) == -4.5
    with pytest.raises(ValueError, match="nonnegative"):
        reversion_fraction_to_alpha(-0.1, 4.5)
    with pytest.raises(ValueError, match="positive"):
        reversion_fraction_to_alpha(0.5, 0)


def test_clip_direction_uses_native_pooled_output_and_is_normalized():
    direction = build_single_pair_clip_direction(_MockPipe(), "ball black", "ball blue", torch.zeros(1, 2))
    assert direction.raw_direction_norm == 3.0
    torch.testing.assert_close(torch.linalg.vector_norm(direction.direction), torch.tensor(1.0))
    assert direction.normalized_direction_norm == 1.0


def test_clip_zero_and_nonfinite_directions_hard_fail():
    with pytest.raises(ValueError, match="zero"):
        build_single_pair_clip_direction(_MockPipe(same=True), "ball black", "ball blue")
    with pytest.raises(ValueError, match="dimension"):
        build_single_pair_clip_direction(_MockPipe(), "ball black", "ball blue", torch.zeros(1, 3))
    base = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="finite"):
        apply_kontext_pooled_steering(base, torch.tensor([float("nan"), 0.0]), -1.0)
    with pytest.raises(ValueError, match="dimension"):
        apply_kontext_pooled_steering(base, torch.ones(3), -1.0)


def test_clip_apply_zero_immutable_and_actual_perturbation_norm():
    base = torch.tensor([[1.0, 2.0]])
    before = base.clone()
    direction = torch.tensor([0.6, 0.8])
    zero = apply_kontext_pooled_steering(base, direction, 0)
    assert torch.equal(zero, base)
    changed = apply_kontext_pooled_steering(base, direction, -2.0)
    assert torch.equal(base, before)
    torch.testing.assert_close(torch.linalg.vector_norm(changed - base), torch.tensor(2.0))


def test_t5_and_clip_updates_are_independent():
    base_t5 = torch.zeros(1, 4, 2)
    base_clip = torch.zeros(1, 3)
    changed_t5 = apply_kontext_text_steering(base_t5, [2], torch.tensor([1.0, 0.0]), -1.0)
    changed_clip = apply_kontext_pooled_steering(base_clip, torch.tensor([0.0, 1.0, 0.0]), -2.0)
    assert torch.equal(base_t5, torch.zeros_like(base_t5))
    assert torch.equal(base_clip, torch.zeros_like(base_clip))
    assert torch.equal(changed_t5[:, [0, 1, 3]], base_t5[:, [0, 1, 3]])
    assert changed_t5[0, 2, 0] == -1.0
    assert changed_clip[0, 1] == -2.0
