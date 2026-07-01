from scienceworld_mas.training.collator import IGNORE_INDEX, SupervisedDataCollator, example_to_features
from scienceworld_mas.training.examples import TrainingExample


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize
        assert add_generation_prompt
        return [11, 12, 13]

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


def test_example_to_features_masks_prompt_only():
    features = example_to_features(
        FakeTokenizer(),
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "ok"},
        ],
        max_length=32,
    )

    assert features["input_ids"][:3] == [11, 12, 13]
    assert features["labels"][:3] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    assert features["labels"][3:] == features["input_ids"][3:]
    assert features["attention_mask"] == [1] * len(features["input_ids"])


def test_example_to_features_left_truncates():
    features = example_to_features(
        FakeTokenizer(),
        [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "abcdef"},
        ],
        max_length=4,
    )

    assert len(features["input_ids"]) == 4
    assert features["input_ids"] == features["labels"]


def test_supervised_collator_pads_inputs_and_masks_padding():
    short = TrainingExample(
        role="system1",
        messages=(
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "x"},
        ),
        source={},
    )
    long = TrainingExample(
        role="system1",
        messages=(
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "xy"},
        ),
        source={},
    )

    batch = SupervisedDataCollator(FakeTokenizer(), max_length=32)([short, long])

    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert batch["input_ids"][0][-1] == FakeTokenizer.pad_token_id
    assert batch["attention_mask"][0][-1] == 0
    assert batch["labels"][0][-1] == IGNORE_INDEX
