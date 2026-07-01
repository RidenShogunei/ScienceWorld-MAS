import pytest

from scienceworld_mas.training.supervised import SupervisedTrainingConfig, train_supervised


def test_train_supervised_validates_lightweight_config_before_loading_models():
    config = SupervisedTrainingConfig(
        base_model="unused",
        data_dir="unused",
        role="system1",
        output_dir="unused",
        batch_size=0,
    )

    with pytest.raises(ValueError, match="batch_size"):
        train_supervised(config)
