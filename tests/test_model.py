import pytest

torch = pytest.importorskip("torch")

from fpsd_reassessment.models import FPSDCNN, parameter_count, validate_architecture


def test_checkpoint_compatible_parameter_count_and_shape():
    model = FPSDCNN()
    assert parameter_count(model) == 141_732
    assert validate_architecture(model) == 141_732
    assert tuple(model(torch.zeros(4, 95)).shape) == (4, 2)


def test_shared_starrelu_instance_adds_two_parameters_only():
    model = FPSDCNN()
    activation = [name for name, _ in model.named_parameters() if name.startswith("activation.")]
    assert activation == ["activation.scale", "activation.bias"]
