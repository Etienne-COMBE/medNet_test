from statistics import mode
import pytest
import numpy as np
import torch
import torch.nn as nn

from medNet import MedNet

def test_constructor():
    assert MedNet(64, 64, 6)

@pytest.mark.parametrize(
    "model_params",
    [
        ([-1, -1, -1]),
        ([64, 64, 0]),
        ([0, 64, 6]),
        ([64, 0, 6]),
    ])
def test_constructor_value(model_params):
    with pytest.raises(ValueError):
        model = MedNet(model_params[0], model_params[1], model_params[2])
    
@pytest.mark.parametrize(
    "model_params",
    [
        (["64", 64, 6]),
        ([64, "64", 6]),
        ([64, 64, "6"]),
        ([np.NaN, np.NaN, np.NaN]),
    ])
def test_constructor_type(model_params):
    with pytest.raises(TypeError):
        model = MedNet(model_params[0], model_params[1], model_params[2])

model = MedNet(64, 64, 6)

@pytest.mark.parametrize(
    "model_attribute, expected",
    [
        (model.cnv1, nn.Conv2d(1, 5, kernel_size=(7, 7), stride=(1, 1))),
        (model.cnv2, nn.Conv2d(5, 10, kernel_size=(3, 3), stride=(1, 1))),
        (model.ful1, nn.Linear(in_features=31360, out_features=200, bias=True)),
        (model.ful2, nn.Linear(in_features=200, out_features=80, bias=True)),
        (model.ful3, nn.Linear(in_features=80, out_features=6, bias=True))
    ]
)
def test_model_attributes(model_attribute, expected):
    assert model_attribute.__eq__(expected)

