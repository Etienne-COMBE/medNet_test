from ast import Index
import pytest

from predict import classNames
from predict import *

def test_classes():
    assert classNames == ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']

@pytest.mark.parametrize(
    "classes",
    [
        (["1", "2", "3", "4", "5"]),
        ([1, 2, 3, 4, 5, 6, 7])
    ]
)
def test_classes_error(classes):
    with pytest.raises(IndexError):
        predict_image(None, classes)


