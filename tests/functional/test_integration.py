import pytest
import os
import PIL
from app import app
import random
import pandas as pd
import numpy as np

from predict import *
from predict import classNames

@pytest.fixture
def client():
    app.config.update({"TESTING": True})

    with app.test_client() as client:
        yield client

# Route Testing
@pytest.mark.parametrize(
    "route, status_code",
    [
        ("/", 200),
        ("/teddy_bear", 404)
    ])
def test_route(client, route, status_code):
    response = client.post(route)
    assert response.status_code == status_code

# Mocking file upload by scouting template
@pytest.mark.parametrize(
    "image, path, method",
    [
        ("simba_resized.png", os.path.join(os.getcwd(),"tests"), 'post'),
        ("simba_resized.png", os.path.join(os.getcwd(),"tests"), 'get'),
        (None, None, 'get')
    ]
)
def test_upload_file(client, image, path, method):
    if method == 'get':
        response = client.get("/")
        assert b"img src" not in response.data
    if method == 'post':
        data = {
            "image": (open(os.path.join(path, image), "rb"), image)
        }

        response = client.post("/", data = data)
        assert b"img src" in response.data

def test_upload_file_error(client):
    with pytest.raises(PIL.UnidentifiedImageError):
        data = {
            "image": open(os.path.join(os.getcwd(),"tests", "simba_resized.txt"), "rb")
        }
        response = client.post("/", data = data)

# Testing model accuracy on different images
def calculate_score(data: list):
    df = pd.DataFrame(data, columns=["image", "label"])
    df["pred"] = df.image.apply(lambda x: predict_image(x, classNames))

    errors = len(df.label.compare(df.pred))
    return 1 - (errors / len(df)), df

def create_dataset(random_ = False):
    if not random_:
        path = os.path.join(os.getcwd(), "resized")
        class_dir = os.listdir(path)
        data = []
        for class_ in class_dir:
            sample = random.sample(os.listdir(os.path.join(path, class_)), 30)
            for img_path in sample:
                img = Image.open(os.path.join(path, class_, img_path))
                data.append([img, class_])
    else:
        data = []
        for _ in range(50):
            img = Image.fromarray(np.random.rand(64, 64) * 255)
            data.append([img, classNames[np.random.randint(6)]])
    return data

@pytest.mark.parametrize(
    "dataset, expected",
    [
        (create_dataset(random_ = False), True),
        (create_dataset(random_ = True), False)
    ]
)
def test_model_score(dataset, expected):
    score, _ = calculate_score(dataset)
    assert (score > 0.99) == expected

# Testing integrity of the confusion matrix
def test_confusion_matrix():
    score, df = calculate_score(create_dataset())
    conf = pd.crosstab(df.label, df.pred)
    assert conf.shape == (6, 6)