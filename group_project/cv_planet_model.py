from .cv_detections import PlanetEnum
import torch
import torch.nn


model_outputs = [
    PlanetEnum.EARTH,
    PlanetEnum.MOON,
    PlanetEnum.OTHER,
    PlanetEnum.OTHER,
    PlanetEnum.OTHER,
    PlanetEnum.OTHER,
    PlanetEnum.OTHER,
    PlanetEnum.OTHER,
    PlanetEnum.OTHER,
]

local_model = False


def get_model_path():
    if local_model:
        return "./supporting_files/planet_model.pt"

    from ament_index_python.packages import get_package_share_directory
    import os

    package_path = get_package_share_directory("group_project")
    model_path = os.path.join(package_path, "supporting_files", "planet_model.pt")

    return model_path


def set_local_model(value: bool):
    global local_model
    local_model = value


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hydrate_mode():

    # import torch.nn as nn
    # import torchvision.models as models

    # model = models.mobilenet_v2(weights=True)

    # for param in model.parameters():
    #     param.requires_grad = False

    # num_features = model.classifier[1].in_features
    # model.classifier = nn.Sequential(
    #     nn.Dropout(p=0.2),
    #     nn.Linear(num_features, len(model_outputs)),
    # )

    # model

    # torch.load()

    model = torch.load(get_model_path(), map_location=device)

    return model


model = None


def get_model():
    global model

    if model is None:
        model = hydrate_mode()
        model.to(device)

    return model


def predict_planet(image):

    import cv2
    import numpy as np

    image = cv2.resize(image, (50, 50))
    image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    image = np.transpose(
        image, (2, 0, 1)
    )  # Rearrange dimensions to (channels, height, width)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    image = image.to(device)

    model = get_model()
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_planet_index = predicted.item()
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][
            predicted_planet_index
        ].item()

    predicted_planet = model_outputs[predicted_planet_index]
    return predicted_planet, confidence
