from .cnn3d import CNN3D
from .resnet3d import resnet18_3d, resnet10_3d
from .densenet3d import densenet121_3d

MODEL_REGISTRY = {
    "cnn3d": CNN3D,
    "resnet3d10": resnet10_3d,
    "resnet3d18": resnet18_3d,
    "densenet3d121": densenet121_3d
}

def build_model(model_name, num_classes=2, in_channels=1, dropout=0.3):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    return model
