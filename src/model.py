"""Construction du modèle."""
import timm


def build_model(name: str, num_classes: int, pretrained: bool = True):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
