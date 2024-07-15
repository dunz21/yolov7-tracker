# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import  detect

from .model import YOLO, YOLOWorld

__all__ = "classify", "segment", "detect", "pose", "obb", "YOLO", "YOLOWorld"
