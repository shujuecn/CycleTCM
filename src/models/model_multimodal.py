"""Multimodal CycleTCM uses the shared, numerically verified visual path."""

from models.model_visual import CycleTCM as VisualCycleTCM
from models.model_visual import count_parameters, print_model_info
from models.model_mllm import MLLM_Adapter


class CycleTCM(VisualCycleTCM):
    def __init__(self, num_classes1=8, num_classes2=5, pretrained=False, **kwargs):
        super().__init__(num_classes1, num_classes2, pretrained, mllm=True, **kwargs)
