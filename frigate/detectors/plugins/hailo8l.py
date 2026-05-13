"""Deprecated: use 'type: hailo' instead. This shim maintains backward compatibility."""

import logging
import os
from typing import Optional

from pydantic import ConfigDict, Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from frigate.detectors.plugins.hailo import HailoDetector

logger = logging.getLogger(__name__)

DETECTOR_KEY = "hailo8l"


class Hailo8lDetector(DetectionApi):
    """Backward-compatible detector for 'type: hailo8l' configs."""

    type_key = DETECTOR_KEY

    def __init__(self, detector_config: "Hailo8lDetectorConfig"):
        logger.warning(
            "'type: hailo8l' is deprecated and will be removed in a future release. "
            "Please update your config to use 'type: hailo' instead."
        )
        self._delegate = HailoDetector(detector_config)

    def detect_raw(self, tensor_input):
        return self._delegate.detect_raw(tensor_input)

    def close(self):
        if hasattr(self, "_delegate"):
            self._delegate.close()

    def __del__(self):
        self.close()


class Hailo8lDetectorConfig(BaseDetectorConfig):
    """Deprecated config for hailo8l detector type."""

    model_config = ConfigDict(
        title="Hailo-8/Hailo-8L (deprecated, use 'hailo')",
    )

    type: Literal[DETECTOR_KEY]
    device: str = Field(
        default="PCIe",
        title="Device Type",
        description="The device to use for Hailo inference (e.g. 'PCIe', 'M.2').",
    )
    hailo_arch: Optional[str] = Field(
        default=None,
        title="Hailo Architecture",
        description=(
            "Hailo device architecture: 'hailo8' or 'hailo8l'. "
            "Auto-detected from hardware when not set."
        ),
    )
    multi_process_service: bool = Field(
        default=False,
        title="Multi-Process Service",
        description=(
            "Use HailoRT multi-process service for shared device access. "
            "Requires hailort_service running externally (e.g. hailo-service add-on)."
        ),
    )
    service_address: str = Field(
        default="",
        title="Service Address",
        description=(
            "HailoRT service socket address. Leave empty to use the system default. "
            "Set to 'unix:/share/hailo/hailort_service.sock' for HA with hailo-service add-on. "
            "Only used when multi_process_service is true."
        ),
    )

    def model_post_init(self, __context) -> None:
        if self.multi_process_service and self.service_address:
            os.environ["HAILORT_SERVICE_ADDRESS"] = self.service_address
