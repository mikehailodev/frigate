"""Deprecated: use 'type: hailo' instead. This shim maintains backward compatibility."""

import logging

from typing_extensions import Literal

from frigate.detectors.plugins.hailo import (
    HailoDetector,
    HailoDetectorConfig,
)

logger = logging.getLogger(__name__)

DETECTOR_KEY = "hailo8l"


class Hailo8lDetector(HailoDetector):
    """Backward-compatible detector for 'type: hailo8l' configs."""

    type_key = DETECTOR_KEY

    def __init__(self, detector_config: "Hailo8lDetectorConfig"):
        logger.warning(
            "'type: hailo8l' is deprecated and will be removed in a future release. "
            "Please update your config to use 'type: hailo' instead."
        )
        super().__init__(detector_config)


class Hailo8lDetectorConfig(HailoDetectorConfig):
    """Deprecated config for hailo8l detector type."""

    type: Literal[DETECTOR_KEY]  # type: ignore[assignment]
