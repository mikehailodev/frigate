import glob
import logging
import os
import re
import threading
import urllib.request
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import ConfigDict, Field
from typing_extensions import Literal

from frigate.const import MODEL_CACHE_DIR
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import (
    BaseDetectorConfig,
)
from frigate.object_detection.util import RequestStore, ResponseStore

logger = logging.getLogger(__name__)


# ----------------- Utility Functions ----------------- #


def preprocess_tensor(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """
    Resize an image with unchanged aspect ratio using padding.
    Assumes input image shape is (H, W, 3).
    """
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    h, w = image.shape[:2]
    scale = min(model_w / w, model_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    padded_image = np.full((model_h, model_w, 3), 114, dtype=image.dtype)
    x_offset = (model_w - new_w) // 2
    y_offset = (model_h - new_h) // 2
    padded_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
        resized_image
    )
    return padded_image


# ----------------- Global Constants ----------------- #
DETECTOR_KEY = "hailo"

# Hailo-8 / Hailo-8L defaults
H8_DEFAULT_MODEL = "yolov6n.hef"
H8L_DEFAULT_MODEL = "yolov6n.hef"
H8_DEFAULT_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8/yolov6n.hef"
H8L_DEFAULT_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/yolov6n.hef"

# PCI device IDs
HAILO_VENDOR_ID = "0x1e60"
HAILO8_DEVICE_ID = "0x2864"


def detect_hailo_hardware() -> bool:
    """Detect Hailo-8/8L hardware by scanning PCI device IDs in sysfs.

    Returns True if a Hailo-8/8L device is found, False otherwise.
    """
    pci_devices_path = Path("/sys/bus/pci/devices")
    if not pci_devices_path.exists():
        logger.warning("Cannot access /sys/bus/pci/devices — PCI scan unavailable")
        return False

    for device_dir in pci_devices_path.iterdir():
        try:
            vendor = (device_dir / "vendor").read_text().strip()
            device_id = (device_dir / "device").read_text().strip()
        except (OSError, IOError):
            continue

        if vendor == HAILO_VENDOR_ID and device_id == HAILO8_DEVICE_ID:
            logger.info(f"Detected Hailo-8/8L device (PCI {device_dir.name})")
            return True

    logger.warning("No Hailo PCI device found")
    return False


def detect_hailo_arch() -> Optional[str]:
    """Detect specific Hailo architecture (hailo8 vs hailo8l) via device properties."""
    try:
        from hailo_platform import Device
        from hailo_platform.pyhailort.pyhailort import DeviceArchitectureTypes

        device = Device()
        board_info = device.control.identify()
        arch_type = board_info.device_architecture
        device.release()

        if arch_type == DeviceArchitectureTypes.HAILO8L:
            return "hailo8l"
        elif arch_type in (
            DeviceArchitectureTypes.HAILO8,
            DeviceArchitectureTypes.HAILO8_A0,
        ):
            return "hailo8"
        else:
            logger.error(f"Unexpected device architecture type: {arch_type}")
            return None
    except Exception as e:
        logger.error(
            f"Failed to detect Hailo-8 vs Hailo-8L: {e}. "
            f"Please set 'hailo_arch: hailo8' or 'hailo_arch: hailo8l' "
            f"in your detector config."
        )
        return None


def check_hailort_version_match() -> Optional[str]:
    """Verify kernel module and userspace library versions match.

    Returns None if versions match (or check is skipped), or an error message string.
    """
    module_path = "/sys/module/hailo_pci/version"

    kernel_version = None
    try:
        with open(module_path) as f:
            kernel_version = f.read().strip()
    except (FileNotFoundError, PermissionError):
        logger.warning(f"Cannot read {module_path} — skipping version check")
        return None

    lib_version = None
    for search_dir in ["/usr/lib/", "/usr/local/lib/"]:
        for path in glob.glob(os.path.join(search_dir, "libhailort.so.4.*.*")):
            match = re.search(r"libhailort\.so\.(\d+\.\d+\.\d+)", path)
            if match:
                lib_version = match.group(1)
                break
        if lib_version:
            break

    if not lib_version:
        logger.warning("Cannot determine libhailort version — skipping version check")
        return None

    if kernel_version != lib_version:
        return (
            f"HailoRT version mismatch: kernel driver (hailo_pci) is v{kernel_version} "
            f"but libhailort is v{lib_version}. HailoRT requires an exact version match. "
            f"Ensure HAOS and this container use the same HailoRT version."
        )

    logger.info(f"HailoRT version match confirmed: v{kernel_version}")
    return None


# ----------------- HailoAsyncInference Class ----------------- #
class HailoAsyncInference:
    def __init__(
        self,
        hef_path: str,
        input_store: RequestStore,
        output_store: ResponseStore,
        batch_size: int = 1,
        input_type: Optional[str] = None,
        output_type: Optional[Dict[str, str]] = None,
        multi_process_service: bool = False,
    ) -> None:
        try:
            from hailo_platform import (
                HEF,
                FormatType,
                HailoSchedulingAlgorithm,
                VDevice,
            )
        except ModuleNotFoundError:
            raise RuntimeError(
                "hailo_platform not found. Ensure HailoRT is installed correctly."
            )

        self.input_store = input_store
        self.output_store = output_store

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        if multi_process_service:
            params.multi_process_service = True
            params.group_id = "SHARED"
            svc_addr = os.environ.get("HAILORT_SERVICE_ADDRESS", "default")
            logger.info(
                f"Using HailoRT multi-process service "
                f"(group_id=SHARED, address={svc_addr})"
            )

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)

        if input_type is not None:
            self.infer_model.input().set_format_type(getattr(FormatType, input_type))

        if output_type is not None:
            for output_name, output_type in output_type.items():
                self.infer_model.output(output_name).set_format_type(
                    getattr(FormatType, output_type)
                )

        self.output_type = output_type

    def callback(
        self,
        completion_info,
        bindings_list: List,
        input_batch: List,
        request_ids: List[int],
    ):
        if completion_info.exception:
            logger.error(f"Inference error: {completion_info.exception}")
        else:
            for i, bindings in enumerate(bindings_list):
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(
                            bindings.output(name).get_buffer(), axis=0
                        )
                        for name in bindings._output_names
                    }
                self.output_store.put(request_ids[i], (input_batch[i], result))

    def _create_bindings(self, configured_infer_model) -> object:
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=getattr(
                        np, str(output_info.format.type).split(".")[1].lower()
                    ),
                )
                for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=getattr(np, self.output_type[name].lower()),
                )
                for name in self.output_type
            }
        return configured_infer_model.create_bindings(output_buffers=output_buffers)

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.hef.get_input_vstream_infos()[0].shape

    def run(self) -> None:
        with self.infer_model.configure() as configured_infer_model:
            while True:
                batch_data = self.input_store.get()

                if batch_data is None:
                    break

                request_id, frame_data = batch_data

                bindings = self._create_bindings(configured_infer_model)
                bindings.input().set_buffer(np.array(frame_data))

                job = configured_infer_model.run_async(
                    [bindings],
                    partial(
                        self.callback,
                        input_batch=[frame_data],
                        request_ids=[request_id],
                        bindings_list=[bindings],
                    ),
                )
                job.wait(10000)


# ----------------- HailoDetector Class ----------------- #
class HailoDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: "HailoDetectorConfig"):
        self.disabled = False
        self.arch = None
        self.multi_process_service = detector_config.multi_process_service

        if self.multi_process_service and detector_config.service_address:
            os.environ["HAILORT_SERVICE_ADDRESS"] = detector_config.service_address

        # Step 1: Detect hardware via PCI scan
        if not detect_hailo_hardware():
            logger.critical("=" * 60)
            logger.critical("HAILO DETECTOR DISABLED")
            logger.critical("No Hailo device detected on PCI bus.")
            logger.critical(
                "Hailo detection is disabled. Recording and streaming "
                "will continue but no objects will be detected."
            )
            logger.critical("=" * 60)
            self.disabled = True
            return

        # Step 2: Version check (kernel driver vs userspace lib)
        version_error = check_hailort_version_match()
        if version_error:
            logger.critical("=" * 60)
            logger.critical("HAILO DETECTOR DISABLED")
            logger.critical(version_error)
            logger.critical(
                "Hailo detection is disabled. Recording and streaming "
                "will continue but no objects will be detected. "
                "Update HAOS or this container so versions match, then restart."
            )
            logger.critical("=" * 60)
            self.disabled = True
            return

        # Step 3: Detect specific architecture (hailo8 vs hailo8l)
        if detector_config.hailo_arch:
            self.arch = detector_config.hailo_arch
            logger.info(f"Using configured Hailo architecture: {self.arch}")
        else:
            self.arch = detect_hailo_arch()
            if self.arch is None:
                logger.critical("=" * 60)
                logger.critical("HAILO DETECTOR DISABLED")
                logger.critical(
                    "Cannot determine device architecture (hailo8 vs hailo8l). "
                    "Set 'hailo_arch' in your detector config."
                )
                logger.critical("=" * 60)
                self.disabled = True
                return
            logger.info(f"Auto-detected Hailo architecture: {self.arch}")

        self.cache_dir = MODEL_CACHE_DIR
        self.device_type = detector_config.device
        self.model_height = (
            detector_config.model.height
            if hasattr(detector_config.model, "height")
            else None
        )
        self.model_width = (
            detector_config.model.width
            if hasattr(detector_config.model, "width")
            else None
        )
        self.model_type = (
            detector_config.model.model_type
            if hasattr(detector_config.model, "model_type")
            else None
        )
        self.tensor_format = (
            detector_config.model.input_tensor
            if hasattr(detector_config.model, "input_tensor")
            else None
        )
        self.pixel_format = (
            detector_config.model.input_pixel_format
            if hasattr(detector_config.model, "input_pixel_format")
            else None
        )
        self.input_dtype = (
            detector_config.model.input_dtype
            if hasattr(detector_config.model, "input_dtype")
            else None
        )
        self.output_type = "FLOAT32"
        self.set_path_and_url(detector_config.model.path)
        self.working_model_path = self.check_and_prepare()

        self.batch_size = 1
        self.input_store = RequestStore()
        self.response_store = ResponseStore()

        try:
            logger.debug(f"[INIT] Loading HEF model from {self.working_model_path}")
            self.inference_engine = HailoAsyncInference(
                self.working_model_path,
                self.input_store,
                self.response_store,
                self.batch_size,
                multi_process_service=self.multi_process_service,
            )
            self.input_shape = self.inference_engine.get_input_shape()
            logger.debug(f"[INIT] Model input shape: {self.input_shape}")
            self.inference_thread = threading.Thread(
                target=self.inference_engine.run, daemon=True
            )
            self.inference_thread.start()
        except Exception as e:
            if self.multi_process_service:
                logger.critical("=" * 60)
                logger.critical("HAILO DETECTOR DISABLED")
                logger.critical(
                    f"Cannot connect to HailoRT multi-process service: {e}"
                )
                logger.critical(
                    "Ensure hailort_service is running (HA: start the Hailo "
                    "Service add-on; Desktop: run 'sudo hailort_service'), "
                    "or set multi_process_service: false in your detector config."
                )
                logger.critical("=" * 60)
                self.disabled = True
                return
            logger.error(f"[INIT] Failed to initialize HailoAsyncInference: {e}")
            raise

    def set_path_and_url(self, path: str = None):
        if not path:
            self.model_path = None
            self.url = None
            return
        if self.is_url(path):
            self.url = path
            self.model_path = None
        else:
            self.model_path = path
            self.url = None

    def is_url(self, url: str) -> bool:
        return (
            url.startswith("http://")
            or url.startswith("https://")
            or url.startswith("www.")
        )

    def extract_model_name(self, path: str = None, url: str = None) -> str:
        if path and path.endswith(".hef"):
            return os.path.basename(path)
        elif url and url.endswith(".hef"):
            return os.path.basename(url)
        else:
            if self.arch == "hailo8":
                return H8_DEFAULT_MODEL
            else:
                return H8L_DEFAULT_MODEL

    @staticmethod
    def download_model(url: str, destination: str):
        if not url.endswith(".hef"):
            raise ValueError("Invalid model URL. Only .hef files are supported.")
        try:
            urllib.request.urlretrieve(url, destination)
            logger.debug(f"Downloaded model to {destination}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {url}: {str(e)}")

    def check_and_prepare(self) -> str:
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        model_name = self.extract_model_name(self.model_path, self.url)
        cached_model_path = os.path.join(self.cache_dir, model_name)
        if not self.model_path and not self.url:
            if os.path.exists(cached_model_path):
                logger.debug(f"Model found in cache: {cached_model_path}")
                return cached_model_path
            else:
                logger.debug(f"Downloading default model: {model_name}")
                if self.arch == "hailo8":
                    self.download_model(H8_DEFAULT_URL, cached_model_path)
                else:
                    self.download_model(H8L_DEFAULT_URL, cached_model_path)
        elif self.url:
            logger.debug(f"Downloading model from URL: {self.url}")
            self.download_model(self.url, cached_model_path)
        elif self.model_path:
            if os.path.exists(self.model_path):
                logger.debug(f"Using existing model at: {self.model_path}")
                return self.model_path
            else:
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        return cached_model_path

    def detect_raw(self, tensor_input):
        if self.disabled:
            return np.zeros((20, 6), dtype=np.float32)

        tensor_input = self.preprocess(tensor_input)

        if isinstance(tensor_input, np.ndarray) and len(tensor_input.shape) == 3:
            tensor_input = np.expand_dims(tensor_input, axis=0)

        request_id = self.input_store.put(tensor_input)

        try:
            _, infer_results = self.response_store.get(request_id, timeout=1.0)
        except TimeoutError:
            logger.error(
                f"Timeout waiting for inference results for request {request_id}"
            )

            if not self.inference_thread.is_alive():
                raise RuntimeError(
                    "HailoRT inference thread has stopped, restart required."
                )

            return np.zeros((20, 6), dtype=np.float32)

        if isinstance(infer_results, list) and len(infer_results) == 1:
            infer_results = infer_results[0]

        threshold = 0.4
        all_detections = []
        for class_id, detection_set in enumerate(infer_results):
            if not isinstance(detection_set, np.ndarray) or detection_set.size == 0:
                continue
            for det in detection_set:
                if det.shape[0] < 5:
                    continue
                score = float(det[4])
                if score < threshold:
                    continue
                all_detections.append([class_id, score, det[0], det[1], det[2], det[3]])

        if len(all_detections) == 0:
            detections_array = np.zeros((20, 6), dtype=np.float32)
        else:
            all_detections.sort(key=lambda x: x[1], reverse=True)
            detections_array = np.array(all_detections, dtype=np.float32)
            if detections_array.shape[0] > 20:
                detections_array = detections_array[:20, :]
            elif detections_array.shape[0] < 20:
                pad = np.zeros((20 - detections_array.shape[0], 6), dtype=np.float32)
                detections_array = np.vstack((detections_array, pad))

        return detections_array

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = preprocess_tensor(
                image, self.input_shape[1], self.input_shape[0]
            )
            return np.expand_dims(processed, axis=0)
        else:
            raise ValueError("Unsupported image format for preprocessing")

    def close(self):
        logger.debug("[CLOSE] Closing HailoDetector")
        try:
            if hasattr(self, "inference_engine"):
                if hasattr(self.inference_engine, "target"):
                    self.inference_engine.target.release()
                logger.debug("Hailo VDevice released successfully")
        except Exception as e:
            logger.error(f"Failed to close Hailo device: {e}")
            raise

    def __del__(self):
        self.close()


# ----------------- HailoDetectorConfig Class ----------------- #
class HailoDetectorConfig(BaseDetectorConfig):
    """Hailo detector supporting Hailo-8 and Hailo-8L accelerators."""

    model_config = ConfigDict(
        title="Hailo",
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
