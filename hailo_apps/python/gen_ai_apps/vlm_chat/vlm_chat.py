import threading
import signal
import os
import cv2
import sys
import concurrent.futures
import time
from typing import Optional, Callable, Any
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = 'xcb'

repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break
if repo_root is not None:
    sys.path.insert(0, str(repo_root))

from hailo_apps.python.gen_ai_apps.vlm_chat.backend import Backend
from hailo_apps.python.core.common.core import get_standalone_parser, get_resource_path, get_logger, handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.camera_utils import get_usb_video_devices
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import get_source_type
from hailo_apps.python.core.common.defines import (
    VLM_CHAT_APP,
    VLM_MODEL_NAME_H10,
    RESOURCES_MODELS_DIR_NAME,
    HAILO10H_ARCH,
    RPI_NAME_I,
    USB_CAMERA
)

# Configuration Constants
MAX_TOKENS = 200
TEMPERATURE = 0.1
SEED = 42
SYSTEM_PROMPT = "You are a helpful assistant that analyzes images and answers questions about them."
INFERENCE_TIMEOUT = 60
SAVE_FRAMES = False

# App States
STATE_STREAMING = "STREAMING"
STATE_CAPTURED = "CAPTURED"
STATE_PROCESSING = "PROCESSING"
STATE_RESULT = "RESULT"

# Initialize logger
logger = get_logger(__name__)

class VLMChatApp:
    """
    Main application class for VLM Chat.
    Handles video display, user input, and interaction with the VLM backend.
    """
    def __init__(self, camera: Any, camera_type: str):
        """
        Initialize the VLM Chat Application.

        Args:
            camera (Any): Camera source (device index or connection object).
            camera_type (str): Type of camera ('usb' or 'rpi').
        """
        self.camera = camera
        self.camera_type = camera_type
        self.running = True
        self.executor = concurrent.futures.ThreadPoolExecutor()
        signal.signal(signal.SIGINT, self.signal_handler)
        self.frozen_frame = None
        self.frozen_inference_frame = None
        self.current_state = STATE_STREAMING
        self.user_question = ''
        self.streamed_response = ''
        self.backend: Optional[Backend] = None
        self.video_thread: Optional[threading.Thread] = None

    def signal_handler(self, sig, frame):
        """Handle interrupt signals."""
        print('')
        logger.info("Signal received, shutting down...")
        self.stop()

    def stop(self):
        """Stop the application and clean up resources."""
        self.running = False
        if self.backend:
            self.backend.close()
        self.executor.shutdown(wait=True)

    @staticmethod
    def _draw_translucent_box(frame, x: int, y: int, w: int, h: int, alpha: float = 0.55) -> None:
        """Darken a region of `frame` in-place, simulating a translucent black box."""
        h_f, w_f = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_f, x + w), min(h_f, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        cv2.addWeighted(roi, 1.0 - alpha, roi, 0.0, 0.0, dst=roi)

    @staticmethod
    def _wrap_text(text: str, max_width_px: int, font: int, scale: float, thickness: int) -> list:
        """Greedy word-wrap; falls back to per-character break for over-long tokens."""
        if not text:
            return ['']
        lines = []
        for paragraph in text.split('\n'):
            current = ''
            for word in paragraph.split(' '):
                candidate = word if not current else current + ' ' + word
                (cw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
                if cw <= max_width_px:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                    current = ''
                # Word alone doesn't fit — break by character.
                buf = ''
                for ch in word:
                    nxt = buf + ch
                    (bw, _), _ = cv2.getTextSize(nxt, font, scale, thickness)
                    if bw <= max_width_px:
                        buf = nxt
                    else:
                        if buf:
                            lines.append(buf)
                        buf = ch
                current = buf
            lines.append(current)
        return lines

    def _draw_overlay(self, frame):
        """Compose the state-specific overlay on a copy of `frame` and return it."""
        out = frame.copy()
        h, w = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, w / 1400.0)  # ≈0.7 at 960 wide; scales sensibly elsewhere.
        thickness = 1
        text_color = (255, 255, 255)
        margin = max(8, w // 80)
        line_h = int(28 * scale) + 6

        # Top banner: state header.
        if self.current_state == STATE_STREAMING:
            header = "LIVE  |  Enter: capture   Q/Esc: quit"
        elif self.current_state == STATE_CAPTURED:
            header = "Type question  |  Enter: send   Esc: cancel"
        elif self.current_state == STATE_PROCESSING:
            header = "Processing..."
        elif self.current_state == STATE_RESULT:
            header = "Enter: continue   Q/Esc: quit"
        else:
            header = ""

        banner_h = line_h + 2 * margin
        self._draw_translucent_box(out, 0, 0, w, banner_h, alpha=0.55)
        cv2.putText(out, header, (margin, margin + line_h - 6),
                    font, scale, text_color, thickness, cv2.LINE_AA)

        # Bottom panel: question / response.
        panel_lines = []
        max_text_w = w - 2 * margin
        if self.current_state == STATE_CAPTURED:
            panel_lines = self._wrap_text("Q: " + self.user_question + "_",
                                          max_text_w, font, scale, thickness)
        elif self.current_state in (STATE_PROCESSING, STATE_RESULT):
            q_lines = self._wrap_text("Q: " + self.user_question,
                                      max_text_w, font, scale, thickness)
            response_text = self.streamed_response.strip()
            r_lines = self._wrap_text(response_text, max_text_w, font, scale, thickness) if response_text else []
            panel_lines = q_lines + ([''] + r_lines if r_lines else [])

        if panel_lines:
            max_panel_lines = max(1, int(h * 0.45) // line_h)
            if len(panel_lines) > max_panel_lines:
                panel_lines = panel_lines[-max_panel_lines:]
            panel_h = len(panel_lines) * line_h + 2 * margin
            panel_y = h - panel_h
            self._draw_translucent_box(out, 0, panel_y, w, panel_h, alpha=0.55)
            for i, line in enumerate(panel_lines):
                y = panel_y + margin + (i + 1) * line_h - 6
                cv2.putText(out, line, (margin, y),
                            font, scale, text_color, thickness, cv2.LINE_AA)

        return out

    def _reset_session(self) -> None:
        """Clear all per-question state."""
        self.frozen_frame = None
        self.frozen_inference_frame = None
        self.user_question = ''
        self.streamed_response = ''

    def _init_camera(self) -> tuple[Callable[[], Any], Callable[[], None], str]:
        """
        Initialize the camera based on type.

        Returns:
            tuple: (get_frame_callback, cleanup_callback, camera_name)
        """
        if self.camera_type == RPI_NAME_I:
            try:
                from picamera2 import Picamera2
                from libcamera import controls
                picam2 = Picamera2()
                # Dual stream: large 'main' for viewfinder, smaller 'lores' for inference
                config = picam2.create_preview_configuration(
                    main={"size": (1920, 1080), "format": "RGB888"},
                    lores={"size": (448, 448), "format": "RGB888"},
                    raw={"size": (2304, 1296)},
                )
                picam2.configure(config)
                picam2.start()
                # Enable continuous autofocus (no-op on fixed-focus sensors).
                try:
                    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                except Exception as e:
                    logger.debug(f"Continuous autofocus not available: {e}")

                def get_frame():
                    arrays = picam2.capture_arrays(["main", "lores"])[0]
                    return arrays[0], arrays[1]

                cleanup = lambda: picam2.stop()
                camera_name = "RPI"
                return get_frame, cleanup, camera_name
            except (ImportError, Exception) as e:
                logger.error(f"Error initializing RPI camera: {e}")
                raise
        else:
            cap = cv2.VideoCapture(self.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            def get_frame():
                ret, fr = cap.read()
                if not ret:
                    return None, None
                # USB has a single stream; use it for both viewfinder and inference
                return fr, fr

            cleanup = lambda: cap.release()
            camera_name = "USB"
            return get_frame, cleanup, camera_name

    def show_video(self):
        """Main loop: render video + overlay and handle keystrokes from the cv2 window."""
        try:
            get_frame, cleanup, _ = self._init_camera()
        except Exception:
            logger.error("Failed to initialize camera. Exiting.")
            self.running = False
            return

        # Full-screen window without the Qt toolbar/status bar.
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Initialize Backend
        try:
            self.backend = Backend(
                hef_path=str(hef_path),
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                seed=SEED,
                system_prompt=SYSTEM_PROMPT
            )
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            cleanup()
            self.running = False
            return

        vlm_future = None
        viewfinder_frame = None
        inference_frame = None

        try:
            while self.running:
                # Acquire / pick the base frame for this iteration.
                if self.current_state == STATE_STREAMING:
                    raw_view, raw_inf = get_frame()
                    if raw_view is None:
                        logger.error("Failed to read frame from camera")
                        break
                    # picamera2 'RGB888' yields BGR-ordered bytes in numpy (matching cv2). USB is BGR. No conversion needed.
                    viewfinder_frame = raw_view
                    inference_frame = raw_inf
                    base_frame = viewfinder_frame
                else:
                    base_frame = self.frozen_frame

                # Drain any pending streamed tokens before drawing.
                if self.current_state == STATE_PROCESSING and self.backend is not None:
                    self.streamed_response += self.backend.poll_stream()

                if base_frame is not None:
                    cv2.imshow('Video', self._draw_overlay(base_frame))

                # Single source of input: the cv2 window.
                key = cv2.waitKey(25)
                key = -1 if key == -1 else key & 0xFFFF
                self._handle_key(key, viewfinder_frame, inference_frame)

                if not self.running:
                    break

                # Inference completion check.
                if self.current_state == STATE_PROCESSING and vlm_future is not None and vlm_future.done():
                    if self.backend is not None:
                        self.streamed_response += self.backend.poll_stream()
                    try:
                        result = vlm_future.result()
                        # Prefer the canonical answer (handles error/timeout strings too).
                        answer = result.get('answer') if isinstance(result, dict) else None
                        if answer:
                            self.streamed_response = answer
                    except Exception as e:
                        logger.error(f"Error getting future result: {e}")
                        self.streamed_response = f"Error processing request: {e}"
                    vlm_future = None
                    self.current_state = STATE_RESULT

                # Schedule the inference future once we've fully entered PROCESSING.
                if (self.current_state == STATE_PROCESSING
                        and vlm_future is None
                        and self.frozen_inference_frame is not None
                        and self.user_question
                        and self.backend is not None):
                    vlm_future = self.executor.submit(
                        self.backend.vlm_inference,
                        self.frozen_inference_frame.copy(),
                        self.user_question,
                        INFERENCE_TIMEOUT
                    )

        finally:
            cleanup()
            cv2.destroyAllWindows()
            self.stop()

    def _handle_key(self, key: int, viewfinder_frame, inference_frame) -> None:
        """Dispatch a key press according to the current state."""
        if key == -1:
            return

        ENTER = (10, 13)
        BACKSPACE = (8, 127)
        ESC = 27

        if self.current_state == STATE_STREAMING:
            if key == ESC or key in (ord('q'), ord('Q')):
                self.stop()
                return
            if key in ENTER:
                if viewfinder_frame is not None and inference_frame is not None:
                    self.frozen_frame = viewfinder_frame.copy()
                    self.frozen_inference_frame = inference_frame.copy()
                    self.user_question = ''
                    self.streamed_response = ''
                    self.current_state = STATE_CAPTURED
            return

        if self.current_state == STATE_CAPTURED:
            if key == ESC:
                self._reset_session()
                self.current_state = STATE_STREAMING
                return
            if key in ENTER:
                if not self.user_question:
                    self.user_question = "Describe the image"
                if SAVE_FRAMES and self.frozen_inference_frame is not None:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f"frame_{timestamp}.jpg", self.frozen_inference_frame)
                if self.backend is not None:
                    self.backend.clear_stream()
                self.streamed_response = ''
                self.current_state = STATE_PROCESSING
                return
            if key in BACKSPACE:
                self.user_question = self.user_question[:-1]
                return
            if 32 <= key <= 126:
                self.user_question += chr(key)
            return

        if self.current_state == STATE_PROCESSING:
            return  # Inputs ignored while inference is running.

        if self.current_state == STATE_RESULT:
            if key == ESC or key in (ord('q'), ord('Q')):
                self.stop()
                return
            if key in ENTER:
                self._reset_session()
                self.current_state = STATE_STREAMING

    def run(self):
        """Start the application thread."""
        self.video_thread = threading.Thread(target=self.show_video)
        self.video_thread.start()
        try:
            self.video_thread.join()
        except KeyboardInterrupt:
            self.stop()
            self.video_thread.join()

if __name__ == "__main__":
    parser = get_standalone_parser()

    # Handle --list-models flag before full initialization
    handle_list_models_flag(parser, VLM_CHAT_APP)

    options_menu = parser.parse_args()

    # Resolve HEF path with auto-download (VLM is Hailo-10H only)
    hef_path = resolve_hef_path(
        options_menu.hef_path if hasattr(options_menu, 'hef_path') else None,
        app_name=VLM_CHAT_APP,
        arch=HAILO10H_ARCH
    )
    if hef_path is None:
        logger.error("Failed to resolve HEF path for VLM model. Exiting.")
        sys.exit(1)
    video_source = options_menu.input
    if video_source == USB_CAMERA:
        logger.debug("USB_CAMERA detected; scanning USB devices...")
        video_source = get_usb_video_devices()
        if not video_source:
            logger.error("No USB camera found for '--input usb'")
            print(
                'Provided argument "--input" is set to "usb", however no available USB cameras found. Please connect a camera or specifiy different input method.'
            )
            sys.exit(1)
        else:
            logger.debug(f"Using USB camera: {video_source[0]}")
            video_source = video_source[0]

    # Determine source type (usb, rpi, file, etc.)
    source_type = get_source_type(video_source) if video_source is not None else None

    if video_source is None:
        print('Please provide an input source using the "--input" argument: "usb" for USB camera or "rpi" for Raspberry Pi camera.')
        sys.exit(1)

    app = VLMChatApp(camera=video_source, camera_type=source_type)
    app.run()
    sys.exit(0)
