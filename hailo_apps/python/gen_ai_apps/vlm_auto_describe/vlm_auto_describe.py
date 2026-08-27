import threading
import signal
import os
import cv2
import sys
import concurrent.futures
import time
import numpy as np
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

from hailo_apps.python.gen_ai_apps.vlm_auto_describe.backend import Backend
from hailo_apps.python.core.common.core import get_standalone_parser, get_logger, handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import (
    VLM_CHAT_APP,
    HAILO10H_ARCH,
    SHARED_VDEVICE_GROUP_ID,
)

# Configuration Constants
MAX_TOKENS = 200
TEMPERATURE = 0.1
SEED = 42
SYSTEM_PROMPT = "You are a helpful assistant that analyzes images and answers questions about them."
AUTO_PROMPT = "Describe this image in one or two sentences."
INFERENCE_TIMEOUT = 60
SAVE_FRAMES = False

# App States
STATE_STREAMING = "STREAMING"
STATE_PROCESSING = "PROCESSING"
STATE_RESULT = "RESULT"

# Strength multiplier for the --blur-fov side-strip blur.
BLUR_STRENGTH = 0.4

# Initialize logger
logger = get_logger(__name__)

class VLMAutoDescribeApp:
    """
    Simplified VLM Application.
    When any key is pressed during live streaming, it automatically captures the frame
    and executes the prompt "Describe this image".
    """
    def __init__(self, hef_path=None, blur_fov: bool = False):
        """
        Initialize the VLM Auto Describe Application.

        Args:
            hef_path: Path to resolved HEF model file.
            blur_fov (bool): Whether to blur the left/right strips outside the central crop.
        """
        self.hef_path = hef_path
        self.running = True
        self.executor = concurrent.futures.ThreadPoolExecutor()
        signal.signal(signal.SIGINT, self.signal_handler)
        self.frozen_frame = None
        self.frozen_inference_frame = None
        self.current_state = STATE_STREAMING
        self.user_question = ''
        self.streamed_response = ''
        self.pip_anim_start: Optional[float] = None
        self.pip_anim_out_start: Optional[float] = None
        self.backend: Optional[Backend] = None
        self.video_thread: Optional[threading.Thread] = None
        self.vlm_future: Optional[concurrent.futures.Future] = None

        self.blur_fov = blur_fov

    @staticmethod
    def _show_splash(text: str) -> None:
        """Render a fullscreen splash frame so the user has feedback during init."""
        frame = np.full((1080, 1920, 3), (85, 35, 205), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.5
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (frame.shape[1] - tw) // 2
        y = (frame.shape[0] + th) // 2
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)

    def signal_handler(self, sig, frame):
        """Handle interrupt signals."""
        print('')
        logger.info("Signal received, shutting down...")
        self.stop()
        sys.exit(0)

    def stop(self):
        """Stop the application and clean up resources."""
        self.running = False
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception as e:
                logger.warning(f"Error closing VLM backend: {e}")
            self.backend = None
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _reset_backend(self) -> None:
        """Tear down the VLM backend and reload it."""
        if self.backend is not None:
            logger.info("Resetting VLM backend process...")
            try:
                self.backend.close()
            except Exception as e:
                logger.warning(f"Error resetting VLM backend: {e}")
            self.backend = None


    def _draw_translucent_box(self, img, x, y, width, height, alpha=0.55, tint=None):
        """Draw a rounded translucent rectangle onto img."""
        h, w = img.shape[:2]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + width)), min(h, int(y + height))
        if x2 <= x1 or y2 <= y1:
            return
        sub = img[y1:y2, x1:x2]
        if tint is None:
            black = np.zeros_like(sub)
            cv2.addWeighted(sub, 1.0 - alpha, black, alpha, 0, dst=sub)
        else:
            colored = np.full_like(sub, tint, dtype=np.uint8)
            cv2.addWeighted(sub, 1.0 - alpha, colored, alpha, 0, dst=sub)

    def _wrap_text(self, text, max_w, font, scale, thickness):
        """Wrap text to fit within max_w pixels."""
        if not text:
            return []
        lines = []
        for paragraph in text.split('\n'):
            words = paragraph.split(' ')
            cur = ""
            for word in words:
                cand = (cur + " " + word).strip() if cur else word
                w_pixels = cv2.getTextSize(cand, font, scale, thickness)[0][0]
                if w_pixels <= max_w:
                    cur = cand
                else:
                    if cur:
                        lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
        return lines

    def _apply_blur_fov(self, frame):
        """Apply side-strip blur outside central square crop."""
        h, w = frame.shape[:2]
        if w <= h:
            return frame
        crop_x0 = (w - h) // 2
        crop_x1 = crop_x0 + h

        out = frame.copy()
        ksize = int(w * 0.05 * BLUR_STRENGTH) | 1
        ksize = max(3, ksize)
        blurred = cv2.GaussianBlur(out, (ksize, ksize), 0)

        out[:, :crop_x0] = blurred[:, :crop_x0]
        out[:, crop_x1:] = blurred[:, crop_x1:]

        guide_color = (80, 80, 80)
        cv2.line(out, (crop_x0, 0), (crop_x0, h), guide_color, 1)
        cv2.line(out, (crop_x1 - 1, 0), (crop_x1 - 1, h), guide_color, 1)
        return out

    def _draw_overlay(self, frame):
        """Compose state-specific overlay on frame."""
        out = frame.copy()
        h, w = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, w / 1400.0)
        thickness = 1
        text_color = (255, 255, 255)
        margin = max(8, w // 80)
        max_text_w = w - 2 * margin
        line_h = int(28 * scale) + 6

        # Top banner
        if self.current_state == STATE_STREAMING:
            header = "Press any key to describe the image using a local vision-language model (QWEN-VL-2B-Instruct), on the Raspberry Pi AI HAT+ 2. No connectivity needed."
        elif self.current_state == STATE_PROCESSING:
            header = "Processing..."
        elif self.current_state == STATE_RESULT:
            header = "Press any key to continue."
        else:
            header = ""

        header_lines = self._wrap_text(header, max_text_w, font, scale, thickness) if header else []
        if header_lines:
            banner_h = len(header_lines) * line_h + 2 * margin
            self._draw_translucent_box(out, 0, 0, w, banner_h, alpha=0.55)
            for i, line in enumerate(header_lines):
                y = margin + (i + 1) * line_h - 6
                cv2.putText(out, line, (margin, y),
                            font, scale, text_color, thickness, cv2.LINE_AA)
        else:
            banner_h = 0

        # PiP animation
        anim_duration = 0.35
        s = None
        is_exiting = False
        if self.pip_anim_out_start is not None and self.frozen_frame is not None:
            elapsed = time.time() - self.pip_anim_out_start
            t = max(0.0, min(1.0, 1.0 - elapsed / anim_duration))
            s = t * t * t
            is_exiting = True
        elif self.current_state in (STATE_PROCESSING, STATE_RESULT) and self.frozen_frame is not None:
            elapsed = (time.time() - self.pip_anim_start) if self.pip_anim_start is not None else anim_duration
            t = max(0.0, min(1.0, elapsed / anim_duration))
            s = 1.0 - (1.0 - t) ** 3

        if s is not None:
            final_w = max(160, w // 4)
            ih, iw = self.frozen_frame.shape[:2]
            final_h = max(1, int(round(final_w * ih / iw)))
            final_x = w - final_w - margin
            final_y = banner_h + margin

            cur_w = max(1, int(round(w + (final_w - w) * s)))
            cur_h = max(1, int(round(h + (final_h - h) * s)))
            cur_x = int(round(final_x * s))
            cur_y = int(round(final_y * s))

            alpha = s if is_exiting else 1.0
            inset = cv2.resize(self.frozen_frame, (cur_w, cur_h), interpolation=cv2.INTER_AREA)
            x_clip0, x_clip1 = max(0, cur_x), min(w, cur_x + cur_w)
            y_clip0, y_clip1 = max(0, cur_y), min(h, cur_y + cur_h)
            if x_clip1 > x_clip0 and y_clip1 > y_clip0:
                src_x0 = x_clip0 - cur_x
                src_x1 = src_x0 + (x_clip1 - x_clip0)
                src_y0 = y_clip0 - cur_y
                src_y1 = src_y0 + (y_clip1 - y_clip0)
                inset_clip = inset[src_y0:src_y1, src_x0:src_x1]
                roi = out[y_clip0:y_clip1, x_clip0:x_clip1]
                if alpha >= 1.0:
                    roi[...] = inset_clip
                else:
                    cv2.addWeighted(inset_clip, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
                if s > 0.6:
                    border_intensity = int(255 * alpha)
                    cv2.rectangle(out, (cur_x - 1, cur_y - 1), (cur_x + cur_w, cur_y + cur_h),
                                  (border_intensity, border_intensity, border_intensity), 1)

        # Bottom panel
        max_text_w = w - 2 * margin
        panel_lines = []
        if self.current_state in (STATE_PROCESSING, STATE_RESULT):
            response_text = self.streamed_response.strip()
            panel_lines = self._wrap_text(response_text, max_text_w, font, scale, thickness) if response_text else []

        if panel_lines:
            max_panel_lines = max(1, int(h * 0.45) // line_h)
            if len(panel_lines) > max_panel_lines:
                panel_lines = panel_lines[-max_panel_lines:]
            panel_h = len(panel_lines) * line_h + 2 * margin
            panel_y = h - panel_h
            self._draw_translucent_box(out, 0, panel_y, w, panel_h, alpha=0.55)
            for i, line in enumerate(panel_lines):
                y = panel_y + margin + (i + 1) * line_h - 6
                cv2.putText(out, line, (margin, y), font, scale, text_color, thickness, cv2.LINE_AA)

        return out

    def _dismiss_pip(self) -> None:
        """Start PiP slide-out animation."""
        if self.frozen_frame is not None and self.pip_anim_out_start is None:
            self.pip_anim_out_start = time.time()

    def _clear_pip(self) -> None:
        """Hard reset PiP state."""
        self.frozen_frame = None
        self.frozen_inference_frame = None
        self.user_question = ''
        self.streamed_response = ''
        self.pip_anim_start = None
        self.pip_anim_out_start = None

    def _init_camera(self) -> tuple[Callable[[], Any], Callable[[], None], str]:
        """Initialize camera feed."""
        try:
            from picamera2 import Picamera2
            from libcamera import controls, Transform
            picam2 = Picamera2()
            raw_size = tuple([v // 2 for v in picam2.camera_properties['PixelArraySize']])
            config = picam2.create_preview_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                lores={"size": (336, 336), "format": "RGB888", "preserve_ar": True},
                raw={"size": raw_size},
                transform=Transform(hflip=1),
            )
            picam2.configure(config)
            picam2.start()
            if 'AfMode' in picam2.camera_controls:
                try:
                    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                except Exception as e:
                    logger.debug(f"Failed to set continuous autofocus: {e}")

            def get_frame():
                arrays = picam2.capture_arrays(["main", "lores"])[0]
                return arrays[0], arrays[1]

            cleanup = lambda: picam2.stop()
            return get_frame, cleanup, "Raspberry Pi Camera"

        except Exception as e:
            logger.info(f"Picamera2 not available ({e}), falling back to OpenCV VideoCapture...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open camera via OpenCV or picamera2")

            def get_frame():
                ret, frame = cap.read()
                if not ret:
                    return None, None
                return frame, None

            def cleanup():
                cap.release()

            return get_frame, cleanup, "USB / OpenCV Camera"

    def show_video(self):
        """Main application loop."""
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._show_splash("Initialising...")

        try:
            get_frame, cleanup, cam_name = self._init_camera()
            logger.info(f"Using camera: {cam_name}")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.running = False
            return

        self._show_splash("Loading VLM...")

        hef_path = self.hef_path or resolve_hef_path(None, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
        if hef_path is None:
            logger.error("Failed to resolve HEF path for VLM model.")
            cleanup()
            self.running = False
            return

        self.backend = Backend(
            hef_path=str(hef_path),
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            seed=SEED,
            system_prompt=SYSTEM_PROMPT
        )

        try:
            while self.running:
                viewfinder_frame, inference_frame = get_frame()
                if viewfinder_frame is None and self.current_state == STATE_STREAMING:
                    logger.warning("Failed to grab frame from camera.")
                    time.sleep(0.1)
                    continue

                if self.blur_fov and viewfinder_frame is not None:
                    viewfinder_frame = self._apply_blur_fov(viewfinder_frame)

                if inference_frame is None and viewfinder_frame is not None:
                    inference_frame = viewfinder_frame

                base_frame = self.frozen_frame if self.current_state in (STATE_PROCESSING, STATE_RESULT) else viewfinder_frame

                if self.current_state == STATE_PROCESSING and self.backend is not None:
                    new_tokens = self.backend.poll_stream()
                    if new_tokens:
                        self.streamed_response += new_tokens

                if base_frame is not None:
                    cv2.imshow('Video', self._draw_overlay(base_frame))

                key = cv2.waitKey(25)
                key = -1 if key == -1 else key & 0xFFFF
                self._handle_key(key, viewfinder_frame, inference_frame)

                if (self.pip_anim_out_start is not None and time.time() - self.pip_anim_out_start >= 0.35):
                    self._clear_pip()

                if not self.running:
                    break

                if self.current_state == STATE_PROCESSING and self.vlm_future is not None and self.vlm_future.done():
                    if self.backend is not None:
                        tail_tokens = self.backend.poll_stream()
                        if tail_tokens:
                            self.streamed_response += tail_tokens
                    try:
                        result = self.vlm_future.result()
                        answer = result.get('answer') if isinstance(result, dict) else None
                        if answer:
                            self.streamed_response = answer
                    except Exception as e:
                        logger.error(f"Error getting future result: {e}")
                        self.streamed_response = f"Error processing request: {e}"
                    self.vlm_future = None
                    self.current_state = STATE_RESULT

                if (self.current_state == STATE_PROCESSING
                        and self.vlm_future is None
                        and self.frozen_inference_frame is not None
                        and self.user_question
                        and self.backend is not None):
                    self.vlm_future = self.executor.submit(
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
        """Dispatch key press according to state."""
        if key == -1:
            return

        ESC = 27

        if self.current_state == STATE_STREAMING:
            if key == ESC or key in (ord('q'), ord('Q')):
                self.stop()
                return

            # Any key press captures frame and triggers "Describe this image"
            if viewfinder_frame is not None and inference_frame is not None:
                self.frozen_frame = viewfinder_frame.copy()
                self.frozen_inference_frame = inference_frame.copy()
                self.user_question = AUTO_PROMPT
                self.streamed_response = ''
                self.pip_anim_start = time.time()
                self.pip_anim_out_start = None
                if self.backend is not None:
                    self.backend.clear_stream()
                self.current_state = STATE_PROCESSING
            return

        if self.current_state == STATE_PROCESSING:
            # Inputs ignored while processing
            return

        if self.current_state == STATE_RESULT:
            if key == ESC or key in (ord('q'), ord('Q')):
                self.stop()
                return
            # Any key press returns to live streaming
            self._dismiss_pip()
            self.current_state = STATE_STREAMING
            return

    def run(self):
        """Start the application thread."""
        self.video_thread = threading.Thread(target=self.show_video)
        self.video_thread.start()
        try:
            while self.video_thread.is_alive():
                self.video_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
            self.stop()
            if self.video_thread.is_alive():
                self.video_thread.join(timeout=1.0)

if __name__ == "__main__":
    parser = get_standalone_parser()
    parser.add_argument('--no-blur-fov', action='store_false', dest='blur_fov',
                        help='Do not blur the left/right strips outside the central square crop.')

    handle_list_models_flag(parser, VLM_CHAT_APP)

    options_menu = parser.parse_args()

    hef_path = resolve_hef_path(
        options_menu.hef_path if hasattr(options_menu, 'hef_path') else None,
        app_name=VLM_CHAT_APP,
        arch=HAILO10H_ARCH
    )
    if hef_path is None:
        logger.error("Failed to resolve HEF path for VLM model. Exiting.")
        sys.exit(1)
    app = VLMAutoDescribeApp(
        hef_path=hef_path,
        blur_fov=options_menu.blur_fov,
    )
    app.run()
    sys.exit(0)
