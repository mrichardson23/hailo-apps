# Hailo VLM Auto-Describe Application

A simplified interactive vision application using Hailo's Vision Language Model (VLM). Pressing **any key** during live video streaming automatically captures the current frame and displays a text description on screen.

## Features

- **Full-screen real-time video stream**: Live camera feed.
- **One-key description**: Pressing any key immediately captures the image and generates a text description on screen.
- **Non-blocking VLM inference**: Streaming text response with picture-in-picture (PiP) captured frame overlay.
- **Press any key to resume**: Pressing any key on the result screen clears the result and resumes live video streaming.
- **Text-only display**: Pure visual output on screen with no text-to-speech audio.

## Usage

```bash
python -m hailo_apps.python.gen_ai_apps.vlm_auto_describe.vlm_auto_describe
```

### Controls

- **Any Key**: Capture frame and describe image (when streaming) / Continue live stream (when viewing result).
- **Q / Esc**: Quit application.
