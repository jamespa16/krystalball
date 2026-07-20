"""JPEG bytes <-> model tensor conversions for the WebSocket transport.

Replaces main.py's old preprocess()/tensor_to_bgr(), which worked directly on
cv2.VideoCapture frames; now the source/destination is JPEG bytes carried over
a WebSocket instead of a local camera/window.
"""

import cv2
import numpy as np
import torch


def decode_jpeg_to_tensor(jpeg_bytes, width, height, device):
    """Browser-sent JPEG bytes -> (1, 3, height, width) float tensor in [0, 1]."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Failed to decode JPEG frame")
    small = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().div_(255.0)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_jpeg(tensor, quality=80):
    """(1, 3, H, W) float tensor in [0, 1] -> JPEG bytes at internal resolution.

    No upscale here -- the browser does the (nearest-neighbor) upscale for
    display, so only the small internal-resolution frame needs to cross the
    wire.
    """
    frame = tensor.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    frame = (frame * 255.0).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()
