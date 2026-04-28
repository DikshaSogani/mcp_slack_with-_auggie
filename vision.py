import os
import sys
import shutil
import subprocess
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Vision is delegated to the `auggie` CLI (Claude Sonnet 4.5 etc.), which is the
# only working Augment surface on this tenant. The image is written to a temp
# file next to the caller's CWD and auggie is asked to describe it.
#
# Optional .env keys:
#   AUGMENT_VISION_MODEL  — auggie short model id (default: AUGMENT_MODEL or sonnet4.5)
#   AUGGIE_BIN            — override the auggie executable path
#   VISION_TIMEOUT        — per-image timeout in seconds (default: 180)

_VISION_MODEL    = os.getenv("AUGMENT_VISION_MODEL") or os.getenv("AUGMENT_MODEL", "sonnet4.5")
_AUGGIE_BIN      = os.getenv("AUGGIE_BIN", "auggie")
_VISION_TIMEOUT  = int(os.getenv("VISION_TIMEOUT", "180"))

_PROMPT = (
    "Describe the image at {path} in detail. Include any text visible "
    "(OCR), objects, people, technical diagrams, screenshots, UI elements "
    "and their layout. Make the description keyword-rich for search. "
    "Return only the description text — no preamble, no markdown headings."
)


def _detect_mime(image_bytes: bytes) -> str:
    """Sniff the real image mimetype from the first bytes."""
    b = image_bytes[:12]
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if b.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    if b[:2] == b"BM":
        return "image/bmp"
    return "image/jpeg"


_EXT_BY_MIME = {
    "image/png":  ".png",
    "image/jpeg": ".jpg",
    "image/gif":  ".gif",
    "image/webp": ".webp",
    "image/bmp":  ".bmp",
}


def _clean_auggie_output(raw: str) -> str:
    """Strip auggie's model/runtime notice lines."""
    lines = [
        line for line in raw.splitlines()
        if not line.lstrip().startswith(("⚠️", "ℹ️"))
        and "Unknown model" not in line
        and "falling back to default model" not in line
    ]
    return "\n".join(lines).strip()


def get_image_description(image_bytes: bytes, mimetype: str = "") -> str:
    """Describe an image using the auggie CLI and return the description text.

    `mimetype` should be Slack's file.mimetype when available; it is only
    used to pick a sensible file extension for the temp file.
    """
    if not image_bytes:
        return ""

    if shutil.which(_AUGGIE_BIN) is None:
        return "[Image description failed: auggie CLI not found on PATH]"

    mt = mimetype if mimetype.startswith("image/") else _detect_mime(image_bytes)
    ext = _EXT_BY_MIME.get(mt, ".img")

    # Temp file in CWD — auggie's default workspace root is the CWD, and
    # absolute paths inside it are always readable by its `view` tool.
    fd, path = tempfile.mkstemp(prefix="slack_viz_", suffix=ext, dir=os.getcwd())
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)

        prompt = _PROMPT.format(path=path)
        proc = subprocess.run(
            [_AUGGIE_BIN, "--print", "--quiet",
             "--model", _VISION_MODEL,
             prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_VISION_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"[Image description failed: timeout after {_VISION_TIMEOUT}s]"
    except Exception as e:
        return f"[Image description failed: {e}]"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or f"auggie exited with code {proc.returncode}"
        print(f"auggie vision error (mime={mt}, bytes={len(image_bytes)}): {err}", file=sys.stderr)
        return f"[Image description failed: {err}]"

    desc = _clean_auggie_output(proc.stdout or "")
    if not desc:
        return "[Image description failed: empty output]"
    return desc


if __name__ == "__main__":
    print("Testing vision module…")
    pass