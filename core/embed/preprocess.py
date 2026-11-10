"""Upload validation (FR-010) and per-modality preprocessing (FR-011).

Everything is driven by `EmbedParams`; each knob changes the produced artifact (unit-tested).
Media decode goes through ffmpeg (imageio-ffmpeg's static binary) with argv lists, -nostdin and
timeouts; decoded buffers are capped by MEDIA_BUFFER_MB.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from core.errors import ValidationError
from core.schemas import AudioParams, EmbedParams, ImageParams, TextParams, VideoParams

MEDIA_BUFFER_MB = 100
FFMPEG_TIMEOUT_S = 120
MAX_FILE_MB = 50
ALLOWED_EXT = {".txt", ".md", ".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3", ".mp4", ".webm"}
IMAGE_TOKENS_DEFAULT = 280  # EmbeddingGemma-2 card default per image
AUDIO_TOKENS_PER_S = 25
FRAME_TOKENS = 140

# --------------------------------------------------------------------------- sniffing


@dataclass(frozen=True)
class Sniffed:
    modality: str  # text | image | audio | video
    mime: str
    ext: str


def sniff(data: bytes) -> Sniffed | None:
    """Detect type from magic bytes (never from the file extension)."""
    head = data[:64]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Sniffed("image", "image/png", ".png")
    if head.startswith(b"\xff\xd8\xff"):
        return Sniffed("image", "image/jpeg", ".jpg")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Sniffed("image", "image/webp", ".webp")
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return Sniffed("audio", "audio/wav", ".wav")
    if head.startswith(b"ID3") or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0 and (head[1] & 0x06) != 0):
        return Sniffed("audio", "audio/mpeg", ".mp3")
    if head[4:8] == b"ftyp":
        return Sniffed("video", "video/mp4", ".mp4")
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return Sniffed("video", "video/webm", ".webm")
    if b"\x00" in data[:4096]:
        return None
    try:
        data[: 1 << 20].decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.start < len(data[: 1 << 20]) - 4:  # tolerate a multibyte char cut at the boundary
            return None
    return Sniffed("text", "text/plain", ".txt")


def validate_upload(name: str, data: bytes, *, used_bytes: int = 0, session_quota_mb: int = 200, max_pixels: int = 40_000_000) -> Sniffed:
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ValidationError(f"'{name}': file type {ext or '(none)'} is not supported. Allowed: {', '.join(sorted(ALLOWED_EXT))}.")
    if len(data) == 0:
        raise ValidationError(f"'{name}' is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ValidationError(f"'{name}' is larger than {MAX_FILE_MB} MB.")
    if used_bytes + len(data) > session_quota_mb * 1024 * 1024:
        raise ValidationError(f"'{name}' would exceed this session's {session_quota_mb} MB upload limit.")
    kind = sniff(data)
    if kind is None:
        raise ValidationError(f"'{name}' does not look like a supported file (content check failed).")
    ext_modality = {
        ".txt": "text",
        ".md": "text",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".webp": "image",
        ".wav": "audio",
        ".mp3": "audio",
        ".mp4": "video",
        ".webm": "video",
    }[ext]
    if ext_modality != kind.modality:
        raise ValidationError(f"'{name}' says {ext} but its content is {kind.mime}; rejected.")
    if kind.modality == "image":
        check_image_pixels(data, max_pixels, name)
    return kind


def check_image_pixels(data: bytes, max_pixels: int, name: str = "image") -> tuple[int, int]:
    """Read only the header: reject decompression bombs BEFORE decoding pixels."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Image.DecompressionBombError as exc:
        raise ValidationError(f"'{name}' has too many pixels.") from exc
    except Exception as exc:
        raise ValidationError(f"'{name}' could not be read as an image.") from exc
    if w * h > max_pixels:
        raise ValidationError(f"'{name}' is {w}x{h} ({w * h / 1e6:.0f} MP), above the {max_pixels / 1e6:.0f} MP limit.")
    return w, h


def sha256_8(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


# --------------------------------------------------------------------------- outputs


@dataclass
class Piece:
    """One embedding request element + its sidecar metadata."""

    modality: str
    content: Any  # str (text) or list of typed parts (media)
    span: dict[str, Any]
    text: str | None = None
    tokens_est: int = 0
    preview: bytes | None = None  # thumbnail jpeg / audio wav clip for source cards
    preview_ext: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- text

_WS = re.compile(r"[ \t]+")
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Heuristic token count with a safety margin (~1.25x word/punct pieces)."""
    return int(len(_TOKEN.findall(text)) * 1.25) + 1


def chunk_text(text: str, p: TextParams) -> list[tuple[str, tuple[int, int]]]:
    """Paragraph-aware packing into ~chunk_tokens windows with overlap_tokens overlap.
    Returns (chunk_text, (char_start, char_end))."""
    text = text.replace("\r\n", "\n")
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if not words:
        return []
    budget = max(8, int(p.chunk_tokens / 1.25))
    overlap = min(int(p.overlap_tokens / 1.25), budget // 2)
    chunks: list[tuple[str, tuple[int, int]]] = []
    i = 0
    while i < len(words):
        j = min(len(words), i + budget)
        # prefer to end at a paragraph boundary in the last third of the window
        if j < len(words):
            for k in range(j, i + (2 * budget) // 3, -1):
                if "\n\n" in text[words[k - 1][2] : words[k][1]]:
                    j = k
                    break
        a, b = words[i][1], words[j - 1][2]
        chunk = _WS.sub(" ", text[a:b]).strip()
        chunks.append((chunk, (a, b)))
        if j >= len(words):
            break
        i = max(i + 1, j - overlap)
    return chunks


def preprocess_text(data: bytes, p: TextParams, prefix_doc: str) -> list[Piece]:
    text = data.decode("utf-8", errors="replace")
    out = []
    for idx, (chunk, (a, b)) in enumerate(chunk_text(text, p)):
        content = prefix_doc.format(content=chunk)
        out.append(Piece("text", content, {"chars": [a, b], "chunk": idx}, text=chunk, tokens_est=estimate_tokens(content)))
    return out


# --------------------------------------------------------------------------- image


def image_to_jpeg(data: bytes, p: ImageParams) -> tuple[bytes, tuple[int, int]]:
    check_image_pixels(data, p.max_pixels)
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.thumbnail((p.max_side_px, p.max_side_px), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        if p.format == "png":
            im.save(buf, format="PNG")
        else:
            im.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), im.size


def thumbnail(data: bytes, side: int = 256) -> bytes:
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return buf.getvalue()


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def image_part(jpeg: bytes, fmt: str = "jpeg") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url(jpeg, "image/png" if fmt == "png" else "image/jpeg")}}


def preprocess_image(data: bytes, p: ImageParams) -> list[Piece]:
    img, size = image_to_jpeg(data, p)
    tokens = p.token_cap_max or IMAGE_TOKENS_DEFAULT
    return [
        Piece(
            "image",
            [image_part(img, p.format)],
            {"bbox": [0, 0, size[0], size[1]]},
            tokens_est=tokens,
            preview=thumbnail(img, 512),
            preview_ext=".jpg",
            meta={"width": size[0], "height": size[1]},
        )
    ]


# --------------------------------------------------------------------------- ffmpeg helpers


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def _run_ffmpeg(args: list[str], data: bytes | None = None, max_out_mb: int = MEDIA_BUFFER_MB) -> bytes:
    argv = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", *args]
    try:
        res = subprocess.run(argv, input=data, capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValidationError("Media decoding took too long and was stopped.") from exc
    if res.returncode != 0:
        raise ValidationError("The media file could not be decoded.", detail=res.stderr.decode(errors="replace")[-500:])
    if len(res.stdout) > max_out_mb * 1024 * 1024:
        raise ValidationError(f"Decoded media exceeds the {max_out_mb} MB buffer limit.")
    return res.stdout


def _with_input(path: Path) -> list[str]:
    return ["-i", str(path)]


def decode_audio_pcm(path: Path, p: AudioParams) -> bytes:
    """Mono s16le PCM at sample_rate_hz, capped at max_total_s."""
    max_bytes_mb = int(p.max_total_s * p.sample_rate_hz * 2 / 1e6) + 1
    return _run_ffmpeg(
        [*_with_input(path), "-t", str(p.max_total_s), "-vn", "-ac", "1", "-ar", str(p.sample_rate_hz), "-f", "s16le", "-"],
        max_out_mb=min(MEDIA_BUFFER_MB, max(1, max_bytes_mb)),
    )


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def audio_windows(n_samples: int, rate: int, p: AudioParams) -> list[tuple[int, int]]:
    win = int(p.window_s * rate)
    hop = max(1, int((p.window_s - p.overlap_s) * rate))
    if n_samples <= win:
        return [(0, n_samples)]
    out: list[tuple[int, int]] = []
    start = 0
    while start < n_samples:
        end = min(n_samples, start + win)
        if end - start < rate * 0.5 and out:  # drop < 0.5 s tail, extend previous instead
            out[-1] = (out[-1][0], end)
            break
        out.append((start, end))
        if end >= n_samples:
            break
        start += hop
    return out


def audio_part(wav: bytes) -> dict[str, Any]:
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}


def preprocess_audio(path: Path, p: AudioParams) -> list[Piece]:
    pcm = decode_audio_pcm(path, p)
    rate = p.sample_rate_hz
    n = len(pcm) // 2
    if n == 0:
        raise ValidationError("The audio file contains no samples.")
    out = []
    for idx, (a, b) in enumerate(audio_windows(n, rate, p)):
        wav = pcm_to_wav(pcm[a * 2 : b * 2], rate)
        secs = (b - a) / rate
        out.append(
            Piece(
                "audio",
                [audio_part(wav)],
                {"t": [round(a / rate, 2), round(b / rate, 2)], "chunk": idx},
                tokens_est=int(secs * AUDIO_TOKENS_PER_S) + 8,
                preview=wav,
                preview_ext=".wav",
                meta={"duration_s": round(n / rate, 2)},
            )
        )
    return out


def media_duration_s(path: Path) -> float:
    """Duration via ffmpeg's stderr banner (no ffprobe in imageio-ffmpeg)."""
    argv = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(path)]
    res = subprocess.run(argv, capture_output=True, timeout=30, check=False)
    m = re.search(rb"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr)
    if not m:
        raise ValidationError("Could not read the media duration.")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def extract_frames(path: Path, p: VideoParams) -> list[tuple[float, bytes]]:
    duration = min(media_duration_s(path), p.max_total_s)
    fps = p.fps
    if duration * fps > p.max_frames:
        fps = p.max_frames / max(duration, 1e-3)
    side = p.frame_max_side_px
    vf = f"fps={fps:.4f},scale='if(gt(iw,ih),min({side},iw),-2)':'if(gt(iw,ih),-2,min({side},ih))'"
    raw = _run_ffmpeg(
        [
            *_with_input(path),
            "-t",
            f"{duration:.3f}",
            "-an",
            "-vf",
            vf,
            "-frames:v",
            str(p.max_frames),
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "4",
            "-",
        ]
    )
    frames = [f for f in raw.split(b"\xff\xd9") if f.startswith(b"\xff\xd8")]
    frames = [f + b"\xff\xd9" for f in frames]
    return [(round(i / fps, 2), f) for i, f in enumerate(frames[: p.max_frames])]


def preprocess_video(path: Path, p: VideoParams, img: ImageParams) -> list[Piece]:
    frames = extract_frames(path, p)
    if not frames:
        raise ValidationError("No frames could be extracted from the video.")
    out = []
    for idx, (t, jpg) in enumerate(frames):
        out.append(
            Piece(
                "video",
                [image_part(jpg)],
                {"t": [t, t], "frame": idx},
                tokens_est=img.token_cap_max or FRAME_TOKENS,
                preview=jpg,
                preview_ext=".jpg",
                meta={"frames": len(frames)},
            )
        )
    return out


# --------------------------------------------------------------------------- dispatch


def preprocess(path: Path, kind: Sniffed, params: EmbedParams) -> list[Piece]:
    if kind.modality == "text":
        return preprocess_text(path.read_bytes(), params.text, params.prefix.document)
    if kind.modality == "image":
        return preprocess_image(path.read_bytes(), params.image)
    if kind.modality == "audio":
        return preprocess_audio(path, params.audio)
    if kind.modality == "video":
        return preprocess_video(path, params.video, params.image)
    raise ValidationError(f"Unsupported modality {kind.modality}.")


def fit_ubatch(piece: Piece, ubatch: int) -> list[Piece]:
    """FR-011 AC2: never send a request larger than one physical batch; split text, refuse media."""
    if piece.tokens_est <= ubatch:
        return [piece]
    if piece.modality == "text" and piece.text:
        words = piece.text.split()
        half = len(words) // 2
        a = Piece("text", " ".join(words[:half]), {**piece.span, "split": 0}, text=" ".join(words[:half]))
        b = Piece("text", " ".join(words[half:]), {**piece.span, "split": 1}, text=" ".join(words[half:]))
        for x in (a, b):
            x.tokens_est = estimate_tokens(str(x.content))
        return fit_ubatch(a, ubatch) + fit_ubatch(b, ubatch)
    raise ValidationError(f"A {piece.modality} segment needs ~{piece.tokens_est} tokens, above the {ubatch}-token batch; lower its size/window.")
