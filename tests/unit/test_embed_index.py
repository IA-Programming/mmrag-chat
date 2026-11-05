"""U-UPLOAD-sniff/bomb/limits, U-PRE-text/image/audio/video (knobs change artifacts), U-PRE-ubatch,
U-STORE-invariant, U-STORE-parity, U-MATRYOSHKA, U-PCA, U-EXPORT-roundtrip, U-SEC-zip, U-SEC-upload."""

from __future__ import annotations

import io
import subprocess
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.analysis import matryoshka_compare, overlap_at_k, pca_2d
from core.embed.client import truncate_renorm
from core.embed.preprocess import (
    Piece,
    audio_windows,
    chunk_text,
    ffmpeg_exe,
    fit_ubatch,
    preprocess_audio,
    preprocess_image,
    preprocess_video,
    sniff,
    validate_upload,
)
from core.errors import ValidationError
from core.index.export_import import export_index, import_index
from core.index.store import IndexHandle, NumpyStore, TurboVecStore
from core.schemas import AudioParams, ImageParams, TextParams, VideoParams


def png_bytes(w: int = 64, h: int = 48, color=(200, 10, 10)) -> bytes:  # type: ignore[no-untyped-def]
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def wav_bytes(seconds: float, rate: int = 16000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    pcm = (np.sin(2 * np.pi * 440 * t) * 8000).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# --------------------------------------------------------------------------- upload validation
def test_sniff_by_magic_bytes() -> None:
    assert sniff(png_bytes()).modality == "image"  # type: ignore[union-attr]
    assert sniff(wav_bytes(0.1)).modality == "audio"  # type: ignore[union-attr]
    assert sniff(b"# hello\nworld").modality == "text"  # type: ignore[union-attr]
    assert sniff(b"MZ\x90\x00\x03\x00\x00\x00binary\x00\x00") is None


def test_renamed_exe_rejected() -> None:
    with pytest.raises(ValidationError, match="does not look like|content"):
        validate_upload("photo.png", b"MZ\x90\x00" + b"\x00" * 100)


def test_extension_content_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="content is image/png"):
        validate_upload("notes.md", png_bytes())


def test_decompression_bomb_rejected_before_decode() -> None:
    big = png_bytes(4000, 3000)
    with pytest.raises(ValidationError, match="MP"):
        validate_upload("big.png", big, max_pixels=1_000_000)


def test_limits() -> None:
    with pytest.raises(ValidationError, match="upload limit"):
        validate_upload("a.txt", b"x" * 1000, used_bytes=200 * 1024 * 1024)
    with pytest.raises(ValidationError, match="not supported"):
        validate_upload("a.exe", b"x")
    with pytest.raises(ValidationError, match="empty"):
        validate_upload("a.txt", b"")


# --------------------------------------------------------------------------- preprocessing knobs
def test_text_chunk_knobs_change_output() -> None:
    text = "\n\n".join(" ".join(f"word{p}_{i}" for i in range(80)) for p in range(10))
    small = chunk_text(text, TextParams(chunk_tokens=64, overlap_tokens=0))
    large = chunk_text(text, TextParams(chunk_tokens=512, overlap_tokens=0))
    overl = chunk_text(text, TextParams(chunk_tokens=64, overlap_tokens=32))
    assert len(small) > len(large) and len(overl) > len(small)
    assert all(a < b for _, (a, b) in small)


def test_image_max_side_changes_artifact() -> None:
    data = png_bytes(2000, 1000)
    a = preprocess_image(data, ImageParams(max_side_px=512))[0]
    b = preprocess_image(data, ImageParams(max_side_px=1024))[0]
    assert a.meta["width"] == 512 and b.meta["width"] == 1024
    assert len(a.content[0]["image_url"]["url"]) < len(b.content[0]["image_url"]["url"])


def test_audio_windows_and_rate(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    p.write_bytes(wav_bytes(25))
    w10 = preprocess_audio(p, AudioParams(window_s=10, overlap_s=0))
    w30 = preprocess_audio(p, AudioParams(window_s=30, overlap_s=0))
    assert len(w10) == 3 and len(w30) == 1 and w10[1].span["t"] == [10.0, 20.0]
    r8 = preprocess_audio(p, AudioParams(window_s=30, sample_rate_hz=8000))[0]
    assert len(r8.preview or b"") < len(w30[0].preview or b"")
    assert audio_windows(16000 * 5, 16000, AudioParams(window_s=2, overlap_s=1))[1] == (16000, 48000)


def test_video_frames_knobs(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=4", "-pix_fmt", "yuv420p", str(p)],
        check=True,
    )
    f1 = preprocess_video(p, VideoParams(fps=1, max_frames=16, frame_max_side_px=224), ImageParams())
    f2 = preprocess_video(p, VideoParams(fps=2, max_frames=16, frame_max_side_px=224), ImageParams())
    f3 = preprocess_video(p, VideoParams(fps=2, max_frames=3, frame_max_side_px=224), ImageParams())
    assert len(f2) > len(f1) and len(f3) == 3 and f1[1].span["t"][0] == pytest.approx(1.0)


def test_ubatch_split_or_refuse() -> None:
    big = Piece("text", "x " * 3000, {"chunk": 0}, text=" ".join(["w"] * 3000), tokens_est=4000)
    parts = fit_ubatch(big, 1024)
    assert len(parts) > 1 and all(p.tokens_est <= 1024 for p in parts)
    with pytest.raises(ValidationError):
        fit_ubatch(Piece("audio", [], {}, tokens_est=5000), 2048)


# --------------------------------------------------------------------------- stores
def _vecs(n: int, d: int = 768, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return truncate_renorm(v, d)


def test_store_invariant_and_parity(tmp_path: Path) -> None:
    h = IndexHandle(tmp_path / "ix", bits=4)
    v = _vecs(300)
    h.add_source("s", "f.md", "text", "a" * 64, 10)
    for start in range(0, 300, 100):
        h.add_items(
            "s",
            [{"modality": "text", "chunk_idx": i, "span": {}, "text": str(i), "tokens_est": 1} for i in range(start, start + 100)],
            v[start : start + 100],
            "p",
        )
        assert h.count() == h.sidecar_count() == h.vectors.shape[0]
    with pytest.raises(ValueError):
        h.add_items("s", [{"modality": "text", "chunk_idx": 0, "span": {}, "tokens_est": 1}], v[:2], "p")
    assert h.count() == h.sidecar_count() == 300  # failed add left no partial rows
    exact = NumpyStore(768)
    exact.add(v)
    _, ie = exact.search(v[:20], 1)
    _, it = h.store.search(v[:20], 1)
    assert (ie[:, 0] == it[:, 0]).all()  # self-retrieval parity
    r = h.recall_at_k(10)
    assert r is not None and r > 0.6


def test_turbovec_mask() -> None:
    s = TurboVecStore(128, 4)
    v = _vecs(50, 128)
    s.add(v)
    mask = np.zeros(50, bool)
    mask[10:20] = True
    _, ids = s.search(v[0], 5, mask)
    assert set(ids[0].tolist()) <= set(range(10, 20))


def test_export_import_roundtrip(tmp_path: Path) -> None:
    h = IndexHandle(tmp_path / "a", fingerprint="fp1")
    v = _vecs(30)
    h.add_source("s", "f.md", "text", "a" * 64, 10)
    h.add_items("s", [{"modality": "text", "chunk_idx": i, "span": {"chunk": i}, "text": f"t{i}", "tokens_est": 1} for i in range(30)], v, "p")
    before = [x.item_id for x in h.search(v[7], 5)]
    blob = export_index(h)
    h2 = import_index(blob, tmp_path / "b", expected_fingerprint="fp1", expected_dim=768)
    assert [x.item_id for x in h2.search(v[7], 5)] == before and h2.item(7).text == "t7"  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="different embedder"):
        import_index(blob, tmp_path / "c", expected_fingerprint="other", expected_dim=768)
    with pytest.raises(ValidationError, match="dimension"):
        import_index(blob, tmp_path / "c", expected_fingerprint="fp1", expected_dim=512)


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, val in entries.items():
            z.writestr(k, val)
    return buf.getvalue()


def test_zip_attacks_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unsafe path"):
        import_index(_zip({"../evil": b"x", "manifest.json": b"{}"}), tmp_path / "x", expected_fingerprint="", expected_dim=768)
    with pytest.raises(ValidationError, match="Unexpected entry"):
        import_index(_zip({"payload.pkl": b"x"}), tmp_path / "x", expected_fingerprint="", expected_dim=768)
    with pytest.raises(ValidationError, match="zip bomb|too large"):
        import_index(
            _zip({"manifest.json": b"{}", "index.tv": b"\0" * (50 * 1024 * 1024), "sidecar.sqlite": b""}),
            tmp_path / "x",
            expected_fingerprint="",
            expected_dim=768,
        )
    with pytest.raises(ValidationError):
        import_index(b"not a zip", tmp_path / "x", expected_fingerprint="", expected_dim=768)


# --------------------------------------------------------------------------- analysis
def test_matryoshka_truncation_unit_norm_and_self_overlap() -> None:
    v = _vecs(100)
    for d in (768, 512, 256, 128):
        t = truncate_renorm(v, d)
        assert t.shape[1] == d and np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)
    res = matryoshka_compare(v, v[5], [768, 512, 256, 128], 5)
    assert res[768]["overlap"] == 1.0 and res[768]["ids"][0] == 5
    assert overlap_at_k([1, 2, 3], [1, 2, 3], 3) == 1.0


def test_pca_deterministic_and_subsampled() -> None:
    v = _vecs(2000)
    a, ia, _ = pca_2d(v, max_points=500, seed=3)
    b, ib, _ = pca_2d(v.copy(), max_points=500, seed=3)
    assert np.array_equal(ia, ib) and np.allclose(a, b) and a.shape == (500, 2)
