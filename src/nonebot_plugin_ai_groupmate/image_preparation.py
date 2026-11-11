"""Model previews of raster images; the stored original is never rewritten here."""

import io
import math
import base64
from dataclasses import dataclass

from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageStat, ImageChops

MAX_IMAGE_PIXELS = 40_000_000
MAX_MODEL_SIDE = 2048
MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_ANIMATION_FRAMES = 9
MAX_SIGNATURES = 300
MAX_SOURCE_FRAMES = 2000
MAX_ANIMATION_PIXELS = 300_000_000
SHEET_SIDE = 1536
SHEET_GAP = 6
ANIMATED_MARK = "动图："
ANIMATION_ANNOTATION_VERSION = 1
ANIMATION_NOTE = "灰色细缝隔开、左上角黑底白字编号的多格画面，是同一张动图按时间顺序截取的关键帧；按编号连起来理解动作、表情和文字的变化，不要当成多格漫画或多张图片。关键帧未必等时间间隔，不据此推断精确速度。描述动图本身，不要提编号、格子或拼图。"


class ImagePreparationError(ValueError):
    """Invalid image content or a permanent preview resource limit."""


def is_animated(data: bytes) -> bool:
    """Inspect playback metadata without decoding and encoding a static preview."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            return getattr(image, "n_frames", 1) > 1
    except (OSError, ValueError, EOFError, SyntaxError, Image.DecompressionBombError) as error:
        raise ImagePreparationError(str(error)) from error


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime_type: str
    source_frames: int
    selected_frames: int

    @property
    def animated(self) -> bool:
        return self.source_frames > 1

    @property
    def data_uri(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode('ascii')}"


def image_suffix(data: bytes, fallback: str = "jpg") -> str:
    """Use the actual format, including when a QQ URL has no useful extension."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            return {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp", "BMP": "bmp"}.get(image.format or "", fallback)
    except (OSError, ValueError, Image.DecompressionBombError):
        return fallback


def _on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    flat = Image.new("RGB", rgba.size, "white")
    flat.paste(rgba, mask=rgba.getchannel("A"))
    return flat


def _difference(a: Image.Image, b: Image.Image) -> float:
    return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / (3 * 255)


def _key_frames(image: Image.Image, count: int, max_frames: int) -> list[Image.Image]:
    # APNG may contain a separate default image that is not part of playback.
    first = int(bool(image.info.get("default_image")))
    samples = min(count - first, MAX_SIGNATURES)
    indices = [first] if samples == 1 else [first + (count - first - 1) * j // (samples - 1) for j in range(samples)]
    starts: list[float] = []
    signatures: list[Image.Image] = []
    elapsed = 0.0
    sampled = set(indices)
    # Seek sequentially so disposal/compositing and variable frame durations
    # remain correct. Only tiny signatures survive this pass.
    for index in range(first, count):
        image.seek(index)
        if index in sampled:
            starts.append(elapsed)
            signatures.append(_on_white(image.resize((64, 64), Image.Resampling.BILINEAR)))
        duration = image.info.get("duration") or 100
        elapsed += 100 if float(duration) <= 10 else float(duration)

    def differs(index: int, others: list[int], threshold: float) -> bool:
        return all(_difference(signatures[index], signatures[other]) >= threshold for other in others)

    last = len(indices) - 1
    if count - first <= max_frames:
        chosen = list(range(len(indices))) if any(differs(j, [0], 0.005) for j in range(1, len(indices))) else [0]
    else:
        chosen: list[int] = []
        for j in range(len(indices)):
            if not chosen or differs(j, chosen, 0.04):
                chosen.append(j)
        if chosen[-1] != last and differs(last, chosen, 0.02):
            chosen.append(last)
        if len(chosen) > max_frames:
            picks = [chosen[0], chosen[-1]]
            while len(picks) < max_frames:
                picks.append(max((j for j in chosen if j not in picks), key=lambda j: min(_difference(signatures[j], signatures[p]) for p in picks)))
            chosen = sorted(picks)
        while len(chosen) < max_frames:
            best: tuple[float, int] | None = None
            for a, b in zip(chosen, [*chosen[1:], None]):
                end = elapsed if b is None else starts[b]
                middle = (starts[a] + end) / 2
                inside = sorted(range(a + 1, last + 1 if b is None else b), key=lambda j: abs(starts[j] - middle))
                found = next((j for j in inside if differs(j, [a] if b is None else [a, b], 0.005)), None)
                if found is not None and (best is None or end - starts[a] > best[0]):
                    best = (end - starts[a], found)
            if best is None:
                break
            chosen = sorted([*chosen, best[1]])
    frames = []
    for j in chosen:
        image.seek(indices[j])
        frame = ImageOps.exif_transpose(image)
        frame.thumbnail((SHEET_SIDE, SHEET_SIDE))
        frames.append(_on_white(frame))
    return frames


def frame_sheet(frames: list[Image.Image]) -> Image.Image:
    """Number key frames in reading order, preserving each frame's aspect ratio."""
    count = len(frames)
    width, height = frames[0].size
    layouts = [(rows, math.ceil(count / rows)) for rows in range(1, count + 1)]
    layouts = [(r, c) for r, c in layouts if r * c - count < min(r, c)]
    rows, cols = min(layouts, key=lambda rc: (abs(math.log(rc[1] * width / (rc[0] * height))), rc[0] * rc[1] - count))
    scale = min((SHEET_SIDE - (cols - 1) * SHEET_GAP) / (cols * width), (SHEET_SIDE - (rows - 1) * SHEET_GAP) / (rows * height))
    width, height = max(1, int(width * scale)), max(1, int(height * scale))
    sheet = Image.new("RGB", (cols * width + (cols - 1) * SHEET_GAP, rows * height + (rows - 1) * SHEET_GAP), (128, 128, 128))
    draw = ImageDraw.Draw(sheet)
    size = max(14, min(width, height) // 8)
    font = ImageFont.load_default(size=size)
    for number, frame in enumerate(frames, 1):
        row, col = divmod(number - 1, cols)
        x, y = col * (width + SHEET_GAP), row * (height + SHEET_GAP)
        sheet.paste(frame.resize((width, height), Image.Resampling.LANCZOS), (x, y))
        draw.rectangle((x, y, x + round(size * 1.5), y + round(size * 1.3)), fill="black")
        draw.text((x + size * 0.3, y + size * 0.1), str(number), fill="white", font=font)
    return sheet


def prepare_image(data: bytes, *, max_frames: int = MAX_ANIMATION_FRAMES) -> PreparedImage:
    """Prepare a bounded still or animation preview; fail rather than silently lose motion."""
    if not 2 <= max_frames <= MAX_ANIMATION_FRAMES:
        raise ImagePreparationError(f"max_frames must be between 2 and {MAX_ANIMATION_FRAMES}")
    try:
        return _prepare_image(data, max_frames)
    except (OSError, ValueError, EOFError, SyntaxError, Image.DecompressionBombError) as error:
        raise ImagePreparationError(str(error)) from error


def _prepare_image(data: bytes, max_frames: int) -> PreparedImage:
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("image exceeds the pixel limit")
        count = getattr(image, "n_frames", 1)
        if count > MAX_SOURCE_FRAMES:
            raise ValueError("animation exceeds the frame limit")
        if count > 1 and image.width * image.height * count > MAX_ANIMATION_PIXELS:
            raise ValueError("animation exceeds the decode budget")
        if count > 1:
            frames = _key_frames(image, count, max_frames)
            normalized = frame_sheet(frames) if len(frames) > 1 else frames[0]
            selected = len(frames)
        else:
            normalized = ImageOps.exif_transpose(image)
            normalized.thumbnail((MAX_MODEL_SIDE, MAX_MODEL_SIDE))
            transparent = image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)
            normalized = normalized.convert("RGBA" if transparent else "RGB")
            selected = 1
    output = io.BytesIO()
    if normalized.mode == "RGBA":
        normalized.save(output, format="PNG", optimize=True)
        if output.tell() <= MAX_MODEL_BYTES:
            return PreparedImage(output.getvalue(), "image/png", count, selected)
        normalized = _on_white(normalized)
    for quality in (90, 80, 65, 50):
        output = io.BytesIO()
        normalized.save(output, format="JPEG", quality=quality, optimize=True)
        if output.tell() <= MAX_MODEL_BYTES:
            return PreparedImage(output.getvalue(), "image/jpeg", count, selected)
    raise ValueError("image exceeds the model byte limit")


def mark_animation_description(description: str) -> str:
    description = description.strip()
    while description.startswith((ANIMATED_MARK, "动图:")):
        description = description[3:].lstrip()
    return ANIMATED_MARK + description if description else ""
