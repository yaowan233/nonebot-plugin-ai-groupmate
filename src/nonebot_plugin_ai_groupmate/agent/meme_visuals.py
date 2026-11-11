"""Small, labelled contact sheets built from retrieved local candidates."""

import base64
from io import BytesIO
from pathlib import Path
from collections.abc import Sequence

from PIL import Image, ImageOps, ImageDraw

from ..image_preparation import prepare_image

VISUAL_CANDIDATE_LIMIT = 6
TILE_SIZE = 384
VISUAL_FRAME_BUDGET = 18
VISUAL_ANIMATION_MAX_FRAMES = 6


def build_meme_review_images(root: Path, candidates: Sequence[tuple[int, str]]) -> tuple[list[dict], set[int]]:
    """Still candidates share a grid; animations get their own readable storyboard."""
    blocks: list[dict] = []
    ids: set[int] = set()
    stills: list[tuple[int, str]] = []
    animations: list[tuple[int, Path]] = []
    resolved_root = root.resolve()
    for media_id, relative_path in candidates[:VISUAL_CANDIDATE_LIMIT]:
        try:
            path = (root / relative_path).resolve()
            if not path.is_relative_to(resolved_root):
                continue
            with Image.open(path) as source:
                animated = getattr(source, "n_frames", 1) > 1
            if not animated:
                stills.append((media_id, relative_path))
                continue
            animations.append((media_id, path))
        except (OSError, ValueError, EOFError, Image.DecompressionBombError):
            continue
    if stills:
        grid, still_ids = build_meme_contact_sheet(root, stills)
        if grid:
            blocks.extend(
                [
                    {"type": "text", "text": "静态候选拼图，每格 pic_id 对应一个候选："},
                    {"type": "image_url", "image_url": {"url": grid}},
                ]
            )
            ids.update(still_ids)
    # Reserve one unit for the entire static grid and share the rest across
    # animations, so early long animations cannot hide later candidates.
    if animations:
        max_frames = min(VISUAL_ANIMATION_MAX_FRAMES, (VISUAL_FRAME_BUDGET - bool(ids)) // len(animations))
        for media_id, path in animations:
            try:
                prepared = prepare_image(path.read_bytes(), max_frames=max_frames)
                blocks.extend(
                    [
                        {"type": "text", "text": f"pic_id: {media_id}（动图，以下是 {prepared.selected_frames} 个关键帧）："},
                        {"type": "image_url", "image_url": {"url": prepared.data_uri}},
                    ]
                )
                ids.add(media_id)
            except (OSError, ValueError, EOFError, Image.DecompressionBombError):
                continue
    return blocks, ids


def build_meme_contact_sheet(
    root: Path,
    candidates: Sequence[tuple[int, str]],
) -> tuple[str | None, set[int]]:
    """Keep readable candidates only; labels are the original database IDs."""
    tiles = []
    ids: set[int] = set()
    resolved_root = root.resolve()
    for media_id, relative_path in candidates[:VISUAL_CANDIDATE_LIMIT]:
        try:
            path = (root / relative_path).resolve()
            if not path.is_relative_to(resolved_root):
                continue
            with Image.open(path) as source:
                # Animations are reviewed separately by build_meme_review_images.
                if getattr(source, "n_frames", 1) > 1:
                    continue
                source.seek(0)
                picture = ImageOps.exif_transpose(source).convert("RGBA")
                picture.thumbnail((TILE_SIZE - 16, TILE_SIZE - 44))
                tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), "white")
                tile.paste(picture, ((TILE_SIZE - picture.width) // 2, 38), picture)
                ImageDraw.Draw(tile).text((10, 10), f"pic_id: {media_id}", fill="black")
            tiles.append(tile)
            ids.add(media_id)
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
    if not tiles:
        return None, set()
    columns = min(2, len(tiles))
    sheet = Image.new("RGB", (columns * TILE_SIZE, ((len(tiles) + columns - 1) // columns) * TILE_SIZE), "#dddddd")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * TILE_SIZE, (index // columns) * TILE_SIZE))
    data = BytesIO()
    sheet.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii"), ids
