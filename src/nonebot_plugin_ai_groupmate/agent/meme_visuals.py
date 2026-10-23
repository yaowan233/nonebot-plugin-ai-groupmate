"""Small, labelled contact sheets built from retrieved local candidates."""

import base64
from io import BytesIO
from pathlib import Path
from collections.abc import Sequence

from PIL import Image, ImageOps, ImageDraw

VISUAL_CANDIDATE_LIMIT = 6
TILE_SIZE = 384


def build_meme_contact_sheet(
    root: Path, candidates: Sequence[tuple[int, str]],
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
                # Animated memes get a representative first frame, not an
                # invented description; animation semantics remain a limitation.
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
