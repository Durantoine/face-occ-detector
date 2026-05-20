from pathlib import Path
from typing import Any, Dict, List

CONFIG: Dict[str, Any] = {
    "sources": [
        "data/extra/lfw/",
        "data/extra/celeba_raw/img_align_celeba/",
        "data/extra/vggface2/",
    ],
    "output_dir": "data/pretrain/",
    "use_symlinks": True,
    "max_per_source": None,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _scan(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def prepare(
    sources: List[str],
    output_dir: str,
    use_symlinks: bool,
    max_per_source: int = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            print(f"[skip] {src} — not present")
            continue
        images = _scan(src_path)
        if max_per_source:
            images = images[:max_per_source]
        link_dir = out / src_path.name
        link_dir.mkdir(exist_ok=True)
        n = 0
        for img in images:
            dest = link_dir / img.name
            if dest.exists():
                continue
            if use_symlinks:
                dest.symlink_to(img.resolve())
            else:
                import shutil
                shutil.copy2(img, dest)
            n += 1
        print(f"[{src_path.name}] {n:,} new images linked into {link_dir}")
        total += n
    print(f"Total pretraining corpus: {total:,} images under {output_dir}")
    print(f"Use as `data_dir` in src/pretrain_mae.py CONFIG.")


if __name__ == "__main__":
    prepare(
        sources=CONFIG["sources"],
        output_dir=CONFIG["output_dir"],
        use_symlinks=CONFIG["use_symlinks"],
        max_per_source=CONFIG["max_per_source"],
    )
