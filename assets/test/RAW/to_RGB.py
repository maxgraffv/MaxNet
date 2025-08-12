import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Brak biblioteki Pillow. Zainstaluj: pip install pillow")
    sys.exit(1)

def convert_image(img_path: Path, out_dir: Path):
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            raw = im.tobytes()  # interleaved RGB, length = w*h*3

        out_dir.mkdir(parents=True, exist_ok=True)
        out_rgb = out_dir / (img_path.stem + ".rgb")
        out_meta = out_dir / (img_path.stem + ".meta.json")

        with open(out_rgb, "wb") as f:
            f.write(raw)

        meta = {"width": w, "height": h, "channels": 3, "format": "RGB"}
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return True, f"OK: {img_path.name} -> {out_rgb.name} ({w}x{h})"
    except Exception as e:
        return False, f"ERR: {img_path} — {e}"

def main():
    parser = argparse.ArgumentParser(description="Convert JPG/JPEG images to raw .rgb files.")
    parser.add_argument("input_dir", type=Path, help="Folder z plikami wejściowymi")
    parser.add_argument("output_dir", type=Path, help="Folder wyjściowy na .rgb")
    parser.add_argument("--recursive", "-r", action="store_true", help="Przeszukuj podfoldery")
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print("Błędny folder wejściowy.")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    files = (args.input_dir.rglob("*") if args.recursive else args.input_dir.glob("*"))
    images = [p for p in files if p.suffix in exts and p.is_file()]

    if not images:
        print("Nie znaleziono obrazów .jpg/.jpeg w podanym folderze.")
        sys.exit(0)

    ok = 0
    for p in images:
        success, msg = convert_image(p, args.output_dir)
        print(msg)
        if success: ok += 1

    print(f"\nGotowe. Sukcesów: {ok}/{len(images)}")
    print("Uwaga: .rgb to surowe bajty RGB bez nagłówka. Wymiary znajdziesz w plikach .meta.json.")

if __name__ == "__main__":
    main()

