"""Copy inventoried external originals into a portable archive; dry run by default."""
import argparse
import csv
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--copy', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / 'manifests/files.csv').open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if not r['bundle_path']]
    print(f'{len(rows)} files, {sum(int(r["bytes"]) for r in rows):,} bytes')
    print(f'Destination: {args.destination.resolve()}')
    if not args.copy:
        print('Dry run only. Add --copy to copy original assets. No files changed.')
        return
    # Validate all originals and targets before the first copy. Never overwrite.
    pairs = []
    for row in rows:
        source = Path(row['original_path'])
        relative = Path(row['project']) / row['relative_path']
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid manifest path: {relative}')
        target = args.destination / relative
        stat = source.stat()
        if stat.st_size != int(row['bytes']) or stat.st_mtime_ns != int(row['mtime_ns']):
            raise RuntimeError(f'Original changed since inventory: {source}')
        if target.exists():
            raise FileExistsError(target)
        pairs.append((source, target))
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f'Copied {len(pairs)} files. Model and data licenses remain unchanged.')


if __name__ == '__main__':
    main()
