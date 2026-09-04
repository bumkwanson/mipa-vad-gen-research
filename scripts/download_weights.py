"""POSTER++ 공식 Google Drive 링크에서 추론 가중치를 내려받습니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = (
    ("1aVm_hmJyZ5E_0p25XTbm3X9ophsKqCxv", Path("raf-db-model_best.pth")),
    ("17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt", Path("pretrain/ir50.pth")),
    (
        "1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX",
        Path("pretrain/mobilefacenet_model_best.pth.tar"),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="POSTER++ 추론 가중치를 다운로드합니다."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "weights",
        help="가중치 폴더 (기본값: 프로젝트의 weights)",
    )
    parser.add_argument(
        "--force", action="store_true", help="이미 있는 파일도 다시 받습니다."
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    for file_id, relative_path in WEIGHTS:
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.is_file() and destination.stat().st_size > 0 and not args.force:
            print(f"건너뜀: {destination}")
            continue

        print(f"다운로드: {relative_path}")
        result = gdown.download(id=file_id, output=str(destination), quiet=False)
        if (
            result is None
            or not destination.is_file()
            or destination.stat().st_size == 0
        ):
            raise RuntimeError(
                f"다운로드에 실패했습니다: {relative_path}\n"
                "weights/README.md의 공식 링크에서 직접 내려받아 주세요."
            )

    print(f"가중치 준비 완료: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
