"""
SRT 자막 생성 + ffmpeg로 영상에 자막 입히기 (vad-gen)

타임라인 -> SRT 파일 -> 자막 입힌 데모 영상
"""

import shutil
import subprocess
from pathlib import Path

# 기계 라벨 -> 읽기 쉬운 표기 (LLM 단계 전 임시 확인용)
EMOTION_DISPLAY = {
    "Anger": "angry",
    "Disgust": "disgusted",
    "Fear": "fearful",
    "Happiness": "happy",
    "Sadness": "sad",
    "Surprise": "surprised",
    "Neutral": "neutral",
}

GESTURE_DISPLAY = {
    "hand_near_face": "hand near face",
    "hand_near_neck": "hand near neck",
    "head_turned": "head turned",
    "head_down": "head down",
    "hands_lowered": "hands lowered",
}


def _fmt_time(sec):
    """초 -> SRT 시간 포맷 (00:00:00,000)"""
    ms = round(sec * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def timeline_to_srt(timeline, include_nonverbal=True):
    """
    타임라인 -> SRT 문자열
    대사는 그대로, 비언어는 대괄호로 (화면해설 관례)
    """
    blocks = []
    idx = 1

    for item in timeline:
        lines = []

        # DescriptionGenerator를 거친 타임라인이면 생성된 결과를 우선 사용한다.
        # 변화가 없는 침묵(description == '')은 실제 SRT 항목에서 제외한다.
        if "description" in item:
            description = item.get("description", "").strip()
            if not description:
                continue
            if item["type"] == "speech":
                lines.append(description)
            else:
                lines.append(f"[{description.strip('[]')}]")
        else:
            # 이전 호출부와의 호환을 위한 원시 타임라인 출력
            if item["type"] == "speech" and item["text"]:
                lines.append(item["text"])
            elif item["type"] == "silence":
                lines.append("[silence]")
            else:
                lines.append("[...]")

        if include_nonverbal and "description" not in item:
            nv = []
            if item["emotion"]:
                nv.append(EMOTION_DISPLAY.get(item["emotion"], item["emotion"].lower()))
            for g in sorted(item["gestures"]):
                nv.append(GESTURE_DISPLAY.get(g, g))
            if nv:
                lines.append(f"[{', '.join(nv)}]")

        blocks.append(
            f"{idx}\n"
            f"{_fmt_time(item['start'])} --> {_fmt_time(item['end'])}\n"
            + "\n".join(lines)
            + "\n"
        )
        idx += 1

    return "\n".join(blocks)


def save_srt(timeline, out_path, include_nonverbal=True):
    srt = timeline_to_srt(timeline, include_nonverbal)
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(srt)
    return out_path


def burn_subtitles(
    video_path, srt_path, out_path, font_size=16, font_name="DejaVu Sans"
):
    """
    ffmpeg로 자막을 영상에 입힘 (hardsub).
    font_name: 시스템에 있는 폰트. 한글 쓰려면 'NanumGothic' 등 필요.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg가 설치되어 있지 않습니다. README의 설치 절차를 확인하세요."
        )

    video_path = Path(video_path).expanduser().resolve()
    srt_path = Path(srt_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    if not srt_path.is_file():
        raise FileNotFoundError(f"SRT 파일을 찾을 수 없습니다: {srt_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg subtitles 필터는 경로에 특수문자가 있으면 문제가 생겨 이스케이프
    srt_escaped = (
        str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    )

    style = (
        f"FontName={font_name},FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,"
        f"BorderStyle=3,Outline=1,Shadow=0,MarginV=20"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"subtitles='{srt_escaped}':force_style='{style}'",
        "-c:a",
        "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg 자막 처리에 실패했습니다:\n" + result.stderr[-1500:])
    return out_path


if __name__ == "__main__":
    print("이 모듈은 run.py를 통해 사용하세요.")
