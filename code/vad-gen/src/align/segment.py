"""
시간축 스무딩 + 구간화 모듈 (vad-gen)

프레임별 관찰 결과(표정 라벨, 제스처 라벨)는 노이즈가 심하므로
1) 슬라이딩 윈도우 다수결로 스무딩
2) 연속된 동일 상태를 구간으로 병합
3) 너무 짧은 구간은 인접 구간에 흡수

출력은 화면해설 생성 단계(LLM)에 넘길 '안정된 구간 목록'.
"""
from collections import Counter, deque


def smooth_labels(labels, window=9):
    """
    labels: 프레임별 라벨 리스트 (str 또는 None)
    window: 다수결 윈도우 크기 (홀수 권장). fps 25 기준 9면 약 0.36초.
    반환: 같은 길이의 스무딩된 라벨 리스트
    """
    if window <= 1:
        return list(labels)

    half = window // 2
    out = []
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        chunk = [l for l in labels[lo:hi] if l is not None]
        if not chunk:
            out.append(None)
        else:
            out.append(Counter(chunk).most_common(1)[0][0])
    return out


def smooth_gesture_sets(gesture_sets, window=9, ratio=0.5):
    """
    제스처는 집합(set)이라 라벨별로 따로 다수결.
    윈도우 내에서 해당 제스처가 ratio 이상 비율로 나타나면 유지.
    """
    if window <= 1:
        return [set(g) for g in gesture_sets]

    half = window // 2
    out = []
    for i in range(len(gesture_sets)):
        lo = max(0, i - half)
        hi = min(len(gesture_sets), i + half + 1)
        chunk = gesture_sets[lo:hi]
        n = len(chunk)
        counts = Counter()
        for g in chunk:
            counts.update(g)
        kept = {label for label, c in counts.items() if c / n >= ratio}
        out.append(kept)
    return out


def segment(times, emotions, gestures, min_duration=0.8):
    """
    스무딩된 프레임별 결과를 구간으로 병합.

    times     : 프레임별 시간(초) 리스트
    emotions  : 스무딩된 표정 라벨 리스트
    gestures  : 스무딩된 제스처 set 리스트
    min_duration: 이보다 짧은 구간은 앞 구간에 흡수 (초)

    반환: [{'start','end','emotion','gestures'}, ...]
    """
    if not times:
        return []

    segs = []
    cur = {
        'start': times[0],
        'end': times[0],
        'emotion': emotions[0],
        'gestures': set(gestures[0]),
        '_emo_votes': Counter([emotions[0]] if emotions[0] else []),
    }

    for i in range(1, len(times)):
        same_emotion = emotions[i] == cur['emotion']
        same_gesture = set(gestures[i]) == cur['gestures']

        if same_emotion and same_gesture:
            cur['end'] = times[i]
            if emotions[i]:
                cur['_emo_votes'][emotions[i]] += 1
        else:
            segs.append(cur)
            cur = {
                'start': times[i],
                'end': times[i],
                'emotion': emotions[i],
                'gestures': set(gestures[i]),
                '_emo_votes': Counter([emotions[i]] if emotions[i] else []),
            }
    segs.append(cur)

    # 너무 짧은 구간 흡수: 앞 구간에 병합 (앞이 없으면 뒤에)
    merged = []
    for s in segs:
        dur = s['end'] - s['start']
        if dur < min_duration and merged:
            prev = merged[-1]
            prev['end'] = s['end']
            prev['_emo_votes'] += s['_emo_votes']
            # 흡수된 구간의 제스처는 지배적이지 않으므로 버림
        else:
            merged.append(s)

    # 병합 후 지배적 감정으로 라벨 확정
    for s in merged:
        if s['_emo_votes']:
            s['emotion'] = s['_emo_votes'].most_common(1)[0][0]
        del s['_emo_votes']

    # 라벨 확정 후 인접한 동일 상태 구간을 한 번 더 병합
    final = []
    for s in merged:
        if final and final[-1]['emotion'] == s['emotion'] and            final[-1]['gestures'] == s['gestures']:
            final[-1]['end'] = s['end']
        else:
            final.append(s)

    return final


def format_segments(segs):
    """사람이 읽기 좋게 출력"""
    lines = []
    for s in segs:
        g = ', '.join(sorted(s['gestures'])) if s['gestures'] else '-'
        lines.append(f"[{s['start']:5.1f}s - {s['end']:5.1f}s] "
                     f"표정: {s['emotion'] or '없음':<10} 동작: {g}")
    return '\n'.join(lines)
