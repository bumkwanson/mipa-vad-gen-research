"""
시간축 통합 모듈 (vad-gen)

영상 분석(표정/제스처 구간) + 오디오 분석(대사/발화/침묵)을
하나의 타임라인으로 병합.

각 항목은 발화 또는 침묵 구간이며, 항상 그 시간대의 비언어 정보를 함께 가짐.
"""
from collections import Counter


def _overlap(a_start, a_end, b_start, b_end):
    """두 구간의 겹치는 길이"""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _nonverbal_at(segments, start, end):
    """
    [start, end] 구간과 가장 많이 겹치는 영상 구간의 표정/제스처를 반환.
    여러 구간에 걸치면 겹침이 가장 큰 것을 대표로.
    """
    best = None
    best_ov = 0.0
    emo_votes = Counter()
    gesture_union = Counter()

    for seg in segments:
        ov = _overlap(start, end, seg['start'], seg['end'])
        if ov <= 0:
            continue
        if seg['emotion']:
            emo_votes[seg['emotion']] += ov
        for g in seg['gestures']:
            gesture_union[g] += ov
        if ov > best_ov:
            best_ov = ov
            best = seg

    if best is None:
        return {'emotion': None, 'gestures': set()}

    emotion = emo_votes.most_common(1)[0][0] if emo_votes else None
    # 해당 구간 길이의 절반 이상 지속된 제스처만 채택
    dur = max(1e-6, end - start)
    gestures = {g for g, ov in gesture_union.items() if ov / dur >= 0.5}

    return {'emotion': emotion, 'gestures': gestures}


def _words_in(words, start, end):
    """구간에 속하는 단어들을 모아 문장으로"""
    sel = [w['word'] for w in words
           if w['start'] is not None and start - 0.15 <= w['start'] < end + 0.15]
    return ''.join(sel).strip()


def build_timeline(video_segments, audio_result):
    """
    video_segments: segment.py의 결과 [{'start','end','emotion','gestures'}, ...]
    audio_result  : audio_analysis.py의 결과 dict

    반환: [{'start','end','type','text','emotion','gestures'}, ...]
          type: 'speech' | 'silence'
    """
    # 모든 단어를 하나의 리스트로
    all_words = []
    for seg in audio_result['transcript']:
        all_words.extend(seg.get('words', []))

    # 발화/침묵 구간을 하나로 합쳐 시간순 정렬
    events = []
    for s in audio_result['speech']:
        events.append({'start': s['start'], 'end': s['end'], 'type': 'speech'})
    for s in audio_result['silence']:
        events.append({'start': s['start'], 'end': s['end'], 'type': 'silence'})
    events.sort(key=lambda e: e['start'])

    timeline = []
    for ev in events:
        nv = _nonverbal_at(video_segments, ev['start'], ev['end'])
        text = ''
        if ev['type'] == 'speech':
            text = _words_in(all_words, ev['start'], ev['end'])

        timeline.append({
            'start': ev['start'],
            'end': ev['end'],
            'type': ev['type'],
            'text': text,
            'emotion': nv['emotion'],
            'gestures': nv['gestures'],
        })

    return timeline


def format_timeline(timeline):
    """사람이 읽기 좋게 출력"""
    lines = []
    for item in timeline:
        head = f"[{item['start']:5.1f}s - {item['end']:5.1f}s]"
        if item['type'] == 'speech':
            body = f'"{item["text"]}"' if item['text'] else '(speech, no words)'
        else:
            body = '[silence]'

        nv_parts = []
        if item['emotion']:
            nv_parts.append(item['emotion'])
        if item['gestures']:
            nv_parts.extend(sorted(item['gestures']))
        nv = ', '.join(nv_parts) if nv_parts else '-'

        lines.append(f"{head}  {body}")
        lines.append(f"{'':17}  ({nv})")
        lines.append('')
    return '\n'.join(lines)
