"""Synthetic timeline-to-SRT check. No model, GPU, network, or external package."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def main():
    timeline = module('timeline', 'code/vad-gen/src/align/timeline.py')
    srt = module('srt_export', 'code/vad-gen/src/output/srt_export.py')
    video = [dict(start=0, end=3, emotion='Neutral', gestures=set())]
    audio = dict(
        transcript=[dict(words=[dict(start=0.2, end=0.6, word='Hello.')])],
        speech=[dict(start=0, end=1)],
        silence=[dict(start=1, end=3)],
    )
    result = timeline.build_timeline(video, audio)
    assert [x['type'] for x in result] == ['speech', 'silence']
    assert result[0]['text'] == 'Hello.'
    result[0]['description'] = result[0]['text']
    result[1]['description'] = 'A neutral expression.'
    expected = (
        '1\n00:00:00,000 --> 00:00:01,000\nHello.\n\n'
        '2\n00:00:01,000 --> 00:00:03,000\n[A neutral expression.]\n'
    )
    actual = srt.timeline_to_srt(result)
    assert actual == expected, repr(actual)
    result[1]['description'] = ''
    assert '[A neutral expression.]' not in srt.timeline_to_srt(result)
    print('PASS: synthetic timeline alignment and SRT export; no model inference performed.')
    print(actual)


if __name__ == '__main__':
    main()
