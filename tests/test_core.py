import unittest

from src.align.segment import segment, smooth_gesture_sets, smooth_labels
from src.align.timeline import build_timeline
from src.generate.describe import DescriptionGenerator
from src.output.srt_export import timeline_to_srt


class SegmentTests(unittest.TestCase):
    def test_smoothing_uses_local_majority(self):
        labels = ["Neutral", "Neutral", "Anger", "Neutral", "Neutral"]
        self.assertEqual(smooth_labels(labels, window=3)[2], "Neutral")

    def test_gesture_smoothing_uses_ratio(self):
        gestures = [{"head_down"}, set(), {"head_down"}]
        self.assertEqual(
            smooth_gesture_sets(gestures, window=3, ratio=0.5)[1], {"head_down"}
        )

    def test_short_segment_is_merged(self):
        result = segment(
            [0.0, 1.0, 1.2],
            ["Neutral", "Anger", "Neutral"],
            [set(), set(), set()],
            min_duration=0.5,
        )
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 1.2)


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.video = [
            {"start": 0.0, "end": 2.0, "emotion": "Neutral", "gestures": set()},
            {
                "start": 2.0,
                "end": 4.0,
                "emotion": "Surprise",
                "gestures": {"head_turned"},
            },
        ]
        self.audio = {
            "transcript": [{"words": [{"start": 2.2, "end": 2.5, "word": " Hello"}]}],
            "speech": [{"start": 2.0, "end": 3.0}],
            "silence": [{"start": 0.0, "end": 2.0}],
        }

    def test_build_timeline_merges_modalities(self):
        timeline = build_timeline(self.video, self.audio)
        self.assertEqual(timeline[0]["type"], "silence")
        self.assertEqual(timeline[0]["emotion"], "Neutral")
        self.assertEqual(timeline[1]["text"], "Hello")
        self.assertEqual(timeline[1]["gestures"], {"head_turned"})

    def test_deterministic_description_needs_no_ollama(self):
        timeline = build_timeline(self.video, self.audio)
        described = DescriptionGenerator(use_llm=False).generate(timeline)
        self.assertEqual(described[1]["description"], "Hello")
        self.assertTrue(described[0]["description"])

    def test_srt_skips_empty_descriptions(self):
        timeline = [
            {
                "start": 0.0,
                "end": 1.0,
                "type": "silence",
                "description": "A neutral expression.",
            },
            {"start": 1.0, "end": 2.0, "type": "silence", "description": ""},
        ]
        srt = timeline_to_srt(timeline)
        self.assertIn("[A neutral expression.]", srt)
        self.assertNotIn("00:00:01,000 --> 00:00:02,000", srt)


if __name__ == "__main__":
    unittest.main()
