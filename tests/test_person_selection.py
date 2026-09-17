"""Verify person selection before pose inference without loading ONNX models."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ControlNet-v1-1-nightly'))
from annotator.dwpose import wholebody


class PersonSelectionTests(unittest.TestCase):
    def setUp(self):
        self.engine = wholebody.Wholebody.__new__(wholebody.Wholebody)
        self.engine.session_det = object()
        self.engine.session_pose = object()
        self.image = np.zeros((1000, 1000, 3), dtype=np.uint8)

    def infer(self, boxes, **kwargs):
        def pose_result(session, selected, image):
            # Preserve box identity in the returned coordinates and scores.
            keypoints = np.broadcast_to(selected[:, None, :2], (len(selected), 133, 2)).copy()
            scores = np.broadcast_to(selected[:, 0, None] / 100, (len(selected), 133)).copy()
            return keypoints, scores

        with patch.object(wholebody, 'inference_detector', return_value=np.asarray(boxes, dtype=float)), \
                patch.object(wholebody, 'inference_pose', side_effect=pose_result) as pose:
            points, scores = self.engine(self.image, **kwargs)
        return points, scores, pose

    def test_zero_through_six_people(self):
        for count in range(7):
            with self.subTest(count=count):
                boxes = [[i, 0, i + 10, (i + 1) * 20] for i in range(count)]
                points, scores, pose = self.infer(boxes)
                self.assertEqual(len(points), min(count, 4))
                if count == 0:
                    pose.assert_not_called()
                    continue
                expected = list(reversed(boxes))[:4]
                np.testing.assert_array_equal(pose.call_args.args[1], expected)
                np.testing.assert_array_equal(points[:, 0, 0], np.asarray(expected)[:, 0])
                np.testing.assert_array_equal(scores[:, 0], np.asarray(expected)[:, 0] / 100)

    def test_area_ranking_and_stable_ties(self):
        boxes = [[1, 0, 11, 100], [2, 0, 102, 20], [3, 0, 23, 100]]
        _, _, pose = self.infer(boxes)
        np.testing.assert_array_equal(pose.call_args.args[1], [boxes[1], boxes[2], boxes[0]])

    def test_height_filters_apply_before_limit(self):
        boxes = [[1, 0, 901, 90]] + [[i, 0, i + 10, i * 100] for i in range(2, 7)]
        _, _, pose = self.infer(boxes, min_height_px=250, min_height_ratio=0.35)
        np.testing.assert_array_equal(pose.call_args.args[1], boxes[:2:-1])

    def test_all_filtered_skips_pose(self):
        points, _, pose = self.infer([[0, 0, 100, 100]], min_height_px=200)
        self.assertEqual(len(points), 0)
        pose.assert_not_called()


if __name__ == '__main__':
    unittest.main()
