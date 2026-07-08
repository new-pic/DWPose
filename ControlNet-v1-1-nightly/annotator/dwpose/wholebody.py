from pathlib import Path

import cv2
import numpy as np

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose

class Wholebody:
    def __init__(self):
        available_providers = ort.get_available_providers()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
        base_dir = Path(__file__).resolve().parents[2]
        onnx_det = base_dir / 'annotator' / 'ckpts' / 'yolox_l.onnx'
        onnx_pose = base_dir / 'annotator' / 'ckpts' / 'dw-ll_ucoco_384.onnx'

        self.session_det = ort.InferenceSession(path_or_bytes=str(onnx_det), providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=str(onnx_pose), providers=providers)
    
    def __call__(self, oriImg, min_height_ratio=0.0, min_height_px=0.0):
        det_result = inference_detector(self.session_det, oriImg)
        
        if len(det_result) > 0 and (min_height_ratio > 0.0 or min_height_px > 0.0):
            h, w = oriImg.shape[:2]
            filtered_dets = []
            for box in det_result:
                box_h = box[3] - box[1]
                if min_height_px > 0.0 and box_h < min_height_px:
                    continue
                if min_height_ratio > 0.0 and (box_h / h) < min_height_ratio:
                    continue
                filtered_dets.append(box)
            det_result = np.array(filtered_dets) if filtered_dets else np.empty((0, 4))

        if len(det_result) == 0:
            return np.empty((0, 133, 2)), np.empty((0, 133))

        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores


