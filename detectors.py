# -*- coding: utf-8 -*-
from typing import List
import numpy as np
import cv2


class BaseDetector(object):
    def __call__(self, frame: np.array) -> float:
        raise NotImplementedError()


class Compose(BaseDetector):
    def __init__(self,
                 detectors: List[BaseDetector],
                 return_max: bool = True):
        self.detectors = detectors
        self.return_max = return_max

    def __call__(self, frame):
        scores = []
        for detector in self.detectors:
            score = detector(frame)
            scores.append(score)
        return np.max(scores) if self.return_max else np.min(scores)


class SIFTObjectDetector(BaseDetector):
    def __init__(self, template_img: np.ndarray, template_mask: np.ndarray,
                 target_mask: np.ndarray = None, testing_ratio: float = 0.75):

        self.template_img = template_img
        self.template_mask = template_mask

        if target_mask is None:
            # ターゲットのマスクが指定されてない場合はテンプレートのものを流用
            self.target_mask = template_mask.copy()
        else:
            self.target_mask = target_mask
        self.testing_ratio = testing_ratio

        # SIFT準備
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

        # テンプレートにおけるSIFT記述子を先に計算しておく
        self.template_kps, self.template_des = self.sift.detectAndCompute(
            self.template_img,
            self.template_mask.astype(np.uint8)
        )

    def __call__(self, target_img):

        target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        # ターゲットにおけるSIFT記述子を計算
        target_kps, target_des = self.sift.detectAndCompute(
            target_img_gray,
            self.target_mask.astype(np.uint8)
        )

        # テンプレートのキーポイントのうち，ターゲットとマッチした割合を返す
        matches = self.matcher.knnMatch(target_des, self.template_des, k=2)
        good_matches = [m for m, n in matches
                        if m.distance < self.testing_ratio*n.distance]
        return len(good_matches) / len(self.template_kps)
