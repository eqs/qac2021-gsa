# -*- coding: utf-8 -*-
import os
from multiprocessing.pool import ThreadPool
from collections import deque

from tqdm.auto import tqdm
import click

import pandas as pd
import cv2

from detectors import SIFTObjectDetector, Compose


def process_frame(frame_idx, fps, frame, detector):
    score = detector(frame)
    return frame_idx, fps, score


def validate_arguments(template_img_path,
                       template_mask_path,
                       target_mask_path):

    if len(template_img_path) != len(template_mask_path):
        raise ValueError('テンプレート画像とマスクの数が一致しません')

    if len(template_img_path) == 0 or len(template_mask_path) == 0:
        raise ValueError('テンプレート画像を最低ひとつは指定してください')

    if (len(target_mask_path) > 1
            and len(template_img_path) != len(target_mask_path)):
        raise ValueError(
            'ターゲットのマスク画像が複数あるときは'
            'テンプレート画像と数が一致する必要があります'
        )


@click.command()
@click.argument('video_path')
@click.option('--template_img_path', '-i', multiple=True)
@click.option('--template_mask_path', '-m', multiple=True)
@click.option('--target_mask_path', '-t', multiple=True)
@click.option('--frame_interval', '-f', default=1)
@click.option('--ncpus', '-n', default=-1)
def main(video_path, template_img_path, template_mask_path, target_mask_path,
         frame_interval, ncpus):

    if ncpus <= 0:
        nthreads = cv2.getNumberOfCPUs()
    else:
        nthreads = ncpus

    validate_arguments(template_img_path, template_mask_path, target_mask_path)

    detectors = []

    if len(target_mask_path) <= 1:
        # ターゲット画像が単数か無いときは
        # 最初に一回だけターゲットのマスクを読み込む（無いときはスルー）
        if len(target_mask_path) == 1:
            target_mask_img = cv2.imread(
                target_mask_path[0], cv2.IMREAD_GRAYSCALE
            )
            assert target_mask_img is not None
        else:
            target_mask_img = None

        for tmpl, tmpl_mask in zip(template_img_path, template_mask_path):
            tmpl_img = cv2.imread(tmpl, cv2.IMREAD_GRAYSCALE)
            tmpl_mask_img = cv2.imread(tmpl_mask, cv2.IMREAD_GRAYSCALE)

            assert tmpl_img is not None
            assert tmpl_mask_img is not None

            if target_mask_img is None:
                detector = SIFTObjectDetector(tmpl_img, tmpl_mask_img)
            else:
                detector = SIFTObjectDetector(
                    tmpl_img, tmpl_mask_img, target_mask_img.copy()
                )
            detectors.append(detector)

    else:
        # ターゲット画像が複数のときはテンプレートと一緒に読み込みながら
        # 検出器を作る
        for tmpl, tmpl_mask, target_mask in zip(template_img_path,
                                                template_mask_path,
                                                target_mask_path):
            tmpl_img = cv2.imread(tmpl, cv2.IMREAD_GRAYSCALE)
            tmpl_mask_img = cv2.imread(tmpl_mask, cv2.IMREAD_GRAYSCALE)
            target_mask_img = cv2.imread(target_mask, cv2.IMREAD_GRAYSCALE)

            assert tmpl_img is not None
            assert tmpl_mask_img is not None
            assert target_mask_img is not None

            detector = SIFTObjectDetector(
                tmpl_img, tmpl_mask_img, target_mask_img
            )
            detectors.append(detector)

    composed_detector = Compose(detectors)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    pool = ThreadPool(processes=nthreads)
    pending = deque()

    frame_idx = 0
    pbar = tqdm(total=frame_count // frame_interval)

    results = []

    try:
        while True:
            while len(pending) > 0 and pending[0].ready():
                ret = pending.popleft().get()
                results.append(ret)
                pbar.update()

            if len(pending) < nthreads:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_idx % frame_interval == 0:
                    task = pool.apply_async(
                        process_frame,
                        (frame_idx, fps, frame.copy(), composed_detector)
                    )
                    pending.append(task)

                frame_idx = frame_idx + 1
    finally:
        pbar.close()
        cap.release()

    out_name = os.path.splitext(video_path)[0] + '.csv'
    out_table = pd.DataFrame(results, columns=['t', 'fps', 'score'])
    out_table.to_csv(out_name, index=False)


if __name__ == '__main__':
    main()
