# -*- coding: utf-8 -*-
from enum import Enum
import os
import logging

import click
import pandas as pd
import cv2


class State(Enum):
    NO_EVENT = 0
    EVENT_CANDIDATE = 1
    ON_EVENT = 2
    END_PREP = 3


def segment_sequence(timing_seq, frame_interval=1,
                     min_event_length=10, min_interval=120):
    """イベントの系列から動画の開始・終了フレームを生成する"""

    state = State.NO_EVENT
    prev_state = State.NO_EVENT

    start_time = 0
    end_time = 0
    zero_count = 0
    one_count = 0

    # min_event_length = max(min_event_length // frame_interval, 1)
    # min_interval = max(min_interval // frame_interval, 1)

    segments = []

    for t, s in enumerate(timing_seq):
        if s:
            one_count = one_count + frame_interval
        else:
            zero_count = zero_count + frame_interval

        if state == State.NO_EVENT:
            if s:
                state = State.EVENT_CANDIDATE
                start_time = t
        elif state == State.EVENT_CANDIDATE:
            if one_count > min_event_length:
                state = State.ON_EVENT
                logging.info(f'Start: {start_time}')
            elif zero_count > min_interval:
                state = State.NO_EVENT
        elif state == State.ON_EVENT:
            if not s:
                state = State.END_PREP
                end_time = t
        elif state == State.END_PREP:
            if zero_count > min_interval:
                state = State.NO_EVENT
                logging.info(f'End: {end_time}')
                segments.append((start_time * frame_interval,
                                 end_time * frame_interval))
            elif s:
                state = State.ON_EVENT
        else:
            raise RuntimeError

        # If state is changed
        if state != prev_state:
            one_count = 0
            zero_count = 0
        prev_state = state

    return segments


def get_screenshot(video_path, idxs):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise RuntimeError

    screenshots = []
    for idx in idxs:
        if video.set(cv2.CAP_PROP_POS_FRAMES, idx):
            ok, screenshot = video.read()
            screenshots.append(screenshot)
    return screenshots


def sec_to_hhmmss(sec):
    h, m, s = sec // 3600, (sec // 60) % 60, sec % 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


@click.command()
@click.argument('video_path')
@click.argument('stats_path')
@click.option('--min_event_length', '-ml', default=10)
@click.option('--min_interval', '-mi', default=120)
@click.option('--score_threshold', '-st', default=0.14)
@click.option('--verbose', '-v', is_flag=True)
def main(video_path, stats_path,
         min_event_length, min_interval, score_threshold, verbose):
    data = pd.read_csv(stats_path)

    timing_seq = data['score'] > score_threshold
    fps = data['fps'].iloc[0]
    frame_interval = data['t'].iloc[1] - data['t'].iloc[0]

    segments = segment_sequence(
        timing_seq, frame_interval,
        min_event_length=min_event_length, min_interval=min_interval
    )
    screenshots = get_screenshot(
        video_path,
        list(map(lambda x: x[0], segments))
    )

    if verbose:
        for k, screenshot in enumerate(screenshots):
            cv2.imwrite(f'{k:03d}.png', screenshot)

    def to_sec(segment_frame):
        s, e = segment_frame
        return (s / fps, e / fps)

    def add_offset(segment_sec, start_offset_sec, end_offset_sec):
        s, e = segment_sec
        return (max(s - start_offset_sec, 0.0), e + end_offset_sec)

    segments_sec = list(map(to_sec, segments))
    segments_sec = [add_offset(seg, 8.0, 1.0) for seg in segments_sec]

    print(list(
        map(
            sec_to_hhmmss,
            map(lambda x: int(x[0]), segments_sec)
        )
    ))
    print(segments_sec)

    out_name = os.path.splitext(stats_path)[0] + '-segments.csv'
    segments_tb = pd.DataFrame(segments_sec, columns=['start_sec', 'end_sec'])
    segments_tb.to_csv(out_name, index=False)


if __name__ == '__main__':
    main()
