# Qiita OpenCV Advent Calendar 2021

## いるもの

* Python 3.7+
* opencv-python 4.5+
* Numpy
* Pandas
* Click

## 使い方

マスク画像作る

```
$ python create_mask.py show
Usage: create_mask.py show [OPTIONS] FILENAME_IMG FILENAME_MASK
```

動画の解析

```
$ python analyze_video.py --help
Usage: analyze_video.py [OPTIONS] VIDEO_PATH

Options:
  -i, --template_img_path TEXT
  -m, --template_mask_path TEXT
  -t, --target_mask_path TEXT
  -f, --frame_interval INTEGER
  -n, --ncpus INTEGER
  --help                         Show this message and exit.
```

イベント検出（`STATS_PATH` は`analyze_video`が吐いたcsvファイル）

```
python detect_event.py --help
Usage: detect_event.py [OPTIONS] VIDEO_PATH STATS_PATH

Options:
  -ml, --min_event_length INTEGER
  -mi, --min_interval INTEGER
  -st, --score_threshold FLOAT
  -v, --verbose
  --help                          Show this message and exit.
```

FFmpegコマンド出力（`SEGMENTS_PATH` は`detect_event`が吐いたcsvファイル）

```
> python to_ffmpeg_command.py
Usage: to_ffmpeg_command.py [OPTIONS] VIDEO_PATH SEGMENTS_PATH
```
