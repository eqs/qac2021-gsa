# -*- coding: utf-8 -*-
import click
import pandas as pd


@click.command()
@click.argument('video_path')
@click.argument('segments_path')
def main(video_path, segments_path):
    data = pd.read_csv(segments_path)
    for k, row in data.iterrows():
        start, end = row
        # print(f'ffmpeg -y -i {video_path}'
        #       f' -ss {start} -to {end} -c copy outputs2/output{k}.mp4')
        print(f'ffmpeg -y'
              f' -i {video_path} -ss {start} -to {end}'
              f' -vcodec libx264 -pix_fmt yuv420p outputs-sp2/output{k}.mp4')


if __name__ == '__main__':
    main()
