# -*- coding: utf-8 -*-
import os
import click

import numpy as np
from matplotlib import pyplot as plt
import cv2


@click.group()
def cmd():
    pass


@cmd.command()
@click.argument('filename_img')
@click.option('--outname', '-o', default='')
def create(filename_img, outname):
    img = cv2.imread(filename_img)
    if img is None:
        raise FileNotFoundError(filename_img)
    rois = cv2.selectROIs('select', img)

    if len(rois) == 0:
        return

    # 選択したROIを基にマスク画像を作る
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for roi in rois:
        x, y, w, h = roi
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=-1)

    # マスクを保存
    if len(outname) == 0:
        root, ext = os.path.splitext(filename_img)
        outname = root + '.mask' + ext
    cv2.imwrite(outname, mask)


@cmd.command()
@click.argument('filename_img')
@click.argument('filename_mask')
def show(filename_img, filename_mask):

    img = cv2.imread(filename_img)
    if img is None:
        raise FileNotFoundError(filename_img)

    mask = cv2.imread(filename_mask)
    if mask is None:
        raise FileNotFoundError(filename_mask)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(mask, alpha=0.5, cmap='bwr')
    ax.set_axis_off()
    fig.tight_layout()
    plt.show()


def main():
    cmd()


if __name__ == '__main__':
    main()
