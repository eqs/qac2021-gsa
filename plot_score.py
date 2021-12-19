# -*- coding: utf-8 -*-
import click
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (16.0, 9.0)
plt.rcParams['font.size'] = 18


@click.command()
@click.argument('stats_path')
def main(stats_path):
    data = pd.read_csv(stats_path)
    data['score'].plot()
    plt.xlabel('frame')
    plt.ylabel('score')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
