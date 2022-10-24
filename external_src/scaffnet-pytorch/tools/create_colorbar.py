import os, subprocess, argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def config_plt():
    plt.box(False)
    plt.axis('off')


parser = argparse.ArgumentParser()

parser.add_argument('--output_path',          type=str, required=True)
parser.add_argument('--cmap',                 type=str, default='gist_stern')
parser.add_argument('--vmin',                 type=float, default=0.10)
parser.add_argument('--vmax',                 type=float, default=100.0)
parser.add_argument('--length',               type=int, default=1000)
parser.add_argument('--width',                type=int, default=50)

args = parser.parse_args()


# Create color bar array
color_range = np.linspace(start=args.vmax, stop=args.vmin, num=args.length)
color_range = np.expand_dims(color_range, axis=-1)
color_bar = np.tile(color_range, reps=[1, args.width])

# Create figure to save color bar
fig = plt.figure(facecolor='w', edgecolor='k')

ax = fig.add_subplot(1, 1, 1)
ax.imshow(color_bar, cmap=args.cmap)
config_plt()

plt.savefig(args.output_path)
plt.close()
subprocess.call(["convert", "-trim", args.output_path, args.output_path])
