import os, sys, glob, subprocess, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, 'src')
import data_utils


def config_plt():
    plt.box(False)
    plt.axis('off')


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--image_dirpath',           type=str, required=True)
parser.add_argument('--sparse_depth_dirpath',    type=str, required=True)
parser.add_argument('--output_depth_dirpath',    type=str, required=True)
parser.add_argument('--ground_truth_dirpath',    type=str, default='')
parser.add_argument('--visualization_dirpath',   type=str, required=True)
parser.add_argument('--image_ext',               type=str, default='.png')
parser.add_argument('--depth_ext',               type=str, default='.png')

# Visualization
parser.add_argument('--cmap',                 type=str, default='gist_stern')
parser.add_argument('--vmin',                 type=float, default=0.10)
parser.add_argument('--vmax',                 type=float, default=100.0)

args = parser.parse_args()


if not os.path.exists(args.visualization_dirpath):
    os.mkdir(args.visualization_dirpath)

'''
Fetch file paths from input directories
'''
image_paths = \
    sorted(glob.glob(os.path.join(args.image_dirpath, '*' + args.image_ext)))
sparse_depth_paths = \
    sorted(glob.glob(os.path.join(args.sparse_depth_dirpath, '*' + args.depth_ext)))
output_depth_paths = \
    sorted(glob.glob(os.path.join(args.output_depth_dirpath, '*' + args.depth_ext)))

n_sample = len(image_paths)

assert n_sample == len(sparse_depth_paths)
assert n_sample == len(output_depth_paths)

ground_truth_available = True if args.ground_truth_dirpath != '' else False

if ground_truth_available:
    ground_truth_paths = \
        sorted(glob.glob(os.path.join(args.ground_truth_dirpath, '*' + args.depth_ext)))

    assert n_sample == len(ground_truth_paths)

'''
Process image, sparse depth and output depth (and groundtruth)
'''
for idx in range(n_sample):

    sys.stdout.write(
        'Processing {}/{} samples...\r'.format(idx + 1, n_sample))
    sys.stdout.flush()

    image_path = image_paths[idx]
    sparse_depth_path = sparse_depth_paths[idx]
    output_depth_path = output_depth_paths[idx]

    # Set up output path
    filename = os.path.basename(image_path)
    visualization_path = os.path.join(args.visualization_dirpath, filename)

    # Load image, sparse depth and output depth (and groundtruth)
    image = Image.open(image_paths[idx]).convert('RGB')
    image = np.asarray(image, dtype=np.uint8)
    sparse_depth = data_utils.load_depth(sparse_depth_path)
    output_depth = data_utils.load_depth(output_depth_path)

    n_row = 3

    if ground_truth_available:
        ground_truth = data_utils.load_depth(args.ground_truth_dirpath)
        n_row = 4

    # Create figure and grid
    plt.figure(figsize=(75, 25), dpi=40, facecolor='w', edgecolor='k')

    gs = gridspec.GridSpec(n_row, 1, wspace=0.0, hspace=0.0)

    # Plot image, sparse depth, output depth
    ax = plt.subplot(gs[0, 0])
    config_plt()
    ax.imshow(image)

    ax = plt.subplot(gs[1, 0])
    config_plt()
    ax.imshow(sparse_depth, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)

    ax = plt.subplot(gs[2, 0])
    config_plt()
    ax.imshow(output_depth, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)

    # Plot groundtruth if available
    if ground_truth_available:
        ax = plt.subplot(gs[0, 3])
        config_plt()
        ax.imshow(ground_truth, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)

    plt.savefig(visualization_path)
    plt.close()
    subprocess.call(["convert", "-trim", visualization_path, visualization_path])
