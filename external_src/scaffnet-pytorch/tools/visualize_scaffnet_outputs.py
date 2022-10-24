import os, sys, glob, argparse
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
sys.path.insert(0, 'src')
import data_utils


EPSILON = 1e-8


parser = argparse.ArgumentParser()

parser.add_argument('--model_output_path', type=str, required=True)
parser.add_argument('--visual_output_path', type=str, required=True)


args = parser.parse_args()

colormap_depth = 'viridis'
colormap_error = 'magma'
cm_depth = plt.cm.get_cmap(colormap_depth)
cm_error = plt.cm.get_cmap(colormap_error)

output_depth_path = os.path.join(args.model_output_path, 'output_depth')
sparse_depth_path = os.path.join(args.model_output_path, 'sparse_depth')
ground_truth_path = os.path.join(args.model_output_path, 'ground_truth')

if not os.path.exists(args.visual_output_path):
    os.makedirs(args.visual_output_path)

output_depth_paths = sorted(glob.glob(os.path.join(output_depth_path, '*.png')))
sparse_depth_paths = sorted(glob.glob(os.path.join(sparse_depth_path, '*.png')))
ground_truth_paths = sorted(glob.glob(os.path.join(ground_truth_path, '*.png')))

n_sample = len(output_depth_paths)

# Sanity checks
assert(len(sparse_depth_paths) == n_sample)
assert(len(ground_truth_paths) == n_sample)

# Generate visualization
for idx in range(n_sample):
    _, filename = os.path.split(output_depth_paths[idx])

    # Load output, sparse and groundtruth depth
    output_depth = data_utils.load_depth(output_depth_paths[idx])
    sparse_depth = data_utils.load_depth(sparse_depth_paths[idx])
    ground_truth = data_utils.load_depth(ground_truth_paths[idx])

    # Visualize output, sparse and groundtruth depth
    validity_map = np.where(sparse_depth > 0.0, 1.0, 0.0)
    sparse_depth_error = np.abs(sparse_depth - output_depth) / (sparse_depth + EPSILON)
    sparse_depth_error = validity_map * sparse_depth_error
    sparse_depth_error = 255 * cm_error(sparse_depth_error / 0.10)[..., 0:3]

    validity_map = np.where(ground_truth > 0.0, 1.0, 0.0)
    ground_truth_error = np.abs(ground_truth - output_depth) / (ground_truth + EPSILON)
    ground_truth_error = validity_map * ground_truth_error
    ground_truth_error = 255 * cm_error(ground_truth_error / 0.10)[..., 0:3]

    output_depth = 255 * cm_depth(output_depth / 90.0)[..., 0:3]
    output_depth = output_depth.astype(np.uint8)

    sparse_depth = 255 * cm_depth(sparse_depth / 90.0)[..., 0:3]
    sparse_depth = sparse_depth.astype(np.uint8)

    ground_truth = 255 * cm_depth(ground_truth / 90.0)[..., 0:3]
    ground_truth = ground_truth.astype(np.uint8)

    output_depth_visual = np.concatenate([output_depth, np.ones_like(output_depth)], axis=1)
    sparse_depth_visual = np.concatenate([sparse_depth, sparse_depth_error], axis=1)
    ground_truth_visual = np.concatenate([ground_truth, ground_truth_error], axis=1)

    # Create visualization
    visual = np.concatenate([
        output_depth_visual.astype(np.uint8),
        sparse_depth_visual.astype(np.uint8),
        ground_truth_visual.astype(np.uint8)], axis=0)

    Image.fromarray(visual).save(os.path.join(args.visual_output_path, filename))
