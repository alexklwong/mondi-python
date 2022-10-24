'''
Authors:
Tian Yu Liu <tianyu@cs.ucla.edu>
Parth Agrawal <parthagrawal24@ucla.edu>
Allison Chen <allisonchen2@ucla.edu>
Alex Wong <alex.wong@yale.edu>

If you use this code, please cite the following paper:
T.Y. Liu, P. Agrawal, A. Chen, B.W. Hong, and A. Wong. Monitored Distillation for Positive Congruent Depth Completion.
https://arxiv.org/abs/2203.16034

@inproceedings{liu2022monitored,
  title={Monitored distillation for positive congruent depth completion},
  author={Liu, Tian Yu and Agrawal, Parth and Chen, Allison and Hong, Byung-Woo and Wong, Alex},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
'''

import os, argparse, time
import numpy as np
import torch
import data_utils, datasets, eval_utils
from log_utils import log
from external_model import ExternalModel
from PIL import Image


def run(model, dataloader, output_paths=None, verbose=False):
    '''
    Runs an external depth completion model
    if output paths are provided, then will save outputs to a predetermined list of paths

    Arg(s):
        model : ExternalModel
            external depth completion model instance
        dataloader : torch.utils.data.DataLoader
            dataloader that outputs an image and a range map
        output_paths : list[str]
            list of paths to store output depth
    Returns:
        list[numpy[float32]] : list of depth maps if output paths is None else no return value
    '''

    output_depths = []
    images = []
    sparse_depths = []

    n_sample = len(dataloader)

    if output_paths is not None:
        assert len(output_paths) == n_sample

    for idx, inputs in enumerate(dataloader):

        # Move inputs to device
        inputs = [
            in_.to(model.device) for in_ in inputs
        ]

        image, sparse_depth, intrinsics = inputs

        n_height, n_width = image.shape[-2:]

        with torch.no_grad():
            # Forward through network
            output_depth = model.forward(
                image=image,
                sparse_depth=sparse_depth,
                intrinsics=intrinsics)

            assert output_depth.shape[-2] == n_height
            assert output_depth.shape[-1] == n_width

        # Convert to numpy (if not converted already)
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        if verbose:
            print('Processed {}/{} samples'.format(idx + 1, n_sample), end='\r')

        # Return output depths as a list if we do not store them
        if output_paths is None:
            output_depths.append(output_depth)
            images.append(np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0)))
            sparse_depths.append(np.squeeze(sparse_depth.cpu().numpy()))
        else:
            data_utils.save_depth(output_depth, output_paths[idx])

    if output_paths is None:
        return images, sparse_depths, output_depths


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Input file paths
    parser.add_argument('--image_path',
        type=str, required=True, help='Path to list of image paths')
    parser.add_argument('--sparse_depth_path',
        type=str, required=True, help='Path to list of sparse depth paths')
    parser.add_argument('--intrinsics_path',
        type=str, required=True, help='Path to list of intrinsics paths')
    parser.add_argument('--ground_truth_path',
        type=str, required=True, help='Path to list of ground truth depth paths')

    # External model settings
    parser.add_argument('--model_name',
        type=str, required=True, help='External model to run')
    parser.add_argument('--restore_path',
        type=str, required=True, help='Path to restore depth model from checkpoint, if pass through model, then none')
    parser.add_argument('--min_predict_depth',
        type=float, default=1.5, help='Minimum value of depth to predict')
    parser.add_argument('--max_predict_depth',
        type=float, default=100, help='Maximum value of depth to predict')

    # Evaluation settings
    parser.add_argument('--load_image_triplets',
        action='store_true')
    parser.add_argument('--min_evaluate_depth',
        type=float, default=0.0, help='Minimum value of depth to evaluate')
    parser.add_argument('--max_evaluate_depth',
        type=float, default=100, help='Maximum value of depth to evaluate')
    parser.add_argument('--do_median_scale_depth',
        default=False, action='store_true', help='If set, then scale outputs by medians of ground truth and output')

    # Checkpoint settings
    parser.add_argument('--output_path',
        type=str, default='', help='Path to log results')
    parser.add_argument('--save_outputs',
        action='store_true', help='If set then strore input sand outputs into output_path')

    # Hardware settings
    parser.add_argument('--device',
        type=str, default='cuda', help='Device to use: gpu, cpu, cuda')

    args = parser.parse_args()

    '''
    Assert arguments
    '''
    args.device = args.device.lower()

    if args.device not in ['gpu', 'cpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    device = torch.device(args.device)

    # Set up logging path
    log_path = os.path.join(args.output_path, 'results.txt')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    '''
    Read input paths and assert paths
    '''
    image_paths = data_utils.read_paths(args.image_path)
    sparse_depth_paths = data_utils.read_paths(args.sparse_depth_path)
    intrinsics_paths = data_utils.read_paths(args.intrinsics_path)
    ground_truth_paths = data_utils.read_paths(args.ground_truth_path)

    n_sample = len(image_paths)

    input_paths = [
        image_paths,
        sparse_depth_paths,
        intrinsics_paths,
        ground_truth_paths
    ]

    for paths in input_paths:
        assert n_sample == len(paths)

    ground_truths = []
    for path in ground_truth_paths:
        ground_truth, validity_map = data_utils.load_depth_with_validity_map(path)
        ground_truths.append(np.stack([ground_truth, validity_map], axis=-1))

    dataloader = torch.utils.data.DataLoader(
        datasets.DepthCompletionInferenceDataset(
            image_paths=image_paths,
            sparse_depth_paths=sparse_depth_paths,
            intrinsics_paths=intrinsics_paths,
            load_image_triplets=args.load_image_triplets),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    # Build external depth completion network
    if args.model_name == 'nlspn' and \
        (os.path.basename(args.restore_path) == 'NLSPN_KITTI_DC.pt' or os.path.basename(args.restore_path) == 'NLSPN_NYUV2.pt'):
        model = ExternalModel(
            model_name=args.model_name,
            min_predict_depth=args.min_predict_depth,
            max_predict_depth=args.max_predict_depth,
            device=device,
            use_pretrained=True)
    else:
        model = ExternalModel(
            model_name=args.model_name,
            min_predict_depth=args.min_predict_depth,
            max_predict_depth=args.max_predict_depth,
            device=device)

    # Restore model and set to evaluation mode
    try:
        model.restore_model(args.restore_path)
    except Exception:
        model.data_parallel()
        model.restore_model(args.restore_path)

    model.eval()

    try:
        model_parameters = model.parameters()
        n_parameter = sum(p.numel() for p in model_parameters)
    except Exception:
        # To account for pass through model
        n_parameter = 0

    log('Input paths:', log_path)
    input_paths = [
        args.image_path,
        args.sparse_depth_path,
        args.intrinsics_path,
        args.ground_truth_path
    ]
    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    log('External model settings:', log_path)
    log('model_name={}'.format(args.model_name),
        log_path)
    log('restore_path={}'.format(args.restore_path),
        log_path)
    log('min_predict_depth={}  max_predict_depth={}'.format(
        args.min_predict_depth, args.max_predict_depth),
        log_path)
    log('', log_path)

    log('Evaluation settings:', log_path)
    log('load_image_triplets={}'.format(args.load_image_triplets),
        log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        args.min_evaluate_depth, args.max_evaluate_depth),
        log_path)
    log('do_median_scale_depth={}'.format(args.do_median_scale_depth), log_path)
    log('', log_path)

    '''
    Run model
    '''
    time_elapse = 0.0
    time_start = time.time()

    images, sparse_depths, output_depths = run(model, dataloader, verbose=True)

    if args.save_outputs:
        image_dirpath = os.path.join(args.output_path, 'image')
        output_depth_dirpath = os.path.join(args.output_path, 'output_depth')
        sparse_depth_dirpath = os.path.join(args.output_path, 'sparse_depth')
        ground_truth_dirpath = os.path.join(args.output_path, 'ground_truth')
        dirpaths = [
            image_dirpath,
            output_depth_dirpath,
            sparse_depth_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        outputs = zip(images, output_depths, sparse_depths, ground_truths)

        for idx, (image, output_depth, sparse_depth, ground_truth) in enumerate(outputs):
            filename = os.path.basename(image_paths[idx])

            image_path = os.path.join(image_dirpath, filename)
            image = (255 * image).astype(np.uint8)
            Image.fromarray(image).save(image_path)

            output_depth_path = os.path.join(output_depth_dirpath, filename)
            data_utils.save_depth(output_depth, output_depth_path)

            sparse_depth_path = os.path.join(sparse_depth_dirpath, filename)
            data_utils.save_depth(sparse_depth, sparse_depth_path)

            ground_truth_path = os.path.join(ground_truth_dirpath, filename)
            data_utils.save_depth(ground_truth[..., 0], ground_truth_path)

    time_elapse = time_elapse + (time.time() - time_start)

    # Set up metrics in case ground truth
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)
    are = np.zeros(n_sample)
    sre = np.zeros(n_sample)

    for idx, (output_depth, ground_truth) in enumerate(zip(output_depths, ground_truths)):

        # Get valid regions in ground truth
        ground_truth = np.squeeze(ground_truth)

        validity_map = ground_truth[:, :, 1]
        ground_truth = ground_truth[:, :, 0]

        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > args.min_evaluate_depth,
            ground_truth < args.max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

       # Select valid regions in ground truth and output depth
        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        if args.do_median_scale_depth:
            output_depth = output_depth * np.median(ground_truth) / np.median(output_depth)

        # Clamp output depth
        # output_depth = np.clip(output_depth, a_min=None, a_max=args.max_evaluate_depth)

        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)
        are[idx] = eval_utils.abs_rel_err(1000.0 * output_depth, 1000.0 * ground_truth)
        sre[idx] = eval_utils.sq_rel_err(1000.0 * output_depth, 1000.0 * ground_truth)

    # Compute total time elapse in ms
    time_elapse = time_elapse * 1000.0

    mae_mean   = np.mean(mae)
    rmse_mean  = np.mean(rmse)
    imae_mean  = np.mean(imae)
    irmse_mean = np.mean(irmse)
    are_mean = np.mean(are)
    sre_mean = np.mean(sre)

    mae_std = np.std(mae)
    rmse_std = np.std(rmse)
    imae_std = np.std(imae)
    irmse_std = np.std(irmse)
    are_std = np.std(are)
    sre_std = np.std(sre)

    # Log evaluation results
    log('Evaluation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'MAE', 'RMSE', 'iMAE', 'iRMSE', 'ARE (%)', 'SRE (%)'),
        log_path)
    log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        mae_mean, rmse_mean, imae_mean, irmse_mean, are_mean * 100, sre_mean * 100),
        log_path)

    log('{:>8}  {:>8}  {:>8}  {:>8} {:>8}  {:>8}'.format(
        '+/-', '+/-', '+/-', '+/-', '+/-', '+/-'),
        log_path)
    log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        mae_std, rmse_std, imae_std, irmse_std, are_std * 100, sre_std * 100),
        log_path)

    log('Total time: {:.2f} ms  Average time per sample: {:.2f} ms'.format(
        time_elapse, time_elapse / float(n_sample)))
