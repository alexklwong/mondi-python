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

import os, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils, eval_utils
from log_utils import log
from mondi_model import MonitoredDistillationModel
from posenet_model import PoseNetModel
from transforms import Transforms
from net_utils import OutlierRemoval
from PIL import Image


def train(train_image0_path,
          train_image1_path,
          train_sparse_depth0_path,
          train_sparse_depth1_path,
          train_teacher_output0_paths,
          train_teacher_output1_paths,
          train_intrinsics0_path,
          train_intrinsics1_path,
          train_focal_length_baseline0_path,
          train_focal_length_baseline1_path,
          train_ground_truth0_path,
          train_ground_truth1_path,
          val_image_path,
          val_sparse_depth_path,
          val_intrinsics_path,
          val_ground_truth_path,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          # Input settings
          input_types,
          input_channels_image,
          input_channels_depth,
          normalized_image_range,
          outlier_removal_kernel_size,
          outlier_removal_threshold,
          # Sparse to dense pool settings
          min_pool_sizes_sparse_to_dense_pool,
          max_pool_sizes_sparse_to_dense_pool,
          n_convolution_sparse_to_dense_pool,
          n_filter_sparse_to_dense_pool,
          # Depth network settings
          encoder_type,
          n_filters_encoder_image,
          n_filters_encoder_depth,
          n_convolutions_encoder,
          resolutions_backprojection,
          resolutions_depthwise_separable_encoder,
          decoder_type,
          n_filters_decoder,
          n_resolution_decoder,
          resolutions_depthwise_separable_decoder,
          min_predict_depth,
          max_predict_depth,
          # Weight settings
          weight_initializer,
          activation_func,
          # Training settings
          supervision_types,
          learning_rates_depth,
          learning_schedule_depth,
          learning_rates_pose,
          learning_schedule_pose,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_swap,
          augmentation_random_crop_type,
          augmentation_random_crop_to_shape,
          augmentation_random_remove_points,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          # Loss function settings
          w_stereo,
          w_monocular,
          w_color,
          w_structure,
          w_sparse_depth,
          w_ensemble_depth,
          w_ensemble_temperature,
          w_sparse_select_ensemble,
          sparse_select_dilate_kernel_size,
          w_smoothness,
          w_weight_decay_depth,
          w_weight_decay_pose,
          loss_func_ensemble_depth,
          epoch_pose_for_ensemble,
          ensemble_method,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path,
          n_checkpoint,
          n_summary,
          n_summary_display,
          validation_start_step,
          depth_model_restore_path,
          pose_model_restore_path,
          # Hardware settings
          device='cuda',
          n_thread=6):

    # Select device to run on
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    # Set up checkpoint and event paths
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, 'selection_stats')):
        os.makedirs(os.path.join(checkpoint_path, 'selection_stats'))

    depth_model_checkpoint_path = os.path.join(checkpoint_path, 'depth_model-{}.pth')
    pose_model_checkpoint_path = os.path.join(checkpoint_path, 'pose_model-{}.pth')
    selection_stats_path = os.path.join(checkpoint_path, 'selection_stats/selection_stats-{}.npy')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty
    }

    '''
    Load input paths and set up dataloaders
    '''
    # Check if stereo stream is available
    stereo_available = \
        train_image1_path is not None and \
        train_sparse_depth1_path is not None and \
        train_teacher_output1_paths is not None and \
        isinstance(train_teacher_output1_paths, list) and \
        len(train_teacher_output1_paths) > 0 and \
        train_intrinsics1_path is not None and \
        train_focal_length_baseline0_path is not None and \
        train_focal_length_baseline1_path is not None

    # Read paths for training
    train_image0_paths = data_utils.read_paths(train_image0_path)

    train_sparse_depth0_paths = data_utils.read_paths(train_sparse_depth0_path)

    train_ground_truth0_paths = data_utils.read_paths(train_ground_truth0_path)

    train_ensemble_teacher_output0_paths = [
        data_utils.read_paths(path) for path in train_teacher_output0_paths
    ]

    train_intrinsics0_paths = data_utils.read_paths(train_intrinsics0_path)

    n_train_sample = len(train_image0_paths)

    # Check if stereo stream is available
    if stereo_available:
        # If stereo stream is available, load paths
        train_image1_paths = data_utils.read_paths(train_image1_path)

        train_sparse_depth1_paths = data_utils.read_paths(train_sparse_depth1_path)

        train_ground_truth1_paths = data_utils.read_paths(train_ground_truth1_path)

        train_ensemble_teacher_output1_paths = [
            data_utils.read_paths(path) for path in train_teacher_output1_paths
        ]

        train_intrinsics1_paths = data_utils.read_paths(train_intrinsics1_path)

        train_focal_length_baseline0_paths = \
            data_utils.read_paths(train_focal_length_baseline0_path)

        train_focal_length_baseline1_paths = \
            data_utils.read_paths(train_focal_length_baseline1_path)
    else:
        # Populate with None
        train_image1_paths = [None] * n_train_sample

        train_sparse_depth1_paths = [None] * n_train_sample

        train_ground_truth1_paths = [None] * n_train_sample

        train_ensemble_teacher_output1_paths = [[None] * n_train_sample] * len(train_teacher_output0_paths)

        train_intrinsics1_paths = [None] * n_train_sample

        train_focal_length_baseline0_paths = [None] * n_train_sample

        train_focal_length_baseline1_paths  = [None] * n_train_sample

    # Make sure number of paths match number of training sample
    input_paths = [
        train_image0_paths,
        train_image1_paths,
        train_sparse_depth0_paths,
        train_sparse_depth1_paths,
        train_intrinsics0_paths,
        train_intrinsics1_paths,
        train_focal_length_baseline0_paths,
        train_focal_length_baseline1_paths,
        train_ground_truth0_paths,
        train_ground_truth1_paths,
    ]

    input_paths = input_paths + \
        train_ensemble_teacher_output0_paths + \
        train_ensemble_teacher_output1_paths

    for paths in input_paths:
        assert len(paths) == n_train_sample

    assert 'image' in input_types and \
        ('sparse_depth' in input_types or 'filtered_sparse_depth' in input_types) and \
        ('validity_map' in input_types or 'filtered_validity_map' in input_types)

    assert learning_schedule_depth[-1] == learning_schedule_pose[-1]

    # Set up training dataloader
    n_train_step = \
        learning_schedule_depth[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.MonitoredDistillationTrainingDataset(
            image0_paths=train_image0_paths,
            image1_paths=train_image1_paths,
            sparse_depth0_paths=train_sparse_depth0_paths,
            sparse_depth1_paths=train_sparse_depth1_paths,
            ground_truth0_paths=train_ground_truth0_paths,
            ground_truth1_paths=train_ground_truth1_paths,
            ensemble_teacher_output0_paths=train_ensemble_teacher_output0_paths,
            ensemble_teacher_output1_paths=train_ensemble_teacher_output1_paths,
            intrinsics0_paths=train_intrinsics0_paths,
            intrinsics1_paths=train_intrinsics1_paths,
            focal_length_baseline0_paths=train_focal_length_baseline0_paths,
            focal_length_baseline1_paths=train_focal_length_baseline1_paths,
            random_crop_shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type,
            random_swap=augmentation_random_swap),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_crop_to_shape=augmentation_random_crop_to_shape,
        random_remove_points=augmentation_random_remove_points,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation)

    # Load validation data if it is available
    validation_available = val_image_path is not None and \
        val_sparse_depth_path is not None and \
        val_intrinsics_path is not None and \
        val_ground_truth_path is not None

    if validation_available:
        val_image_paths = data_utils.read_paths(val_image_path)
        val_sparse_depth_paths = data_utils.read_paths(val_sparse_depth_path)
        val_intrinsics_paths = data_utils.read_paths(val_intrinsics_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        n_val_sample = len(val_image_paths)

        for paths in [val_sparse_depth_paths, val_intrinsics_paths, val_ground_truth_paths]:
            assert len(paths) == n_val_sample

        ground_truths = []
        for path in val_ground_truth_paths:
            ground_truth, validity_map = data_utils.load_depth_with_validity_map(path)
            ground_truths.append(np.stack([ground_truth, validity_map], axis=-1))

        val_dataloader = torch.utils.data.DataLoader(
            datasets.DepthCompletionInferenceDataset(
                image_paths=val_image_paths,
                sparse_depth_paths=val_sparse_depth_paths,
                intrinsics_paths=val_intrinsics_paths,
                load_image_triplets=False),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        val_transforms = Transforms(
            normalized_image_range=normalized_image_range)

    # Initialize outlier removal for sparse depth
    outlier_removal = OutlierRemoval(
        kernel_size=outlier_removal_kernel_size,
        threshold=outlier_removal_threshold)

    '''
    Set up the model
    '''
    # Build student network
    depth_model = MonitoredDistillationModel(
        min_pool_sizes_sparse_to_dense_pool=min_pool_sizes_sparse_to_dense_pool,
        max_pool_sizes_sparse_to_dense_pool=max_pool_sizes_sparse_to_dense_pool,
        n_convolution_sparse_to_dense_pool=n_convolution_sparse_to_dense_pool,
        n_filter_sparse_to_dense_pool=n_filter_sparse_to_dense_pool,
        encoder_type=encoder_type,
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        n_convolutions_encoder=n_convolutions_encoder,
        resolutions_backprojection=resolutions_backprojection,
        resolutions_depthwise_separable_encoder=resolutions_depthwise_separable_encoder,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        n_resolution_decoder=n_resolution_decoder,
        resolutions_depthwise_separable_decoder=resolutions_depthwise_separable_decoder,
        activation_func=activation_func,
        weight_initializer=weight_initializer,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    parameters_depth_model = depth_model.parameters()

    depth_model.train()

    # Bulid PoseNet (only needed for training)
    if 'monocular' in supervision_types:
        pose_model = PoseNetModel(
            encoder_type='resnet18',
            rotation_parameterization='axis',
            weight_initializer=weight_initializer,
            activation_func='relu',
            device=device)

        parameters_pose_model = pose_model.parameters()

        pose_model.train()
    else:
        parameters_pose_model = []

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    if stereo_available:
        train_input_paths = [
            train_image0_path,
            train_image1_path,
            train_sparse_depth0_path,
            train_sparse_depth1_path,
            train_ground_truth0_path,
            train_ground_truth1_path,
            train_intrinsics0_path,
            train_intrinsics1_path,
            train_focal_length_baseline0_path,
            train_focal_length_baseline1_path,
        ]

        train_input_paths = train_input_paths + \
            train_teacher_output0_paths + \
            train_teacher_output1_paths

    else:
        train_input_paths = [
            train_image0_path,
            train_sparse_depth0_path,
            train_intrinsics0_path,
            train_ground_truth0_path,
        ]
        train_input_paths = train_input_paths + \
            train_teacher_output0_paths

    for path in train_input_paths:
        log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_sparse_depth_path,
        val_intrinsics_path,
        val_ground_truth_path
    ]
    for path in val_input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        # Batch settings
        n_batch=n_batch,
        n_height=n_height,
        n_width=n_width,
        # Input settings
        input_types=input_types,
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        normalized_image_range=normalized_image_range,
        outlier_removal_kernel_size=outlier_removal_kernel_size,
        outlier_removal_threshold=outlier_removal_threshold)

    log_network_settings(
        log_path,
        # Sparse to dense pool settings
        min_pool_sizes_sparse_to_dense_pool=min_pool_sizes_sparse_to_dense_pool,
        max_pool_sizes_sparse_to_dense_pool=max_pool_sizes_sparse_to_dense_pool,
        n_convolution_sparse_to_dense_pool=n_convolution_sparse_to_dense_pool,
        n_filter_sparse_to_dense_pool=n_filter_sparse_to_dense_pool,
        # Depth network settings
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        n_convolutions_encoder=n_convolutions_encoder,
        resolutions_backprojection=resolutions_backprojection,
        resolutions_depthwise_separable_encoder=resolutions_depthwise_separable_encoder,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        n_resolution_decoder=n_resolution_decoder,
        deconv_type='up',
        resolutions_depthwise_separable_decoder=resolutions_depthwise_separable_decoder,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_depth_model=parameters_depth_model,
        parameters_pose_model=parameters_pose_model)

    log_training_settings(
        log_path,
        # Training settings
        n_batch=n_batch,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        supervision_types=supervision_types,
        learning_rates_depth=learning_rates_depth,
        learning_schedule_depth=learning_schedule_depth,
        learning_rates_pose=learning_rates_pose,
        learning_schedule_pose=learning_schedule_pose,
        # Augmentation settings
        augmentation_probabilities=augmentation_probabilities,
        augmentation_schedule=augmentation_schedule,
        augmentation_random_crop_type=augmentation_random_crop_type,
        augmentation_random_crop_to_shape=augmentation_random_crop_to_shape,
        augmentation_random_remove_points=augmentation_random_remove_points,
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_saturation=augmentation_random_saturation)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        w_stereo=w_stereo,
        w_monocular=w_monocular,
        w_color=w_color,
        w_structure=w_structure,
        w_sparse_depth=w_sparse_depth,
        w_ensemble_depth=w_ensemble_depth,
        w_ensemble_temperature=w_ensemble_temperature,
        w_sparse_select_ensemble=w_sparse_select_ensemble,
        sparse_select_dilate_kernel_size=sparse_select_dilate_kernel_size,
        w_smoothness=w_smoothness,
        w_weight_decay_depth=w_weight_decay_depth,
        w_weight_decay_pose=w_weight_decay_pose,
        loss_func_ensemble_depth=loss_func_ensemble_depth,
        epoch_pose_for_ensemble=epoch_pose_for_ensemble,
        ensemble_method=ensemble_method)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        n_checkpoint=n_checkpoint,
        summary_event_path=event_path,
        n_summary=n_summary,
        n_summary_display=n_summary_display,
        validation_start_step=validation_start_step,
        depth_model_restore_path=depth_model_restore_path,
        pose_model_restore_path=pose_model_restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos_depth = 0
    learning_schedule_pos_pose = 0

    learning_rate_depth = learning_rates_depth[0]
    learning_rate_pose = learning_rates_pose[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    optimizer_depth = torch.optim.Adam([
        {
            'params' : parameters_depth_model,
            'weight_decay' : w_weight_decay_depth
        }],
        lr=learning_rate_depth)

    optimizer_pose = torch.optim.Adam([
        {
            'params' : parameters_pose_model,
            'weight_decay' : w_weight_decay_pose
        }],
        lr=learning_rate_pose)

    # Start training
    train_step = 0

    if depth_model_restore_path is not None and depth_model_restore_path != '':
        try:
            train_step, optimizer_depth = depth_model.restore_model(
                depth_model_restore_path,
                optimizer=optimizer_depth)
        except Exception:
            print("[Depth] Failed to restore optimizer: Ignoring...")
            train_step, _ = depth_model.restore_model(
                depth_model_restore_path,
                optimizer=None)

        for g in optimizer_depth.param_groups:
            g['lr'] = learning_rate_depth

    if pose_model_restore_path is not None and pose_model_restore_path != '':
        try:
            _, optimizer_pose = pose_model.restore_model(
                pose_model_restore_path,
                optimizer=optimizer_pose)
        except Exception:
            print("[Pose] Failed to restore optimizer: Ignoring...")
            _, _ = pose_model.restore_model(
                pose_model_restore_path,
                optimizer=None)

        for g in optimizer_pose.param_groups:
            g['lr'] = learning_rate_pose

    time_start = time.time()

    n_teachers = len(train_teacher_output0_paths)

    log('Begin training...', log_path)
    for epoch in range(1, learning_schedule_depth[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule_depth[learning_schedule_pos_depth]:
            learning_schedule_pos_depth = learning_schedule_pos_depth + 1
            learning_rate_depth = learning_rates_depth[learning_schedule_pos_depth]

            # Update optimizer learning rates
            for g in optimizer_depth.param_groups:
                g['lr'] = learning_rate_depth

        if epoch > learning_schedule_pose[learning_schedule_pos_pose]:
            learning_schedule_pos_pose = learning_schedule_pos_pose + 1
            learning_rate_pose = learning_rates_pose[learning_schedule_pos_pose]

            # Update optimizer learning rates
            for g in optimizer_pose.param_groups:
                g['lr'] = learning_rate_pose

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        selection_stats = np.zeros(n_teachers)

        for inputs in train_dataloader:

            train_step = train_step + 1

            # Fetch data
            inputs = [
                in_.to(device) for in_ in inputs
            ]

            image0, \
                image1, \
                image2, \
                image3, \
                sparse_depth0, \
                ground_truth0, \
                teacher_output0, \
                intrinsics0, \
                focal_length_baseline0 = inputs

            # Validity map is where sparse depth is available
            validity_map0 = torch.where(
                sparse_depth0 > 0,
                torch.ones_like(sparse_depth0),
                sparse_depth0)

            # Remove outlier points and update sparse depth and validity map
            filtered_sparse_depth0, \
                filtered_validity_map0 = outlier_removal.remove_outliers(
                    sparse_depth=sparse_depth0,
                    validity_map=validity_map0)

            if 'sparse_depth' in input_types:
                pass
            elif 'filtered_sparse_depth' in input_types:
                sparse_depth0 = filtered_sparse_depth0
            else:
                raise ValueError('Sparse depth type is required')

            if 'validity_map' in input_types:
                pass
            elif 'filtered_validity_map' in input_types:
                validity_map0 = filtered_validity_map0
            else:
                raise ValueError('Validity map type is required')

            validity_map_ground_truth0 = torch.where(
                ground_truth0 > 0,
                torch.ones_like(ground_truth0),
                ground_truth0)

            # Do data augmentation
            [image0, image1, image2, image3], \
                [sparse_depth0, ground_truth0], \
                [validity_map0, validity_map_ground_truth0, teacher_output0], \
                [intrinsics0] = train_transforms.transform(
                    images_arr=[image0, image1, image2, image3],
                    range_maps_arr=[sparse_depth0, ground_truth0],
                    validity_maps_arr=[validity_map0, validity_map_ground_truth0, teacher_output0],
                    intrinsics_arr=[intrinsics0],
                    random_transform_probability=augmentation_probability)

            # Forward through the network
            output_depth0 = depth_model.forward(
                image=image0,
                sparse_depth=sparse_depth0,
                validity_map=validity_map0,
                intrinsics=intrinsics0)

            if 'monocular' in supervision_types:
                pose0to1 = pose_model.forward(image0, image1)
                pose0to2 = pose_model.forward(image0, image2)
            else:
                image1 = None
                image2 = None
                pose0to1 = None
                pose0to2 = None

            if 'stereo' not in supervision_types:
                image3 = None

            # Only use pose after epoch 10
            use_pose_for_ensemble = True if epoch >= epoch_pose_for_ensemble else False

            # Compute loss function
            loss, loss_info, images_info = depth_model.compute_loss(
                output_depth0=output_depth0,
                sparse_depth0=sparse_depth0,
                validity_map0=validity_map0,
                teacher_output0=teacher_output0,
                ground_truth0=ground_truth0,
                validity_map_ground_truth0=validity_map_ground_truth0,
                image0=image0,
                image1=image1,
                image2=image2,
                image3=image3,
                pose0to1=pose0to1,
                pose0to2=pose0to2,
                intrinsics0=intrinsics0,
                focal_length_baseline0=focal_length_baseline0,
                w_stereo=w_stereo,
                w_monocular=w_monocular,
                w_color=w_color,
                w_structure=w_structure,
                w_sparse_depth=w_sparse_depth,
                w_ensemble_depth=w_ensemble_depth,
                w_ensemble_temperature=w_ensemble_temperature,
                w_sparse_select_ensemble=w_sparse_select_ensemble,
                sparse_select_dilate_kernel_size=sparse_select_dilate_kernel_size,
                w_smoothness=w_smoothness,
                loss_func_ensemble_depth=loss_func_ensemble_depth,
                use_pose_for_ensemble=use_pose_for_ensemble,
                ensemble_method=ensemble_method)

            # Compute gradient and backpropagate
            optimizer_depth.zero_grad()
            optimizer_pose.zero_grad()

            loss.backward()

            optimizer_depth.step()
            optimizer_pose.step()

            # Save teacher statistics
            teacher_idxs = images_info.pop('teacher_idxs')
            if teacher_idxs is not None:
                for teacher_idx in range(n_teachers):
                    selection_stats[teacher_idx] += (teacher_idxs == teacher_idx).sum().detach().item()

            if (train_step % n_summary) == 0:

                if pose0to1 is not None:
                    image1to0 = images_info.pop('image1to0').detach().clone()
                else:
                    image1to0 = None

                if pose0to2 is not None:
                    image2to0 = images_info.pop('image2to0').detach().clone()
                else:
                    image2to0 = None

                if 'stereo' in supervision_types:
                    image3to0 = images_info.pop('image3to0').detach().clone()
                else:
                    image3to0 = None

                # Log summary
                depth_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    image0=image0,
                    image1to0=image1to0,
                    image2to0=image2to0,
                    image3to0=image3to0,
                    output_depth0=output_depth0.detach().clone(),
                    sparse_depth0=sparse_depth0.detach().clone(),
                    validity_map0=validity_map0,
                    pose0to1=pose0to1,
                    pose0to2=pose0to2,
                    scalars=loss_info,
                    n_display=min(n_batch, n_summary_display))

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                print(f"Saving to: {selection_stats_path.format(train_step)}")
                np.save(selection_stats_path.format(train_step), selection_stats)
                selection_stats = np.zeros(n_teachers)

                if train_step >= validation_start_step and validation_available:
                    # Switch to validation mode
                    depth_model.eval()

                    with torch.no_grad():
                        # Perform validation
                        best_results = validate(
                            depth_model=depth_model,
                            dataloader=val_dataloader,
                            input_types=input_types,
                            transforms=val_transforms,
                            outlier_removal=outlier_removal,
                            ground_truths=ground_truths,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_summary_display=n_summary_display,
                            log_path=log_path)

                    # Switch back to training
                    depth_model.train()

                # Save checkpoints
                depth_model.save_model(
                    depth_model_checkpoint_path.format(train_step), train_step, optimizer_depth)

                if 'monocular' in supervision_types:
                    pose_model.save_model(
                        pose_model_checkpoint_path.format(train_step), train_step, optimizer_pose)

    # Perform validation for final step
    depth_model.eval()

    with torch.no_grad():
        best_results = validate(
            depth_model=depth_model,
            dataloader=val_dataloader,
            input_types=input_types,
            transforms=val_transforms,
            outlier_removal=outlier_removal,
            ground_truths=ground_truths,
            step=train_step,
            best_results=best_results,
            min_evaluate_depth=min_evaluate_depth,
            max_evaluate_depth=max_evaluate_depth,
            device=device,
            summary_writer=val_summary_writer,
            n_summary_display=n_summary_display,
            log_path=log_path)

    # Save checkpoints
    depth_model.save_model(
        depth_model_checkpoint_path.format(train_step), train_step, optimizer_depth)

    if 'monocular' in supervision_types:
        pose_model.save_model(
            pose_model_checkpoint_path.format(train_step), train_step, optimizer_pose)

def validate(depth_model,
             dataloader,
             input_types,
             transforms,
             outlier_removal,
             ground_truths,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             summary_writer,
             n_summary_display=4,
             n_summary_display_interval=250,
             log_path=None):

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    image_summary = []
    output_depth_summary = []
    sparse_depth_summary = []
    validity_map_summary = []
    ground_truth_summary = []

    for idx, (inputs, ground_truth) in enumerate(zip(dataloader, ground_truths)):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, sparse_depth, intrinsics = inputs

        ground_truth = np.expand_dims(ground_truth, axis=0)
        ground_truth = np.transpose(ground_truth, (0, 3, 1, 2))
        ground_truth = torch.from_numpy(ground_truth).to(device)

        with torch.no_grad():
            # Validity map is where sparse depth is available
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            # Remove outlier points and update sparse depth and validity map
            filtered_sparse_depth, \
                filtered_validity_map = outlier_removal.remove_outliers(
                    sparse_depth=sparse_depth,
                    validity_map=validity_map)

            if 'sparse_depth' in input_types:
                pass
            elif 'filtered_sparse_depth' in input_types:
                sparse_depth = filtered_sparse_depth
            else:
                raise ValueError('Sparse depth type is required')

            if 'validity_map' in input_types:
                pass
            elif 'filtered_validity_map' in input_types:
                validity_map = filtered_validity_map
            else:
                raise ValueError('Validity map type is required')

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            # Forward through network
            output_depth = depth_model.forward(
                image=image,
                sparse_depth=sparse_depth,
                validity_map=validity_map,
                intrinsics=intrinsics)

        if (idx % n_summary_display_interval) == 0 and summary_writer is not None:
            image_summary.append(image)
            output_depth_summary.append(output_depth)
            sparse_depth_summary.append(sparse_depth)
            validity_map_summary.append(validity_map)
            ground_truth_summary.append(ground_truth)

        # Convert to numpy to validate
        output_depth = np.squeeze(output_depth.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        validity_map = ground_truth[1, :, :]
        ground_truth = ground_truth[0, :, :]

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Compute mean metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    # Log to tensorboard
    if summary_writer is not None:
        depth_model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image0=torch.cat(image_summary, dim=0),
            output_depth0=torch.cat(output_depth_summary, dim=0),
            sparse_depth0=torch.cat(sparse_depth_summary, dim=0),
            validity_map0=torch.cat(validity_map_summary, dim=0),
            ground_truth0=torch.cat(ground_truth_summary, dim=0),
            scalars={'mae' : mae, 'rmse' : rmse, 'imae' : imae, 'irmse': irmse},
            n_display=n_summary_display)

    # Print validation results to console
    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse, imae, irmse),
        log_path)

    n_improve = 0
    if np.round(mae, 2) <= np.round(best_results['mae'], 2):
        n_improve = n_improve + 1
    if np.round(rmse, 2) <= np.round(best_results['rmse'], 2):
        n_improve = n_improve + 1
    if np.round(imae, 2) <= np.round(best_results['imae'], 2):
        n_improve = n_improve + 1
    if np.round(irmse, 2) <= np.round(best_results['irmse'], 2):
        n_improve = n_improve + 1

    if n_improve > 2:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse

    log('Best results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse'],
        best_results['imae'],
        best_results['irmse']), log_path)

    return best_results


def run(image_path,
        sparse_depth_path,
        intrinsics_path,
        ground_truth_path,
        # Input settings
        load_image_triplets,
        input_types,
        input_channels_image,
        input_channels_depth,
        normalized_image_range,
        outlier_removal_kernel_size,
        outlier_removal_threshold,
        # Sparse to dense pool settings
        min_pool_sizes_sparse_to_dense_pool,
        max_pool_sizes_sparse_to_dense_pool,
        n_convolution_sparse_to_dense_pool,
        n_filter_sparse_to_dense_pool,
        # Depth network settings
        encoder_type,
        n_filters_encoder_image,
        n_filters_encoder_depth,
        n_convolutions_encoder,
        resolutions_backprojection,
        resolutions_depthwise_separable_encoder,
        decoder_type,
        n_filters_decoder,
        n_resolution_decoder,
        resolutions_depthwise_separable_decoder,
        min_predict_depth,
        max_predict_depth,
        # Weight settings
        weight_initializer,
        activation_func,
        # Evaluation settings
        min_evaluate_depth,
        max_evaluate_depth,
        # Checkpoint settings
        checkpoint_path,
        depth_model_restore_path,
        # Output settings
        save_outputs,
        keep_input_filenames,
        # Hardware settings
        device):

    # Select device to run on
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    # Set up checkpoint and output paths
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    log_path = os.path.join(checkpoint_path, 'results.txt')
    output_path = os.path.join(checkpoint_path, 'outputs')

    '''
    Load input paths and set up dataloader
    '''
    image_paths = data_utils.read_paths(image_path)
    sparse_depth_paths = data_utils.read_paths(sparse_depth_path)
    intrinsics_paths = data_utils.read_paths(intrinsics_path)

    ground_truth_available = False

    if ground_truth_path and ground_truth_path != '':
        ground_truth_available = True
        ground_truth_paths = data_utils.read_paths(ground_truth_path)

    n_sample = len(image_paths)

    input_paths = [
        image_paths,
        sparse_depth_paths,
        intrinsics_paths
    ]

    if ground_truth_available:
        input_paths.append(ground_truth_paths)

    for paths in input_paths:
        assert n_sample == len(paths)

    if ground_truth_available:

        ground_truths = []
        for path in ground_truth_paths:
            ground_truth, validity_map = data_utils.load_depth_with_validity_map(path)
            ground_truths.append(np.stack([ground_truth, validity_map], axis=-1))
    else:
        ground_truths = [None] * n_sample

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(
        datasets.DepthCompletionInferenceDataset(
            image_paths=image_paths,
            sparse_depth_paths=sparse_depth_paths,
            intrinsics_paths=intrinsics_paths,
            load_image_triplets=load_image_triplets),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    # Initialize transforms to normalize image and outlier removal for sparse depth
    transforms = Transforms(
        normalized_image_range=normalized_image_range)

    outlier_removal = OutlierRemoval(
        kernel_size=outlier_removal_kernel_size,
        threshold=outlier_removal_threshold)

    '''
    Set up the model
    '''
    depth_model = MonitoredDistillationModel(
        min_pool_sizes_sparse_to_dense_pool=min_pool_sizes_sparse_to_dense_pool,
        max_pool_sizes_sparse_to_dense_pool=max_pool_sizes_sparse_to_dense_pool,
        n_convolution_sparse_to_dense_pool=n_convolution_sparse_to_dense_pool,
        n_filter_sparse_to_dense_pool=n_filter_sparse_to_dense_pool,
        encoder_type=encoder_type,
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        n_convolutions_encoder=n_convolutions_encoder,
        resolutions_backprojection=resolutions_backprojection,
        resolutions_depthwise_separable_encoder=resolutions_depthwise_separable_encoder,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        n_resolution_decoder=n_resolution_decoder,
        resolutions_depthwise_separable_decoder=resolutions_depthwise_separable_decoder,
        activation_func=activation_func,
        weight_initializer=weight_initializer,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    # Restore model and set to evaluation mode
    depth_model.restore_model(depth_model_restore_path)
    depth_model.eval()

    parameters_depth_model = depth_model.parameters()

    '''
    Log input paths
    '''
    log('Input paths:', log_path)
    input_paths = [
        image_path,
        sparse_depth_path,
        intrinsics_path,
    ]

    if ground_truth_available:
        input_paths.append(ground_truth_path)

    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    '''
    Log all settings
    '''
    log_input_settings(
        log_path,
        # Input settings
        input_types=input_types,
        input_channels_image=input_channels_image,
        input_channels_depth=input_channels_depth,
        normalized_image_range=normalized_image_range,
        outlier_removal_kernel_size=outlier_removal_kernel_size,
        outlier_removal_threshold=outlier_removal_threshold)

    log_network_settings(
        log_path,
        # Sparse to dense pool settings
        min_pool_sizes_sparse_to_dense_pool=min_pool_sizes_sparse_to_dense_pool,
        max_pool_sizes_sparse_to_dense_pool=max_pool_sizes_sparse_to_dense_pool,
        n_convolution_sparse_to_dense_pool=n_convolution_sparse_to_dense_pool,
        n_filter_sparse_to_dense_pool=n_filter_sparse_to_dense_pool,
        # Depth network settings
        encoder_type=encoder_type,
        n_filters_encoder_image=n_filters_encoder_image,
        n_filters_encoder_depth=n_filters_encoder_depth,
        n_convolutions_encoder=n_convolutions_encoder,
        resolutions_backprojection=resolutions_backprojection,
        resolutions_depthwise_separable_encoder=resolutions_depthwise_separable_encoder,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        n_resolution_decoder=n_resolution_decoder,
        deconv_type='up',
        resolutions_depthwise_separable_decoder=resolutions_depthwise_separable_decoder,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_depth_model=parameters_depth_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        depth_model_restore_path=depth_model_restore_path,
        # Hardware settings
        device=device,
        n_thread=1)

    '''
    Run model
    '''
    # Set up metrics in case groundtruth is available
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    images = []
    output_depths = []
    sparse_depths = []

    time_elapse = 0.0

    for idx, (inputs, ground_truth) in enumerate(zip(dataloader, ground_truths)):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, sparse_depth, intrinsics = inputs

        time_start = time.time()

        with torch.no_grad():
            # Validity map is where sparse depth is available
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            # Remove outlier points and update sparse depth and validity map
            filtered_sparse_depth, \
                filtered_validity_map = outlier_removal.remove_outliers(
                    sparse_depth=sparse_depth,
                    validity_map=validity_map)

            if 'sparse_depth' in input_types:
                pass
            elif 'filtered_sparse_depth' in input_types:
                sparse_depth = filtered_sparse_depth
            else:
                raise ValueError('Sparse depth type is required')

            if 'validity_map' in input_types:
                pass
            elif 'filtered_validity_map' in input_types:
                validity_map = filtered_validity_map
            else:
                raise ValueError('Validity map type is required')

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            # Forward through network
            output_depth = depth_model.forward(
                image=image,
                sparse_depth=sparse_depth,
                validity_map=validity_map,
                intrinsics=intrinsics)

        time_elapse = time_elapse + (time.time() - time_start)

        # Convert to numpy
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        # Save to output
        if save_outputs:
            images.append(np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0)))
            sparse_depths.append(np.squeeze(filtered_sparse_depth.cpu().numpy()))
            output_depths.append(output_depth)

        if ground_truth_available:
            ground_truth = np.squeeze(ground_truth)

            validity_map = ground_truth[:, :, 1]
            ground_truth = ground_truth[:, :, 0]

            validity_mask = np.where(validity_map > 0, 1, 0)
            min_max_mask = np.logical_and(
                ground_truth > min_evaluate_depth,
                ground_truth < max_evaluate_depth)
            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            output_depth = output_depth[mask]
            ground_truth = ground_truth[mask]

            mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
            rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
            imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
            irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Compute total time elapse in ms
    time_elapse = time_elapse * 1000.0

    if ground_truth_available:
        mae_mean   = np.mean(mae)
        rmse_mean  = np.mean(rmse)
        imae_mean  = np.mean(imae)
        irmse_mean = np.mean(irmse)

        mae_std = np.std(mae)
        rmse_std = np.std(rmse)
        imae_std = np.std(imae)
        irmse_std = np.std(irmse)

        # Print evaluation results to console and file
        log('Evaluation results:', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}'.format(
            'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_mean, rmse_mean, imae_mean, irmse_mean),
            log_path)

        log('{:>8}  {:>8}  {:>8}  {:>8}'.format(
            '+/-', '+/-', '+/-', '+/-'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_std, rmse_std, imae_std, irmse_std),
            log_path)

    # Log run time
    log('Total time: {:.2f} ms  Average time per sample: {:.2f} ms'.format(
        time_elapse, time_elapse / float(n_sample)))

    if save_outputs:
        log('Saving outputs to {}'.format(output_path), log_path)

        outputs = zip(images, output_depths, sparse_depths, ground_truths)

        image_dirpath = os.path.join(output_path, 'image')
        output_depth_dirpath = os.path.join(output_path, 'output_depth')
        sparse_depth_dirpath = os.path.join(output_path, 'sparse_depth')
        ground_truth_dirpath = os.path.join(output_path, 'ground_truth')

        dirpaths = [
            image_dirpath,
            output_depth_dirpath,
            sparse_depth_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        for idx, (image, output_depth, sparse_depth, ground_truth) in enumerate(outputs):

            if keep_input_filenames:
                filename = os.path.basename(image_paths[idx])
            else:
                filename = '{:010d}.png'.format(idx)

            image_path = os.path.join(image_dirpath, filename)
            image = (255 * image).astype(np.uint8)
            Image.fromarray(image).save(image_path)

            output_depth_path = os.path.join(output_depth_dirpath, filename)
            data_utils.save_depth(output_depth, output_depth_path)

            sparse_depth_path = os.path.join(sparse_depth_dirpath, filename)
            data_utils.save_depth(sparse_depth, sparse_depth_path)

            if ground_truth_available:
                ground_truth_path = os.path.join(ground_truth_dirpath, filename)
                data_utils.save_depth(ground_truth[..., 0], ground_truth_path)


'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       input_types,
                       input_channels_image,
                       input_channels_depth,
                       normalized_image_range,
                       outlier_removal_kernel_size,
                       outlier_removal_threshold,
                       n_batch=None,
                       n_height=None,
                       n_width=None):

    batch_settings_text = ''
    batch_settings_vars = []

    if n_batch is not None:
        batch_settings_text = batch_settings_text + 'n_batch={}'
        batch_settings_vars.append(n_batch)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_height is not None:
        batch_settings_text = batch_settings_text + 'n_height={}'
        batch_settings_vars.append(n_height)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_width is not None:
        batch_settings_text = batch_settings_text + 'n_width={}'
        batch_settings_vars.append(n_width)

    log('Input settings:', log_path)

    if len(batch_settings_vars) > 0:
        log(batch_settings_text.format(*batch_settings_vars),
            log_path)

    log('input_types={}'.format(input_types), log_path)

    log('input_channels_image={}  input_channels_depth={}'.format(
        input_channels_image, input_channels_depth),
        log_path)
    log('normalized_image_range={}'.format(normalized_image_range),
        log_path)
    log('outlier_removal_kernel_size={}  outlier_removal_threshold={:.2f}'.format(
        outlier_removal_kernel_size, outlier_removal_threshold),
        log_path)
    log('', log_path)

def log_network_settings(log_path,
                         # Sparse to dense pool settings
                         min_pool_sizes_sparse_to_dense_pool,
                         max_pool_sizes_sparse_to_dense_pool,
                         n_convolution_sparse_to_dense_pool,
                         n_filter_sparse_to_dense_pool,
                         # Depth network settings
                         encoder_type,
                         n_filters_encoder_image,
                         n_filters_encoder_depth,
                         n_convolutions_encoder,
                         resolutions_backprojection,
                         resolutions_depthwise_separable_encoder,
                         decoder_type,
                         n_filters_decoder,
                         n_resolution_decoder,
                         deconv_type,
                         resolutions_depthwise_separable_decoder,
                         min_predict_depth,
                         max_predict_depth,
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         parameters_depth_model=[],
                         parameters_pose_model=[]):

    # Computer number of parameters
    n_parameter_depth = sum(p.numel() for p in parameters_depth_model)
    n_parameter_pose = sum(p.numel() for p in parameters_pose_model)

    n_parameter = n_parameter_depth + n_parameter_pose

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    if n_parameter_depth > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_depth={}'
        n_parameter_vars.append(n_parameter_depth)

    if n_parameter_pose > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_pose={}'
        n_parameter_vars.append(n_parameter_pose)

    log('Sparse to dense pooling settings:', log_path)
    log('min_pool_sizes_sparse_to_dense_pool={}'.format(min_pool_sizes_sparse_to_dense_pool),
        log_path)
    log('max_pool_sizes_sparse_to_dense_pool={}'.format(max_pool_sizes_sparse_to_dense_pool),
        log_path)
    log('n_convolution_sparse_to_dense_pool={}'.format(n_convolution_sparse_to_dense_pool),
        log_path)
    log('n_filter_sparse_to_dense_pool={}'.format(n_filter_sparse_to_dense_pool),
        log_path)
    log('', log_path)

    log('Depth network settings:', log_path)
    log('encoder_type={}'.format(encoder_type),
        log_path)
    log('n_filters_encoder_image={}'.format(n_filters_encoder_image),
        log_path)
    log('n_filters_encoder_depth={}'.format(n_filters_encoder_depth),
        log_path)
    log('n_convolutions_encoder={}'.format(n_convolutions_encoder),
        log_path)
    log('resolutions_backprojection={}'.format(resolutions_backprojection),
        log_path)
    log('resolutions_depthwise_separable_encoder={}'.format(resolutions_depthwise_separable_encoder),
        log_path)
    log('decoder_type={}'.format(decoder_type),
        log_path)
    log('n_filters_decoder={}  n_resolution_decoder={}'.format(
        n_filters_decoder, n_resolution_decoder),
        log_path)
    log('deconv_type={}'.format(deconv_type),
        log_path)
    log('resolutions_depthwise_separable_decoder={}'.format(resolutions_depthwise_separable_decoder),
        log_path)
    log('min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('weight_initializer={}  activation_func={}'.format(
        weight_initializer, activation_func),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          n_batch,
                          n_train_sample,
                          n_train_step,
                          supervision_types,
                          learning_rates_depth,
                          learning_schedule_depth,
                          learning_rates_pose,
                          learning_schedule_pose,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          augmentation_random_crop_type,
                          augmentation_random_crop_to_shape,
                          augmentation_random_remove_points,
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_saturation):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule_depth[-1], n_train_step),
        log_path)
    log('supervision_types={}'.format(
        supervision_types),
        log_path)
    log('learning_schedule_depth=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule_depth[:-1], learning_schedule_depth, learning_rates_depth)),
        log_path)

    log('learning_schedule_pose=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule_pose[:-1], learning_schedule_pose, learning_rates_pose)),
        log_path)

    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)
    log('augmentation_random_crop_type={}'.format(augmentation_random_crop_type),
        log_path)
    log('augmentation_random_crop_to_shape={}'.format(augmentation_random_crop_to_shape),
        log_path)
    log('augmentation_random_remove_points={}'.format(augmentation_random_remove_points),
        log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_stereo,
                           w_monocular,
                           w_color,
                           w_structure,
                           w_sparse_depth,
                           w_ensemble_depth,
                           w_ensemble_temperature,
                           w_sparse_select_ensemble,
                           sparse_select_dilate_kernel_size,
                           w_smoothness,
                           w_weight_decay_depth,
                           w_weight_decay_pose,
                           loss_func_ensemble_depth,
                           epoch_pose_for_ensemble,
                           ensemble_method):

    log('Loss function settings:', log_path)
    log('w_stereo={:.1e}  w_monocular={:.1e}'.format(
        w_stereo, w_monocular),
        log_path)
    log('w_color={:.1e}  w_structure={:.1e}  w_sparse_depth={:.1e}'.format(
        w_color, w_structure, w_sparse_depth),
        log_path)
    log('w_ensemble_depth={:.1e}  w_ensemble_temperature={:.1e}'.format(
        w_ensemble_depth, w_ensemble_temperature),
        log_path)
    log('w_sparse_select_ensemble={:.1e}  sparse_select_dilate_kernel_size={}'.format(
        w_sparse_select_ensemble, sparse_select_dilate_kernel_size),
        log_path)
    log('w_smoothness={:.1e}'.format(w_smoothness),
        log_path)
    log('w_weight_decay_depth={:.1e}  w_weight_decay_pose={:.1e}'.format(
        w_weight_decay_depth, w_weight_decay_pose),
        log_path)
    log('loss_func_ensemble_depth={}  epoch_pose_for_ensemble={}'.format(
        loss_func_ensemble_depth, epoch_pose_for_ensemble),
        log_path)
    log('ensemble_method={}'.format(ensemble_method),
        log_path)
    log('', log_path)

def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth):

    log('Evaluation settings:', log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_checkpoint=None,
                        summary_event_path=None,
                        n_summary=None,
                        n_summary_display=None,
                        validation_start_step=None,
                        depth_model_restore_path=None,
                        pose_model_restore_path=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_checkpoint is not None:
            log('checkpoint_save_frequency={}'.format(n_checkpoint), log_path)

        if validation_start_step is not None:
            log('validation_start_step={}'.format(validation_start_step), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_summary is not None:
        summary_settings_text = summary_settings_text + 'log_summary_frequency={}'
        summary_settings_vars.append(n_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if n_summary_display is not None:
        summary_settings_text = summary_settings_text + 'n_summary_display={}'
        summary_settings_vars.append(n_summary_display)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if depth_model_restore_path is not None and depth_model_restore_path != '':
        log('depth_model_restore_path={}'.format(depth_model_restore_path),
            log_path)

    if pose_model_restore_path is not None and pose_model_restore_path != '':
        log('pose_model_restore_path={}'.format(pose_model_restore_path),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
