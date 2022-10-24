import os, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils, eval_utils
from log_utils import log
from fusionnet_model import FusionNetModel
from scaffnet_model import ScaffNetModel
from posenet_model import PoseNetModel
from transforms import Transforms
from net_utils import OutlierRemoval
from fusionnet_main import log_input_settings
from fusionnet_main import log_evaluation_settings
from fusionnet_main import log_loss_func_settings
from fusionnet_main import log_system_settings
from fusionnet_main import log_training_settings
from scaffnet_main import log_network_settings as log_scaffnet_settings
from fusionnet_main import log_network_settings as log_fusionnet_settings


def train(train_images_path,
          train_sparse_depth_path,
          train_intrinsics_path,
          val_image_path,
          val_sparse_depth_path,
          val_ground_truth_path,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          # Input settings
          normalized_image_range,
          outlier_removal_kernel_size,
          outlier_removal_threshold,
          # Spatial pyramid pool settings
          max_pool_sizes_spatial_pyramid_pool,
          n_convolution_spatial_pyramid_pool,
          n_filter_spatial_pyramid_pool,
          # ScaffNet settings
          encoder_type_scaffnet,
          n_filters_encoder_scaffnet,
          decoder_type_scaffnet,
          n_filters_decoder_scaffnet,
          min_predict_depth_scaffnet,
          max_predict_depth_scaffnet,
          # FusionNet network settings
          encoder_type_fusionnet,
          n_filters_encoder_image_fusionnet,
          n_filters_encoder_depth_fusionnet,
          decoder_type_fusionnet,
          n_filters_decoder_fusionnet,
          scale_match_method_fusionnet,
          scale_match_kernel_size_fusionnet,
          min_predict_depth_fusionnet,
          max_predict_depth_fusionnet,
          min_multiplier_depth_fusionnet,
          max_multiplier_depth_fusionnet,
          min_residual_depth_fusionnet,
          max_residual_depth_fusionnet,
          # Weight settings
          weight_initializer,
          activation_func,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_random_crop_type,
          # Loss function settings
          w_color,
          w_structure,
          w_sparse_depth,
          w_smoothness,
          w_prior_depth,
          threshold_prior_depth,
          w_weight_decay_depth,
          w_weight_decay_pose,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          n_summary,
          n_summary_display,
          n_checkpoint,
          checkpoint_path,
          scaffnet_model_restore_path,
          fusionnet_model_restore_path,
          posenet_model_restore_path,
          # Hardware settings
          device='cuda',
          n_thread=8):

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

    depth_model_checkpoint_path = os.path.join(checkpoint_path, 'depth_model-{}.pth')
    pose_model_checkpoint_path = os.path.join(checkpoint_path, 'pose_model-{}.pth')
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
    train_images_paths = data_utils.read_paths(train_images_path)
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)
    train_intrinsics_paths = data_utils.read_paths(train_intrinsics_path)

    n_train_sample = len(train_images_paths)

    # Make sure number of paths match number of training sample
    input_paths = [
        train_sparse_depth_paths,
        train_intrinsics_paths
    ]

    for paths in input_paths:
        assert len(paths) == n_train_sample

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetStandaloneTrainingDataset(
            images_paths=train_images_paths,
            sparse_depth_paths=train_sparse_depth_paths,
            intrinsics_paths=train_intrinsics_paths,
            random_crop_shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range)

    # Load validation data if it is available
    validation_available = \
        val_image_path is not None and \
        val_sparse_depth_path is not None and \
        val_ground_truth_path is not None

    if validation_available:
        val_image_paths = data_utils.read_paths(val_image_path)
        val_sparse_depth_paths = data_utils.read_paths(val_sparse_depth_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        n_val_sample = len(val_image_paths)

        input_paths = [
            val_sparse_depth_paths, val_ground_truth_paths
        ]

        for paths in input_paths:
            assert len(paths) == n_val_sample

        ground_truths = []
        for path in val_ground_truth_paths:
            ground_truth, validity_map = data_utils.load_depth_with_validity_map(path)
            ground_truths.append(np.stack([ground_truth, validity_map], axis=-1))

        val_dataloader = torch.utils.data.DataLoader(
            datasets.FusionNetStandaloneInferenceDataset(
                image_paths=val_image_paths,
                sparse_depth_paths=val_sparse_depth_paths),
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

    # Build ScaffNet
    scaffnet_model = ScaffNetModel(
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        encoder_type=encoder_type_scaffnet,
        n_filters_encoder=n_filters_encoder_scaffnet,
        decoder_type=decoder_type_scaffnet,
        n_filters_decoder=n_filters_decoder_scaffnet,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        min_predict_depth=min_predict_depth_scaffnet,
        max_predict_depth=max_predict_depth_scaffnet,
        device=device)

    scaffnet_model.restore_model(scaffnet_model_restore_path)
    scaffnet_model.eval()
    parameters_scaffnet_model = scaffnet_model.parameters()

    # Build FusionNet
    fusionnet_model = FusionNetModel(
        encoder_type=encoder_type_fusionnet,
        n_filters_encoder_image=n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
        decoder_type=decoder_type_fusionnet,
        n_filters_decoder=n_filters_decoder_fusionnet,
        scale_match_method=scale_match_method_fusionnet,
        scale_match_kernel_size=scale_match_kernel_size_fusionnet,
        min_predict_depth=min_predict_depth_fusionnet,
        max_predict_depth=max_predict_depth_fusionnet,
        min_multiplier_depth=min_multiplier_depth_fusionnet,
        max_multiplier_depth=max_multiplier_depth_fusionnet,
        min_residual_depth=min_residual_depth_fusionnet,
        max_residual_depth=max_residual_depth_fusionnet,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        device=device)

    parameters_fusionnet_model = fusionnet_model.parameters()

    # Bulid PoseNet (only needed for training) network
    posenet_model = PoseNetModel(
        encoder_type='posenet',
        rotation_parameterization='axis',
        weight_initializer=weight_initializer,
        activation_func='relu',
        device=device)

    parameters_posenet_model = posenet_model.parameters()

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        train_images_path,
        train_sparse_depth_path,
        train_intrinsics_path,
    ]

    for path in train_input_paths:
        log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_sparse_depth_path,
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
        normalized_image_range=normalized_image_range,
        outlier_removal_kernel_size=outlier_removal_kernel_size,
        outlier_removal_threshold=outlier_removal_threshold)

    log_scaffnet_settings(
        log_path,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        # Depth network settings
        encoder_type=encoder_type_scaffnet,
        n_filters_encoder=n_filters_encoder_scaffnet,
        decoder_type=decoder_type_scaffnet,
        n_filters_decoder=n_filters_decoder_scaffnet,
        min_predict_depth=min_predict_depth_scaffnet,
        max_predict_depth=max_predict_depth_scaffnet,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_scaffnet_model)

    log_fusionnet_settings(
        log_path,
        # Depth network settings
        encoder_type=encoder_type_fusionnet,
        n_filters_encoder_image=n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
        decoder_type=decoder_type_fusionnet,
        n_filters_decoder=n_filters_decoder_fusionnet,
        scale_match_method=scale_match_method_fusionnet,
        scale_match_kernel_size=scale_match_kernel_size_fusionnet,
        min_predict_depth=min_predict_depth_fusionnet,
        max_predict_depth=max_predict_depth_fusionnet,
        min_multiplier_depth=min_multiplier_depth_fusionnet,
        max_multiplier_depth=max_multiplier_depth_fusionnet,
        min_residual_depth=min_residual_depth_fusionnet,
        max_residual_depth=max_residual_depth_fusionnet,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_depth_model=parameters_fusionnet_model,
        parameters_pose_model=parameters_posenet_model)

    log_training_settings(
        log_path,
        # Training settings
        n_batch=n_batch,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        # Augmentation settings
        augmentation_random_crop_type=augmentation_random_crop_type)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        w_color=w_color,
        w_structure=w_structure,
        w_sparse_depth=w_sparse_depth,
        w_smoothness=w_smoothness,
        w_prior_depth=w_prior_depth,
        threshold_prior_depth=threshold_prior_depth,
        w_weight_decay_depth=w_weight_decay_depth,
        w_weight_decay_pose=w_weight_decay_pose)

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
        depth_model_restore_path=fusionnet_model_restore_path,
        pose_model_restore_path=posenet_model_restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    optimizer_fusionnet = torch.optim.Adam([
        {
            'params' : parameters_fusionnet_model,
            'weight_decay' : w_weight_decay_depth
        }],
        lr=learning_rate)

    optimizer_posenet = torch.optim.Adam([
        {
            'params' : parameters_posenet_model,
            'weight_decay' : w_weight_decay_pose
        }],
        lr=learning_rate)

    # Start training
    train_step = 0

    if fusionnet_model_restore_path is not None and fusionnet_model_restore_path != '':
        train_step, optimizer_fusionnet = fusionnet_model.restore_model(
            fusionnet_model_restore_path,
            optimizer=optimizer_fusionnet,
            scaffnet_model=scaffnet_model if scaffnet_model_restore_path is None else None)

        for g in optimizer_fusionnet.param_groups:
            g['lr'] = learning_rate

    if posenet_model_restore_path is not None and posenet_model_restore_path != '':
        _, optimizer_posenet = posenet_model.restore_model(
            posenet_model_restore_path,
            optimizer=optimizer_posenet)

        for g in optimizer_posenet.param_groups:
            g['lr'] = learning_rate

    time_start = time.time()

    log('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer_fusionnet.param_groups:
                g['lr'] = learning_rate

            for g in optimizer_posenet.param_groups:
                g['lr'] = learning_rate

        for inputs in train_dataloader:

            train_step = train_step + 1

            # Fetch data
            inputs = [
                in_.to(device) for in_ in inputs
            ]

            image0, \
                image1, \
                image2, \
                sparse_depth0, \
                intrinsics = inputs

            with torch.no_grad():
                # Forward through ScaffNet
                input_depth0 = scaffnet_model.forward(sparse_depth0)

                if 'uncertainty' in scaffnet_model.decoder_type:
                    input_depth0 = input_depth0[:, 0:1, :, :]

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

            # Transforms
            [image0, image1, image2], \
                [filtered_sparse_depth0], \
                [input_depth0, filtered_validity_map0] = train_transforms.transform(
                    images_arr=[image0, image1, image2],
                    range_maps_arr=[filtered_sparse_depth0],
                    validity_maps_arr=[input_depth0, filtered_validity_map0])

            # Forward through FusionNet
            output_depth0 = fusionnet_model.forward(
                image=image0,
                input_depth=input_depth0,
                sparse_depth=filtered_sparse_depth0)

            # Forward through PoseNet
            pose0to1 = posenet_model.forward(image0, image1)
            pose0to2 = posenet_model.forward(image0, image2)

            # Compute loss function
            loss, loss_info = fusionnet_model.compute_loss(
                output_depth0=output_depth0,
                sparse_depth0=filtered_sparse_depth0,
                validity_map0=filtered_validity_map0,
                input_depth0=input_depth0,
                image0=image0,
                image1=image1,
                image2=image2,
                pose0to1=pose0to1,
                pose0to2=pose0to2,
                intrinsics=intrinsics,
                w_color=w_color,
                w_structure=w_structure,
                w_sparse_depth=w_sparse_depth,
                w_smoothness=w_smoothness,
                w_prior_depth=w_prior_depth,
                threshold_prior_depth=threshold_prior_depth)

            # Compute gradient and backpropagate
            optimizer_fusionnet.zero_grad()
            optimizer_posenet.zero_grad()

            loss.backward()

            optimizer_fusionnet.step()
            optimizer_posenet.step()

            if (train_step % n_summary) == 0:

                image1to0 = loss_info.pop('image1to0')
                image2to0 = loss_info.pop('image2to0')

                fusionnet_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    image0=image0,
                    image1to0=image1to0.detach().clone(),
                    image2to0=image2to0.detach().clone(),
                    output_depth0=output_depth0.detach().clone(),
                    sparse_depth0=filtered_sparse_depth0,
                    validity_map0=filtered_validity_map0,
                    input_depth0=input_depth0.detach().clone(),
                    pose0to1=pose0to1,
                    pose0to2=pose0to2,
                    scalars=loss_info,
                    n_display=min(n_batch, n_summary_display))

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain), log_path)

                if validation_available:
                    # Switch to validation mode
                    fusionnet_model.eval()

                    with torch.no_grad():
                        # Perform validation
                        best_results = validate(
                            scaffnet_model=scaffnet_model,
                            fusionnet_model=fusionnet_model,
                            dataloader=val_dataloader,
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
                    fusionnet_model.train()

                fusionnet_model.save_model(
                    depth_model_checkpoint_path.format(train_step),
                    train_step,
                    optimizer_fusionnet,
                    scaffnet_model)
                posenet_model.save_model(
                    pose_model_checkpoint_path.format(train_step),
                    train_step,
                    optimizer_posenet)

    fusionnet_model.save_model(
        depth_model_checkpoint_path.format(train_step),
        train_step,
        optimizer_fusionnet,
        scaffnet_model)
    posenet_model.save_model(
        pose_model_checkpoint_path.format(train_step),
        train_step,
        optimizer_posenet)

def validate(scaffnet_model,
             fusionnet_model,
             dataloader,
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
    input_depth_summary = []
    ground_truth_summary = []

    for idx, (inputs, ground_truth) in enumerate(zip(dataloader, ground_truths)):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, sparse_depth = inputs

        ground_truth = np.expand_dims(ground_truth, axis=0)
        ground_truth = np.transpose(ground_truth, (0, 3, 1, 2))
        ground_truth = torch.from_numpy(ground_truth).to(device)

        with torch.no_grad():

            # Forward through ScaffNet
            input_depth = scaffnet_model.forward(sparse_depth)

            if 'uncertainty' in scaffnet_model.decoder_type:
                input_depth = input_depth[:, 0:1, :, :]

            # Validity map is where sparse depth is available
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            # Remove outlier points and update sparse depth and validity map
            filtered_sparse_depth, _ = outlier_removal.remove_outliers(
                sparse_depth=sparse_depth,
                validity_map=validity_map)

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            # Forward through network
            output_depth = fusionnet_model.forward(
                image=image,
                input_depth=input_depth,
                sparse_depth=filtered_sparse_depth)

        if (idx % n_summary_display_interval) == 0 and summary_writer is not None:
            image_summary.append(image)
            output_depth_summary.append(output_depth)
            sparse_depth_summary.append(filtered_sparse_depth)
            input_depth_summary.append(input_depth)
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
        fusionnet_model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image0=torch.cat(image_summary, dim=0),
            output_depth0=torch.cat(output_depth_summary, dim=0),
            sparse_depth0=torch.cat(sparse_depth_summary, dim=0),
            input_depth0=torch.cat(input_depth_summary, dim=0),
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
        ground_truth_path,
        # Input settings
        normalized_image_range,
        outlier_removal_kernel_size,
        outlier_removal_threshold,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool,
        # ScaffNet settings
        encoder_type_scaffnet,
        n_filters_encoder_scaffnet,
        decoder_type_scaffnet,
        n_filters_decoder_scaffnet,
        min_predict_depth_scaffnet,
        max_predict_depth_scaffnet,
        # FusionNet network settings
        encoder_type_fusionnet,
        n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth_fusionnet,
        decoder_type_fusionnet,
        n_filters_decoder_fusionnet,
        scale_match_method_fusionnet,
        scale_match_kernel_size_fusionnet,
        min_predict_depth_fusionnet,
        max_predict_depth_fusionnet,
        min_multiplier_depth_fusionnet,
        max_multiplier_depth_fusionnet,
        min_residual_depth_fusionnet,
        max_residual_depth_fusionnet,
        # Weight settings
        weight_initializer,
        activation_func,
        # Evaluation settings
        min_evaluate_depth,
        max_evaluate_depth,
        # Checkpoint settings
        checkpoint_path,
        restore_path,
        # Output settings
        save_outputs,
        keep_input_filenames,
        # Hardware settings
        device='cuda'):

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

    log_path = os.path.join(checkpoint_path, 'results.txt')
    output_path = os.path.join(checkpoint_path, 'outputs')

    '''
    Load input paths and set up dataloaders
    '''
    image_paths = data_utils.read_paths(image_path)
    sparse_depth_paths = data_utils.read_paths(sparse_depth_path)

    ground_truth_available = False

    if ground_truth_path != '':
        ground_truth_available = True
        ground_truth_paths = data_utils.read_paths(ground_truth_path)

    n_sample = len(image_paths)

    input_paths = [
        sparse_depth_paths, ground_truth_paths
    ]

    for paths in input_paths:
        assert len(paths) == n_sample

    if ground_truth_available:

        ground_truths = []
        for path in ground_truth_paths:
            ground_truth, validity_map = data_utils.load_depth_with_validity_map(path)
            ground_truths.append(np.stack([ground_truth, validity_map], axis=-1))
    else:
        ground_truths = [None] * n_sample

    dataloader = torch.utils.data.DataLoader(
        datasets.FusionNetStandaloneInferenceDataset(
            image_paths=image_paths,
            sparse_depth_paths=sparse_depth_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    transforms = Transforms(
        normalized_image_range=normalized_image_range)

    # Initialize outlier removal for sparse depth
    outlier_removal = OutlierRemoval(
        kernel_size=outlier_removal_kernel_size,
        threshold=outlier_removal_threshold)

    # Build ScaffNet
    scaffnet_model = ScaffNetModel(
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        encoder_type=encoder_type_scaffnet,
        n_filters_encoder=n_filters_encoder_scaffnet,
        decoder_type=decoder_type_scaffnet,
        n_filters_decoder=n_filters_decoder_scaffnet,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        min_predict_depth=min_predict_depth_scaffnet,
        max_predict_depth=max_predict_depth_scaffnet,
        device=device)

    # Build FusionNet
    fusionnet_model = FusionNetModel(
        encoder_type=encoder_type_fusionnet,
        n_filters_encoder_image=n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
        decoder_type=decoder_type_fusionnet,
        n_filters_decoder=n_filters_decoder_fusionnet,
        scale_match_method=scale_match_method_fusionnet,
        scale_match_kernel_size=scale_match_kernel_size_fusionnet,
        min_predict_depth=min_predict_depth_fusionnet,
        max_predict_depth=max_predict_depth_fusionnet,
        min_multiplier_depth=min_multiplier_depth_fusionnet,
        max_multiplier_depth=max_multiplier_depth_fusionnet,
        min_residual_depth=min_residual_depth_fusionnet,
        max_residual_depth=max_residual_depth_fusionnet,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        device=device)

    # Restore model and set to evaluation mode
    fusionnet_model.restore_model(
        checkpoint_path=restore_path,
        scaffnet_model=scaffnet_model)

    scaffnet_model.eval()
    fusionnet_model.eval()

    parameters_scaffnet_model = scaffnet_model.parameters()
    parameters_fusionnet_model = fusionnet_model.parameters()

    '''
    Log input paths
    '''
    log('Input paths:', log_path)
    input_paths = [
        image_path,
        sparse_depth_path,
        ground_truth_path
    ]
    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        # Input settings
        normalized_image_range=normalized_image_range,
        outlier_removal_kernel_size=outlier_removal_kernel_size,
        outlier_removal_threshold=outlier_removal_threshold)

    log_input_settings(
        log_path,
        # Input settings
        normalized_image_range=normalized_image_range,
        outlier_removal_kernel_size=outlier_removal_kernel_size,
        outlier_removal_threshold=outlier_removal_threshold)

    log_scaffnet_settings(
        log_path,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        # Depth network settings
        encoder_type=encoder_type_scaffnet,
        n_filters_encoder=n_filters_encoder_scaffnet,
        decoder_type=decoder_type_scaffnet,
        n_filters_decoder=n_filters_decoder_scaffnet,
        min_predict_depth=min_predict_depth_scaffnet,
        max_predict_depth=max_predict_depth_scaffnet,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_scaffnet_model)

    log_fusionnet_settings(
        log_path,
        # Depth network settings
        encoder_type=encoder_type_fusionnet,
        n_filters_encoder_image=n_filters_encoder_image_fusionnet,
        n_filters_encoder_depth=n_filters_encoder_depth_fusionnet,
        decoder_type=decoder_type_fusionnet,
        n_filters_decoder=n_filters_decoder_fusionnet,
        scale_match_method=scale_match_method_fusionnet,
        scale_match_kernel_size=scale_match_kernel_size_fusionnet,
        min_predict_depth=min_predict_depth_fusionnet,
        max_predict_depth=max_predict_depth_fusionnet,
        min_multiplier_depth=min_multiplier_depth_fusionnet,
        max_multiplier_depth=max_multiplier_depth_fusionnet,
        min_residual_depth=min_residual_depth_fusionnet,
        max_residual_depth=max_residual_depth_fusionnet,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_depth_model=parameters_fusionnet_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        depth_model_restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=1)

    # Set up metrics in case groundtruth is available
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    output_depths = []
    input_depths = []
    sparse_depths = []

    time_elapse = 0.0

    for idx, (inputs, ground_truth) in enumerate(zip(dataloader, ground_truths)):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, sparse_depth = inputs

        time_start = time.time()

        with torch.no_grad():

            # Forward through ScaffNet
            input_depth = scaffnet_model.forward(sparse_depth)

            if 'uncertainty' in scaffnet_model.decoder_type:
                input_depth = input_depth[:, 0:1, :, :]

            # Validity map is where sparse depth is available
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            # Remove outlier points and update sparse depth and validity map
            filtered_sparse_depth, _ = outlier_removal.remove_outliers(
                sparse_depth=sparse_depth,
                validity_map=validity_map)

            [image] = transforms.transform(
                images_arr=[image],
                random_transform_probability=0.0)

            # Forward through network
            output_depth = fusionnet_model.forward(
                image=image,
                input_depth=input_depth,
                sparse_depth=filtered_sparse_depth)

        time_elapse = time_elapse + (time.time() - time_start)

        # Convert to numpy
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        # Save to output
        if save_outputs:
            sparse_depths.append(np.squeeze(sparse_depth.cpu().numpy()))
            output_depths.append(output_depth)
            input_depths.append(np.squeeze(input_depth.cpu().numpy()))

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

        outputs = zip(output_depths, input_depths, sparse_depths, ground_truths)

        output_depth_dirpath = os.path.join(output_path, 'output_depth')
        input_depth_dirpath = os.path.join(output_path, 'input_depth')
        sparse_depth_dirpath = os.path.join(output_path, 'sparse_depth')
        ground_truth_dirpath = os.path.join(output_path, 'ground_truth')

        dirpaths = [
            output_depth_dirpath,
            input_depth_dirpath,
            sparse_depth_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        for idx, (output_depth, input_depth, sparse_depth, ground_truth) in enumerate(outputs):

            if keep_input_filenames:
                filename = os.path.basename(sparse_depth_paths[idx])
            else:
                filename = '{:010d}.png'.format(idx)

            output_depth_path = os.path.join(output_depth_dirpath, filename)
            data_utils.save_depth(output_depth, output_depth_path)

            input_depth_path = os.path.join(input_depth_dirpath, filename)
            data_utils.save_depth(input_depth, input_depth_path)

            sparse_depth_path = os.path.join(sparse_depth_dirpath, filename)
            data_utils.save_depth(sparse_depth, sparse_depth_path)

            if ground_truth_available:
                ground_truth_path = os.path.join(ground_truth_dirpath, filename)
                data_utils.save_depth(ground_truth[..., 0], ground_truth_path)
