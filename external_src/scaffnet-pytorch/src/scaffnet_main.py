import os, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils, eval_utils
from log_utils import log
from scaffnet_model import ScaffNetModel


def train(train_sparse_depth_path,
          train_ground_truth_path,
          val_sparse_depth_path,
          val_ground_truth_path,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          # Dataset settings
          cap_dataset_depth_method,
          min_dataset_depth,
          max_dataset_depth,
          # Spatial pyramid pool settings
          max_pool_sizes_spatial_pyramid_pool,
          n_convolution_spatial_pyramid_pool,
          n_filter_spatial_pyramid_pool,
          # Network settings
          encoder_type,
          n_filters_encoder,
          decoder_type,
          n_filters_decoder,
          min_predict_depth,
          max_predict_depth,
          # Weight settings
          weight_initializer,
          activation_func,
          # Training settings
          learning_rates,
          learning_schedule,
          freeze_network_modules,
          # Augmentation settings
          augmentation_random_crop_type,
          # Loss settings
          loss_func,
          w_supervised,
          w_weight_decay,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          n_summary,
          n_summary_display,
          n_checkpoint,
          checkpoint_path,
          validation_start_step,
          restore_path,
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

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    model_checkpoint_path = os.path.join(checkpoint_path, 'model-{}.pth')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty,
        'mare_sparse_depth': np.infty
    }

    # Read paths for training
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)
    assert len(train_sparse_depth_paths) == len(train_ground_truth_paths)

    n_train_sample = len(train_sparse_depth_paths)

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.ScaffNetTrainingDataset(
            sparse_depth_paths=train_sparse_depth_paths,
            dense_depth_paths=train_ground_truth_paths,
            cap_dataset_depth_method=cap_dataset_depth_method,
            min_dataset_depth=min_dataset_depth,
            max_dataset_depth=max_dataset_depth,
            random_crop_shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    # Load validation data if it is available
    validation_available = val_sparse_depth_path is not None and \
        val_ground_truth_path is not None

    if validation_available:
        val_sparse_depth_paths = data_utils.read_paths(val_sparse_depth_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        assert(len(val_sparse_depth_paths) == len(val_ground_truth_paths))

        val_ground_truths = []
        for path in val_ground_truth_paths:
            ground_truth, validity_map = data_utils.load_depth_with_validity_map(path)
            val_ground_truths.append(np.stack([ground_truth, validity_map], axis=-1))

        val_dataloader = torch.utils.data.DataLoader(
            datasets.ScaffNetInferenceDataset(depth_paths=val_sparse_depth_paths),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

    # Build ScaffNet
    model = ScaffNetModel(
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        encoder_type=encoder_type,
        n_filters_encoder=n_filters_encoder,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    parameters_model = model.parameters()

    if restore_path is not None:
        if os.path.exists(restore_path):
            model.restore_model(restore_path)
        else:
            raise ValueError('Restore path does not exist: {}'.format(restore_path))

    # Log settings
    train_summary = SummaryWriter(event_path + '-train')
    val_summary = SummaryWriter(event_path + '-val')

    if 'weight_decay' not in loss_func:
        w_weight_decay = 0.0

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        train_sparse_depth_path,
        train_ground_truth_path,
    ]

    for path in train_input_paths:
        log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_sparse_depth_path,
        val_ground_truth_path
    ]
    for path in val_input_paths:
        log(path, log_path)
    log('', log_path)

    '''
    Log all settings
    '''
    log_input_settings(
        log_path,
        # Batch settings
        n_batch=n_batch,
        n_height=n_height,
        n_width=n_width,
        # Dataset settings
        cap_dataset_depth_method=cap_dataset_depth_method,
        min_dataset_depth=min_dataset_depth,
        max_dataset_depth=max_dataset_depth)

    log_network_settings(
        log_path,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        # Depth network settings
        encoder_type=encoder_type,
        n_filters_encoder=n_filters_encoder,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_model)

    log_training_settings(
        log_path,
        # Training settings
        n_batch=n_batch,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        freeze_network_modules=freeze_network_modules,
        # Augmentation settings
        augmentation_random_crop_type=augmentation_random_crop_type)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        w_supervised=w_supervised,
        w_weight_decay=w_weight_decay)

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
        model_restore_path=restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    optimizer = torch.optim.Adam([
        {
            'params' : parameters_model,
            'weight_decay' : w_weight_decay
        }],
        lr=learning_rate)

    # Start training
    model.train()

    start_step = 0

    if restore_path is not None and restore_path != '':
        start_step, optimizer = model.restore_model(
            restore_path,
            optimizer=optimizer)

        for g in optimizer.param_groups:
            g['lr'] = learning_rate

    # Freeze a portion of the network
    if freeze_network_modules is not None:
        model.freeze(freeze_network_modules)

    train_step = start_step

    time_start = time.time()

    # Split along batch across multiple GPUs
    if torch.cuda.device_count() > 1:
        model.data_parallel()

    log('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        for inputs in train_dataloader:

            train_step = train_step + 1

            # Fetch data
            inputs = [
                in_.to(device) for in_ in inputs
            ]

            sparse_depth, ground_truth = inputs

            # Forward through the network
            output_depth = model.forward(sparse_depth)

            if 'uncertainty' in decoder_type:
                output_uncertainty = output_depth[:, 1:2, :, :]
                output_depth = output_depth[:, 0:1, :, :]
            else:
                output_uncertainty = None

            # Compute loss function
            loss, loss_info = model.compute_loss(
                loss_func=loss_func,
                target_depth=ground_truth,
                output_depth=output_depth,
                output_uncertainty=output_uncertainty,
                w_supervised=w_supervised)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_summary) == 0:
                model.log_summary(
                    summary_writer=train_summary,
                    tag='train',
                    step=train_step,
                    output_depth=output_depth.detach().clone(),
                    output_uncertainty=output_uncertainty,
                    ground_truth=torch.cat([ground_truth, torch.ones_like(ground_truth)], dim=1),
                    scalars=loss_info,
                    n_display=min(n_batch, n_summary_display))

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step + start_step) * time_elapse / (train_step - start_step)

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step + start_step, loss.item(), time_elapse, time_remain), log_path)

                if train_step >= validation_start_step and validation_available:
                    # Switch to validation mode
                    model.eval()

                    with torch.no_grad():
                        best_results = validate(
                            model=model,
                            dataloader=val_dataloader,
                            ground_truths=val_ground_truths,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary,
                            n_summary_display=n_summary_display,
                            log_path=log_path)

                    # Switch back to training
                    model.train()

                    # Keep modules frozen if set
                    if freeze_network_modules is not None:
                        model.freeze(freeze_network_modules)

                # Save checkpoint
                model.save_model(
                    model_checkpoint_path.format(train_step),
                    step=train_step,
                    optimizer=optimizer)

    log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
        train_step, n_train_step + start_step, loss.item(), time_elapse, time_remain), log_path)

    # Switch to validation mode
    model.eval()

    with torch.no_grad():
        best_results = validate(
            model=model,
            dataloader=val_dataloader,
            ground_truths=val_ground_truths,
            step=train_step,
            best_results=best_results,
            min_evaluate_depth=min_evaluate_depth,
            max_evaluate_depth=max_evaluate_depth,
            device=device,
            summary_writer=val_summary,
            n_summary_display=n_summary_display,
            log_path=log_path)

    # Save checkpoint at last step
    model.save_model(
        model_checkpoint_path.format(train_step),
        step=train_step,
        optimizer=optimizer)

def validate(model,
             dataloader,
             ground_truths,
             step,
             best_results,
             # Depth evaluation range
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             summary_writer=None,
             n_summary_display=4,
             n_summary_display_interval=200,
             log_path=None):

    n_sample = len(dataloader)

    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)
    mare_sparse_depth = np.zeros(n_sample)

    output_depth_summary = []
    output_uncertainty_summary = []
    ground_truth_summary = []

    for idx, (sparse_depth, ground_truth) in enumerate(zip(dataloader, ground_truths)):
        ground_truth = np.expand_dims(ground_truth, axis=0)
        ground_truth = np.transpose(ground_truth, (0, 3, 1, 2))

        # Fetch data
        sparse_depth = sparse_depth.to(device)
        ground_truth = torch.from_numpy(ground_truth).to(device)

        # Forward through the network
        output_depth = model.forward(sparse_depth)

        if 'uncertainty' in model.decoder_type:
            output_uncertainty = output_depth[:, 1:2, :, :]
            output_depth = output_depth[:, 0:1, :, :]
        else:
            output_uncertainty = None

        if (idx % n_summary_display_interval) == 0 and summary_writer is not None:
            output_depth_summary.append(output_depth)
            output_uncertainty_summary.append(output_uncertainty)
            ground_truth_summary.append(ground_truth)

        # Convert to numpy to validate
        sparse_depth = np.squeeze(sparse_depth.cpu().numpy())
        output_depth = np.squeeze(output_depth.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        validity_map_ground_truth = ground_truth[1, :, :]
        ground_truth = ground_truth[0, :, :]

        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)

        # Compute MARE metric on sparse depth
        validity_mask_sparse_depth = np.where(sparse_depth > 0, 1, 0)
        mask_sparse_depth = \
            np.where(np.logical_and(validity_mask_sparse_depth, min_max_mask) > 0)

        mare_sparse_depth[idx] = eval_utils.mean_abs_rel_err(
            output_depth[mask_sparse_depth],
            sparse_depth[mask_sparse_depth])

        # Compute MAE, RMSE, iMAE, iRMSE metric on ground truth
        validity_mask_ground_truth = np.where(validity_map_ground_truth > 0, 1, 0)
        mask_ground_truth = \
            np.where(np.logical_and(validity_mask_ground_truth, min_max_mask) > 0)

        output_depth = output_depth[mask_ground_truth]
        ground_truth = ground_truth[mask_ground_truth]

        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)
    mare_sparse_depth = np.mean(mare_sparse_depth)

    # Log to tensorboard
    if summary_writer is not None:

        scalars = {
            'mae' : mae,
            'rmse' : rmse,
            'imae' : imae,
            'irmse': irmse,
            'mare_sparse_depth': mare_sparse_depth
        }

        if None not in output_uncertainty_summary:
            output_uncertainty_summary = torch.cat(output_uncertainty_summary, dim=0)
        else:
            output_uncertainty_summary = None

        model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            output_depth=torch.cat(output_depth_summary, dim=0),
            output_uncertainty=output_uncertainty_summary,
            ground_truth=torch.cat(ground_truth_summary, dim=0),
            scalars=scalars,
            n_display=n_summary_display)

    # Print validation results to console
    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', 'MARE Sparse Depth'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse, imae, irmse, mare_sparse_depth),
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
        best_results['mare_sparse_depth'] = mare_sparse_depth

    log('Best results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', 'MARE Sparse Depth'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse'],
        best_results['imae'],
        best_results['irmse'],
        best_results['mare_sparse_depth']), log_path)

    return best_results

def run(sparse_depth_path,
        ground_truth_path,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool,
        # Network settings
        encoder_type,
        n_filters_encoder,
        decoder_type,
        n_filters_decoder,
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

    # Set up checkpoint and output paths
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    log_path = os.path.join(checkpoint_path, 'results.txt')
    output_path = os.path.join(checkpoint_path, 'outputs')

    '''
    Load input paths and set up dataloader
    '''
    sparse_depth_paths = data_utils.read_paths(sparse_depth_path)

    ground_truth_available = False

    if ground_truth_path != '':
        ground_truth_available = True
        ground_truth_paths = data_utils.read_paths(ground_truth_path)

    n_sample = len(sparse_depth_paths)

    input_paths = [
        sparse_depth_paths
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
        datasets.ScaffNetInferenceDataset(depth_paths=sparse_depth_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    '''
    Set up the model
    '''
    # Build ScaffNet
    model = ScaffNetModel(
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        encoder_type=encoder_type,
        n_filters_encoder=n_filters_encoder,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    # Restore model and set to evaluation mode
    model.restore_model(restore_path)
    model.eval()

    parameters_model = model.parameters()

    '''
    Log input paths
    '''
    log('Input paths:', log_path)
    input_paths = [
        sparse_depth_path
    ]

    if ground_truth_available:
        input_paths.append(ground_truth_path)

    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    '''
    Log all settings
    '''
    log_network_settings(
        log_path,
        # Spatial pyramid pool settings
        max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
        n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
        n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
        # Depth network settings
        encoder_type=encoder_type,
        n_filters_encoder=n_filters_encoder,
        decoder_type=decoder_type,
        n_filters_decoder=n_filters_decoder,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        parameters_model=parameters_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        model_restore_path=restore_path,
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

    output_depths = []
    output_uncertainties = []
    sparse_depths = []

    time_elapse = 0.0

    for idx, (sparse_depth, ground_truth) in enumerate(zip(dataloader, ground_truths)):

        # Fetch data
        sparse_depth = sparse_depth.to(device)

        time_start = time.time()

        with torch.no_grad():
            # Forward through the network
            output_depth = model.forward(sparse_depth)

        time_elapse = time_elapse + (time.time() - time_start)

        if 'uncertainty' in model.decoder_type:
            output_uncertainty = output_depth[:, 1:2, :, :]
            output_depth = output_depth[:, 0:1, :, :]
        else:
            output_uncertainty = None

        # Convert to numpy
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        # Save to output
        if save_outputs:
            sparse_depths.append(np.squeeze(sparse_depth.cpu().numpy()))
            output_depths.append(output_depth)

            if output_uncertainty is not None:
                output_uncertainties.append(np.squeeze(output_uncertainty.detach().cpu().numpy()))

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

        outputs = zip(output_depths, output_uncertainties, sparse_depths, ground_truths)

        output_depth_dirpath = os.path.join(output_path, 'output_depth')
        output_uncertainty_dirpath = os.path.join(output_path, 'output_uncertainty')
        sparse_depth_dirpath = os.path.join(output_path, 'sparse_depth')
        ground_truth_dirpath = os.path.join(output_path, 'ground_truth')

        dirpaths = [
            output_depth_dirpath,
            output_uncertainty_dirpath,
            sparse_depth_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        for idx, (output_depth, output_uncertainty, sparse_depth, ground_truth) in enumerate(outputs):

            if keep_input_filenames:
                filename = os.path.basename(sparse_depth_paths[idx])
            else:
                filename = '{:010d}.png'.format(idx)

            output_depth_path = os.path.join(output_depth_dirpath, filename)
            data_utils.save_depth(output_depth, output_depth_path)

            if output_uncertainty is not None:
                output_uncertainty_path = os.path.join(output_uncertainty_dirpath, filename)
                data_utils.save_uncertainty(output_uncertainty, output_uncertainty_path)

            sparse_depth_path = os.path.join(sparse_depth_dirpath, filename)
            data_utils.save_depth(sparse_depth, sparse_depth_path)

            if ground_truth_available:
                ground_truth_path = os.path.join(ground_truth_dirpath, filename)
                data_utils.save_depth(ground_truth[..., 0], ground_truth_path)


'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       # Batch settings
                       n_batch=None,
                       n_height=None,
                       n_width=None,
                       # Dataset settings
                       cap_dataset_depth_method=None,
                       min_dataset_depth=None,
                       max_dataset_depth=None):

    batch_settings_text = ''
    batch_settings_vars = []

    dataset_settings_text = ''
    dataset_settings_vars = []

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

    if cap_dataset_depth_method is not None:
        dataset_settings_text = dataset_settings_text + 'cap_dataset_depth_method={}'
        dataset_settings_vars.append(cap_dataset_depth_method)

    dataset_settings_text = \
        dataset_settings_text + '  ' if len(dataset_settings_text) > 0 else dataset_settings_text

    if min_dataset_depth is not None:
        dataset_settings_text = dataset_settings_text + 'min_dataset_depth={}'
        dataset_settings_vars.append(min_dataset_depth)

    dataset_settings_text = \
        dataset_settings_text + '  ' if len(dataset_settings_text) > 0 else dataset_settings_text

    if max_dataset_depth is not None:
        dataset_settings_text = dataset_settings_text + 'max_dataset_depth={}'
        dataset_settings_vars.append(max_dataset_depth)

    log('Input settings:', log_path)

    if len(batch_settings_vars) > 0:
        log(batch_settings_text.format(*batch_settings_vars),
            log_path)

    if len(dataset_settings_vars) > 0:
        log(dataset_settings_text.format(*dataset_settings_vars),
            log_path)

    log('', log_path)

def log_network_settings(log_path,
                         # Spatial pyramid pool settings
                         max_pool_sizes_spatial_pyramid_pool,
                         n_convolution_spatial_pyramid_pool,
                         n_filter_spatial_pyramid_pool,
                         # Depth network settings
                         encoder_type,
                         n_filters_encoder,
                         decoder_type,
                         n_filters_decoder,
                         min_predict_depth,
                         max_predict_depth,
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         parameters_model=[]):

    # Computer number of parameters
    n_parameter = sum(p.numel() for p in parameters_model)

    log('Spatial pyramid pooling settings:', log_path)
    log('max_pool_sizes_spatial_pyramid_pool={}'.format(max_pool_sizes_spatial_pyramid_pool),
        log_path)
    log('n_convolution_spatial_pyramid_pool={}'.format(n_convolution_spatial_pyramid_pool),
        log_path)
    log('n_filter_spatial_pyramid_pool={}'.format(n_filter_spatial_pyramid_pool),
        log_path)
    log('', log_path)

    log('Depth network settings:', log_path)
    log('encoder_type={}'.format(encoder_type),
        log_path)
    log('n_filters_encoder={}'.format(n_filters_encoder),
        log_path)
    log('decoder_type={}'.format(decoder_type),
        log_path)
    log('n_filters_decoder={}'.format(
        n_filters_decoder),
        log_path)
    log('min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log('n_parameter={}'.format(n_parameter),
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
                          learning_rates,
                          learning_schedule,
                          freeze_network_modules,
                          # Augmentation settings
                          augmentation_random_crop_type):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('freeze_network_modules={}'.format(
        freeze_network_modules),
        log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_random_crop_type={}'.format(augmentation_random_crop_type),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_supervised,
                           w_weight_decay):

    log('Loss function settings:', log_path)
    log('w_supervised={:.1e}'.format(
        w_supervised),
        log_path)
    log('w_weight_decay={:.1e}'.format(
        w_weight_decay),
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
                        model_restore_path=None,
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

    if model_restore_path is not None and model_restore_path != '':
        log('depth_model_restore_path={}'.format(model_restore_path),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
