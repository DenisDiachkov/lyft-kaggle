import os
import torch
import numpy as np


def set_device(args):
    if args.cpu:
        if args.gpu is not None:
            raise ArgumentError("Can't use CPU and GPU at the same time")
    elif args.gpu is None:
        args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]


def get_train_cfg(args):
    return {
        "format_version": 4,
        "model_params": {
            "model_architecture": "resnet34",
            "history_num_frames": args.history_num_frames,
            "history_step_size": 1,
            "history_delta_time": 0.1,
            "future_num_frames": args.future_num_frames,
            "future_step_size": 1,
            "future_delta_time": 0.1,
        },
        "raster_params": {
            
            "raster_size": [224, 224], # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
            "pixel_size": [0.5, 0.5], # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.
            "ego_center": [0.25, 0.5],
            "map_type": "py_semantic",
            
            # the keys are relative to the dataset environment variable
            "satellite_map_key": "aerial_map/aerial_map.png",
            "semantic_map_key": "semantic_map/semantic_map.pb",
            "dataset_meta_key": "meta.json",

            # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being
            # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.
            "filter_agents_threshold": 0.5,
            # Just to remove a warning
            "disable_traffic_light_faces": False
        },
    }


def get_test_cfg(args):
    return {
        "format_version": 4,
        "model_params": {
            "history_num_frames": args.history_num_frames,
            "history_step_size": 1,
            "history_delta_time": 0.1,
            "future_num_frames": args.future_num_frames,
            "future_step_size": 1,
            "future_delta_time": 0.1
        },
        
        "raster_params": {
            "raster_size": [300, 300],
            "pixel_size": [0.5, 0.5],
            "ego_center": [0.25, 0.5],
            "map_type": "py_semantic",
            "satellite_map_key": "aerial_map/aerial_map.png",
            "semantic_map_key": "semantic_map/semantic_map.pb",
            "dataset_meta_key": "meta.json",
            "filter_agents_threshold": 0.5,
            # Just to remove a warning
            "disable_traffic_light_faces": False
        },
    }


def pytorch_neg_multi_log_likelihood_batch(gt, pred, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(gt, pred, avails):
    """
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)
