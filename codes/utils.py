import os


def set_device(args):
    if args.cpu:
        if args.gpu is not None:
            raise ArgumentError("Can't use CPU and GPU at the same time")
    elif args.gpu is None:
        args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]


def get_train_cfg(args):
    return {
        'format_version': 4,
        'model_params': {
            'model_architecture': 'resnet34',
            'history_num_frames': args.history_num_frames,
            'history_step_size': 1,
            'history_delta_time': 0.1,
            'future_num_frames': args.future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1,
        },
        'raster_params': {
            
            'raster_size': [224, 224], # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
            'pixel_size': [0.5, 0.5], # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.
            'ego_center': [0.25, 0.5],
            'map_type': "py_semantic",
            
            # the keys are relative to the dataset environment variable
            'satellite_map_key': "aerial_map/aerial_map.png",
            'semantic_map_key': "semantic_map/semantic_map.pb",
            'dataset_meta_key': "meta.json",

            # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being
            # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.
            'filter_agents_threshold': 0.5,
            # Just to remove a warning
            'disable_traffic_light_faces': False
        },
    }


def get_test_cfg(args):
    return {
        'format_version': 4,
        'model_params': {
            'history_num_frames': args.history_num_frames,
            'history_step_size': 1,
            'history_delta_time': 0.1,
            'future_num_frames': args.future_num_frames,
            'future_step_size': 1,
            'future_delta_time': 0.1
        },
        
        'raster_params': {
            'raster_size': [300, 300],
            'pixel_size': [0.5, 0.5],
            'ego_center': [0.25, 0.5],
            'map_type': 'py_semantic',
            'satellite_map_key': 'aerial_map/aerial_map.png',
            'semantic_map_key': 'semantic_map/semantic_map.pb',
            'dataset_meta_key': 'meta.json',
            'filter_agents_threshold': 0.5,
            # Just to remove a warning
            'disable_traffic_light_faces': False
        },
    }