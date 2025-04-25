from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Disable PIL's DecompressionBombError (optional, if trusted source)
Image.MAX_IMAGE_PIXELS = None

def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start
    return heatmap, total_time

def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    start_time = time.time()
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    start_time = time.time()
    file_path = WSI_object.process_contours(**kwargs)
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
                  patch_size=256, step_size=256,
                  seg_params=None, filter_params=None,
                  vis_params=None, patch_params=None,
                  patch_level=0, use_default_params=False,
                  seg=False, save_mask=True, stitch=False,
                  patch=False, auto_skip=True, process_list=None):

    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]
    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
                          'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
                          'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
                          'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
                          'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times, patch_times, stitch_times = 0., 0., 0.

    for i in tqdm(range(total)):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Try initializing WSI safely
        full_path = os.path.join(source, slide)
        try:
            WSI_object = WholeSlideImage(full_path)
        except Exception as e:
            print(f"Failed to open {slide} due to error: {e}")
            df.loc[idx, 'status'] = f'failed_open: {str(e).splitlines()[0][:100]}'
            continue

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
        else:
            current_vis_params, current_filter_params = {}, {}
            current_seg_params, current_patch_params = {}, {}

            for key in vis_params:
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params[key] = df.loc[idx, key]

            for key in filter_params:
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    df.loc[idx, key] = adjusted_area
                    current_filter_params[key] = adjusted_area
                else:
                    current_filter_params[key] = df.loc[idx, key]

            for key in seg_params:
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params[key] = df.loc[idx, key]

            for key in patch_params:
                current_patch_params[key] = df.loc[idx, key]

        if current_vis_params['vis_level'] < 0:
            current_vis_params['vis_level'] = WSI_object.getOpenSlide().get_best_level_for_downsample(64)

        if current_seg_params['seg_level'] < 0:
            current_seg_params['seg_level'] = WSI_object.getOpenSlide().get_best_level_for_downsample(64)

        current_seg_params['keep_ids'] = np.array(current_seg_params['keep_ids'].split(','), dtype=int) if current_seg_params['keep_ids'] != 'none' else []
        current_seg_params['exclude_ids'] = np.array(current_seg_params['exclude_ids'].split(','), dtype=int) if current_seg_params['exclude_ids'] != 'none' else []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e8:
            print('level_dim {} x {} too large, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask.save(os.path.join(mask_save_dir, slide_id + '.jpg'))

        patch_time_elapsed = -1
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params)

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id + '.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                heatmap.save(os.path.join(stitch_save_dir, slide_id + '.jpg'))

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))

        df.loc[idx, 'status'] = 'processed'
        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))
    return seg_times, patch_times

