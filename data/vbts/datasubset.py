from pathlib import Path
from typing import Optional, List

from torch.utils.data import Dataset

from data.utils.augmentor import FlowAugmentor, PhotoAugmentor
from data.utils.keys import DataLoading, DataSetType
from data.utils.representations import norm_voxel_grid

from data.vbts.sample import Sample


class Datasubset(Dataset):
    def __init__(self,
                 train_or_val_or_pre_path: Path,
                 data_augm: bool,
                 num_bins_context: int,
                 flow_every_n_ms: int,
                 load_voxel_grid: bool=True,
                 extended_voxel_grid: bool=True,
                 normalize_voxel_grid: bool=False,
                 downsample: bool=False,
                 photo_augm: bool=False,
                 return_img: bool=True,
                 return_ev: bool=True):
        assert train_or_val_or_pre_path.is_dir()#判断目录是否存在
        assert train_or_val_or_pre_path.name in ('train', 'val', 'pre')#判断目录下面是否是这三个文件夹之一

        # Save output dimensions
        # 这是davis346相机，分辨率346x240
        original_height = 240
        original_width = 346

        #裁剪后的分辨率（待定），裁剪掉传感器内壁和标定图像的边缘区域
        # 原文裁剪分辨率的宽和高都能被16整除，疑似方便卷积
        crop_height = 368
        crop_width = 496

        self.return_img = return_img
        if not self.return_img:
            raise NotImplementedError
        self.return_ev = return_ev

        #下采样
        if downsample:
            crop_height = crop_height // 2
            crop_width = crop_width // 2

        #光流标签生成的时间间隔，这个很重要，后续对三维力估计的性能有影响，应该要兼顾力估计的准确性和光流推理计算的时间与误差。
        self.delta_ts_flow_ms = flow_every_n_ms

        self.spatial_augmentor = FlowAugmentor(
            crop_size_hw=(crop_height, crop_width),
            h_flip_prob=0.5,
            v_flip_prob=0.5) if data_augm else None
        
        #数据增强模块
        #亮度抖动    ±40%    光照突变场景；（VBTS要考虑，但应该不用太考虑）
        #色相偏移    ±0.5 rad    不同光源色温变化；（VBTS光源稳定）
        #高斯噪声    方差0.001-0.01    传感器读出噪声（VBTS要多多考虑，后面多研究）
        self.photo_augmentor = PhotoAugmentor(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.5/3.14,
            probability_color=0.2,
            noise_variance_range=(0.001, 0.01),
            probability_noise=0.2) if data_augm and photo_augm else None
        #归一化体素网格，提升计算效率
        self.normalize_voxel_grid: Optional[norm_voxel_grid] = norm_voxel_grid if normalize_voxel_grid else None

        sample_list: List[Sample] = list()
        for sample_path in train_or_val_or_pre_path.iterdir():
            if not sample_path.is_dir():
                continue
            sample_list.append(
                Sample(sample_path, original_height, original_width, num_bins_context, load_voxel_grid, extended_voxel_grid, downsample)
            )
        self.sample_list = sample_list

    def get_num_bins_context(self):
        return self.sample_list[0].num_bins_context

    def get_num_bins_correlation(self):
        return self.sample_list[0].num_bins_correlation

    def get_num_bins_total(self):
        return self.sample_list[0].num_bins_total

    def _voxel_grid_bin_idx_for_reference(self) -> int:
        return self.sample_list[0].voxel_grid_bin_idx_for_reference()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = self.sample_list[index]

        voxel_grid = sample.get_voxel_grid() if self.return_ev else None
        if voxel_grid is not None and self.normalize_voxel_grid is not None:
            voxel_grid = self.normalize_voxel_grid(voxel_grid)

        gt_flow_dict = sample.get_flow_gt(self.delta_ts_flow_ms)
        gt_flow = gt_flow_dict['flow']
        gt_flow_ts = gt_flow_dict['timestamps']

        imgs_with_ts = sample.get_images()
        imgs = imgs_with_ts['images']
        img_ts = imgs_with_ts['timestamps']

        # normalize image timestamps from 0 to 1
        assert len(img_ts) == 2
        ts_start = img_ts[0]
        ts_end = img_ts[1]
        assert ts_end > ts_start
        img_ts = [(x - ts_start)/(ts_end - ts_start) for x in img_ts]
        assert img_ts[0] == 0
        assert img_ts[1] == 1

        # we assume that img_ts[0] refers to reference time and img_ts[1] to the final target time
        gt_flow_ts = [(x - ts_start)/(ts_end - ts_start) for x in gt_flow_ts]
        assert gt_flow_ts[-1] == 1
        assert len(gt_flow_ts) == len(gt_flow)

        if self.spatial_augmentor is not None:
            if voxel_grid is None:
                gt_flow, imgs = self.spatial_augmentor(flow=gt_flow, images=imgs)
            else:
                voxel_grid, gt_flow, imgs = self.spatial_augmentor(ev_repr=voxel_grid, flow=gt_flow, images=imgs)
        if self.photo_augmentor is not None:
            imgs = self.photo_augmentor(imgs)
        out = {
            DataLoading.BIN_META: {
                'bin_idx_for_reference': self._voxel_grid_bin_idx_for_reference(),
                'nbins_context': self.get_num_bins_context(),
                'nbins_correlation': self.get_num_bins_correlation(),
                'nbins_total': self.get_num_bins_total(),
            },
            DataLoading.FLOW: gt_flow,
            DataLoading.FLOW_TIMESTAMPS: gt_flow_ts,
            DataLoading.IMG: imgs,
            DataLoading.IMG_TIMESTAMPS: img_ts,
            DataLoading.DATASET_TYPE: DataSetType.MULTIFLOW2D,
        }
        if voxel_grid is not None:
            out.update({DataLoading.EV_REPR: voxel_grid})

        return out
