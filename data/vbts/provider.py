import copy
from pathlib import Path
from typing import Dict, Any

import torch.utils.data

# 添加项目根目录到系统路径
import sys
sys.path.append("/home/lyh/bflow_new")  # 项目根目录绝对路径


from data.utils.provider0 import DatasetProviderBase
from data.vbts.datasubset import Datasubset


class DatasetProvider(DatasetProviderBase):
    def __init__(self,
                 dataset_params: Dict[str, Any],
                 nbins_context: int):
        dataset_path = Path(dataset_params['path'])
        pre_path = dataset_path / 'pre'
        train_path = dataset_path / 'train'
        assert dataset_path.is_dir(), str(dataset_path)
        assert pre_path.is_dir(), str(pre_path)


        return_img = True
        return_img_key = 'return_img'#加载帧图像
        if return_img_key in dataset_params:#如果用户配置了，按用户的来。dataset_params是config的dataset目录下的配置文件
            return_img = dataset_params[return_img_key]
        return_ev = True
        return_ev_key = 'return_ev'
        if return_ev_key in dataset_params:#加载事件流图像
            return_ev = dataset_params[return_ev_key]#如果用户配置了，按用户的来。dataset_params是config的dataset目录下的配置文件

        #注意！！！！下面这个都要写在pre.yaml配置文件里面！！！！！！！！！    
        base_args = {
            'num_bins_context': 41,#先随便写个数字
            'load_voxel_grid': dataset_params['load_voxel_grid'],
            'normalize_voxel_grid': dataset_params['normalize_voxel_grid'],
            'extended_voxel_grid': dataset_params['extended_voxel_grid'],
            'flow_every_n_ms': dataset_params['flow_every_n_ms'],
            'downsample': dataset_params['downsample'],
            'photo_augm': dataset_params['photo_augm'],
            return_img_key: return_img,
            return_ev_key: return_ev,
        }  #参数配置
        train_args = copy.deepcopy(base_args)#拷贝一份
        train_args.update({'data_augm': True})
        #val_test_args = copy.deepcopy(base_args)
        #val_test_args.update({'data_augm': False})

        
        pre_args = copy.deepcopy(base_args)
        pre_args.update({'data_augm': False}) #data_augm有啥影响？？????

        train_dataset = Datasubset(train_path, **train_args)
        self.nbins_context = 41
        self.nbins_correlation = 25

        #self.train_dataset = train_dataset
        #self.val_dataset = Datasubset(val_path, **val_test_args)
        #assert self.val_dataset.get_num_bins_context() == self.nbins_context
        #assert self.val_dataset.get_num_bins_correlation() == self.nbins_correlation
        self.train_dataset = train_dataset
        self.pre_dataset = Datasubset(pre_path, **pre_args)
        # assert self.pre_dataset.get_num_bins_context() == self.nbins_context           #推理模式可能不需要这个判断，先注释掉
        # assert self.pre_dataset.get_num_bins_correlation() == self.nbins_correlation


    def get_pre_dataset(self):
        return self.pre_dataset

    def get_nbins_context(self):
        return self.nbins_context

    def get_nbins_correlation(self):
        return self.nbins_correlation
