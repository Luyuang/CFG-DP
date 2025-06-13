from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()   # 由于添加了Jpeg压缩，读取时需要注册Jpeg2k

class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        self.img_keys = ['img01', 'img02', 'img03'] 
        self.low_dim_keys = ['state_joint_with_hand', 'cmd_joint_with_hand']
        self.slic = [(0, 8), (13, 21)]
        
        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_path, keys=self.img_keys+self.low_dim_keys)

        import zarr, os
        print('Loading cached ReplayBuffer from Disk.')
        with zarr.ZipStore(zarr_path, mode='r') as zip_store:
            self.replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore())
        print('Loaded!')

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'agent_pos': np.hstack([self.replay_buffer['state_joint_with_hand'][:, slice_range[0]:slice_range[1]] for slice_range in self.slic]),
            'action': np.hstack([self.replay_buffer['cmd_joint_with_hand'][:, slice_range[0]:slice_range[1]] for slice_range in self.slic]),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key in self.img_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state_joint_with_hand'].astype(np.float32) # (agent_posx2, block_posex3)
        action = sample['cmd_joint_with_hand'].astype(np.float32)
        # Initialize the data dictionary
        data = {
            'obs': {}
        }

        # Process each image key dynamically
        for img_key in self.img_keys:
            img = np.moveaxis(sample[img_key], -1, 1) / 255
            data['obs'][img_key] = img
        

        data['obs']['agent_pos'] = np.hstack([agent_pos[:, slice_range[0]:slice_range[1]] for slice_range in self.slic])
        data['action'] = np.hstack([action[:, slice_range[0]:slice_range[1]] for slice_range in self.slic])
    
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('/home/lab/hanxiao/dataset/kuavo/task_toy/toy_1/zarr/toy_1.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    plt.figure(figsize=(10, 5))
    plt.plot(dists)
    plt.xlabel('Time Step')
    plt.ylabel('Action Distance')
    plt.title('Action Distance over Time')
    plt.savefig('pusht_action_distance.png')
    
    
if "__main__" == __name__:
    test()
    print("here")