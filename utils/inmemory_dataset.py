import os
from os.path import join, exists

import numpy as np

from pgl.utils.data import Dataloader
import sys
sys.path.insert(0, '/home/chenmingan/workplace/paddle/terpenoid-paddle/env/')
from pahelix.utils.data_utils import save_data_list_to_npz, load_npz_to_data_list
from pahelix.utils.basic_utils import mp_pool_map


__all__ = ['InMemoryDataset']


class InMemoryDataset(object):
    def __init__(self, 
            data_list=None,
            npz_data_path=None,
            npz_data_files=None):

        super(InMemoryDataset, self).__init__()
        self.data_list = data_list
        self.npz_data_path = npz_data_path
        self.npz_data_files = npz_data_files

        if not npz_data_path is None:
            self.data_list = self._load_npz_data_path(npz_data_path)

        if not npz_data_files is None:
            self.data_list = self._load_npz_data_files(npz_data_files)

    def _load_npz_data_path(self, data_path):
        data_list = []
        files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
        files = sorted(files)
        for f in files:
            data_list += load_npz_to_data_list(join(data_path, f))
        return data_list

    def _load_npz_data_files(self, data_files):
        data_list = []
        for f in data_files:
            data_list += load_npz_to_data_list(f)
        return data_list

    def _save_npz_data(self, data_list, data_path, max_num_per_file=10000):
        if not exists(data_path):
            os.makedirs(data_path)
        n = len(data_list)
        for i in range(int((n - 1) / max_num_per_file) + 1):
            filename = 'part-%06d.npz' % i
            sub_data_list = self.data_list[i * max_num_per_file: (i + 1) * max_num_per_file]
            save_data_list_to_npz(sub_data_list, join(data_path, filename))

    def save_data(self, data_path):
        """
        Save the ``data_list`` to the disk specified by ``data_path`` with npz format.
        After that, call `InMemoryDataset(data_path)` to reload the ``data_list``.

        Args:
            data_path(str): the path to the cached npz path.
        """
        self._save_npz_data(self.data_list, data_path)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            dataset = InMemoryDataset(
                    data_list=[self[i] for i in range(start, stop, step)])
            return dataset
        elif isinstance(key, int) or \
                isinstance(key, np.int64) or \
                isinstance(key, np.int32):
            return self.data_list[key]
        elif isinstance(key, list):
            dataset = InMemoryDataset(
                    data_list=[self[i] for i in key])
            return dataset
        else:
            raise TypeError('Invalid argument type: %s of %s' % (type(key), key))

    def __len__(self):
        return len(self.data_list)

    def transform(self, transform_fn, num_workers=1, drop_none=False):
        """
        Inplace apply `transform_fn` on the `data_list` with multiprocess.
        """
        data_list = mp_pool_map(self.data_list, transform_fn, num_workers)
        if drop_none:
            self.data_list = [data for data in data_list if not data is None]
        else:
            self.data_list = data_list

    def get_data_loader(self, batch_size, num_workers=1, shuffle=False, collate_fn=None):
        return Dataloader(self, 
                batch_size=batch_size, 
                num_workers=num_workers, 
                shuffle=shuffle,
                collate_fn=collate_fn)
