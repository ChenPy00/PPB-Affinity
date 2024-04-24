# -*- coding: utf-8 -*-
import math
import torch
from torch.utils.data._utils.collate import default_collate
from torch.nn import functional as F


DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'aa_masked': 21,
    'aa_true': 21,
    'chain_nb': -1, 
    'pos14': 0.0,
    'chain_id': ' ', 
    'icode': ' ',
    'aa_ligand': 21,
    'aa_receptor': 21,
    'chain_nb_ligand': 0, 
    'chain_nb_receptor': 0
}


class PaddingCollate_struc(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)
        
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        return batch


# +
# 只考虑二/三/四聚体结构，其他赋予mask
class PaddingCollate_seq(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = {k for k in self._get_common_keys(data_list) if 'ligand' not in k and 'receptor' not in k}
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
            
#         ligand_max_length = max([data['ligand_embedding'].size(0) for data in data_list])
#         receptor_max_length = max([data['receptor_embedding'].size(0) for data in data_list])
        
        ligand_max_length = max([data['aa_ligand'].size(0) for data in data_list])
        receptor_max_length = max([data['aa_receptor'].size(0) for data in data_list])
        
        keys = {k for k in self._get_common_keys(data_list) if 'embedding' in k}
        data_list_padded_embedding = []
        for data in data_list:
            if data['aa_ligand'].size(0)<ligand_max_length:                
                pad_length = ligand_max_length-data['aa_ligand'].size(0)
                ligand_embedding =  F.pad(data['ligand_embedding'], 
                                                  pad=(0, 0, 0, pad_length), 
                                                  mode='constant', value=0)
                ligand_aa = torch.cat([
                    data['aa_ligand'], 
                    torch.tensor([DEFAULT_PAD_VALUES['aa_ligand']] * pad_length),
                ], dim=0)

            else:
                ligand_embedding = data['ligand_embedding']
                ligand_aa = data['aa_ligand']

            mask_ligand = torch.cat([
                torch.ones([data['ligand_embedding'].size(0)], dtype=torch.bool),
                torch.zeros([ligand_max_length-data['ligand_embedding'].size(0)], dtype=torch.bool)
            ], dim=0)

            if data['aa_receptor'].size(0)<receptor_max_length:
                pad_length = receptor_max_length-data['aa_receptor'].size(0)
                receptor_embedding =  F.pad(data['receptor_embedding'], 
                                                  pad=(0, 0, 0, pad_length), 
                                                  mode='constant', value=0)
                receptor_aa = torch.cat([
                    data['aa_receptor'], 
                    torch.tensor([DEFAULT_PAD_VALUES['aa_receptor']] * pad_length),
                ], dim=0)

            else:
                receptor_embedding = data['receptor_embedding']
                receptor_aa = data['aa_receptor']

            mask_receptor = torch.cat([
                torch.ones([data['receptor_embedding'].size(0)], dtype=torch.bool),
                torch.zeros([receptor_max_length-data['receptor_embedding'].size(0)], dtype=torch.bool)
            ], dim=0)
                    
            data_list_padded_embedding.append({'ligand_embedding': ligand_embedding.float(), 
                                              'aa_ligand': ligand_aa,
                                              'mask_ligand': mask_ligand, 
                                              'receptor_embedding': receptor_embedding.float(),
                                              'aa_receptor': receptor_aa,
                                              'mask_receptor': mask_receptor})
        

        data_list_padded = [{**d1, **d2} for d1, d2 in zip(data_list_padded, data_list_padded_embedding)]
        try:
            batch = default_collate(data_list_padded)
        except:
            print([data['pdbcode'] for data in data_list])
            print([data['mutstr'] for data in data_list])
            print('ligand', [(data['aa_ligand'].size(0), data['ligand_embedding'].size(0)) for data in data_list])
            print('receptor', [(data['aa_receptor'].size(0), data['receptor_embedding'].size(0)) for data in data_list])            
            import pdb
            pdb.set_trace()
        batch['size'] = len(data_list_padded)
        return batch
