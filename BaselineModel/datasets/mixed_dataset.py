# -*- coding: utf-8 -*-
import os
import copy
import random
import pickle
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import one_to_index
import itertools
from BaselineModel.utils.protein.parsers import parse_biopython_structure


def get_indexes(tensor):
    unique_values, inverse_indices = torch.unique_consecutive(tensor, return_inverse=True)
    indexes = [torch.nonzero(inverse_indices == i).squeeze() for i in range(len(unique_values))]
    return indexes


def insert_eos(index_tensor, original_tensor, src:torch.Tensor=torch.tensor([22])):
    index_list = get_indexes(index_tensor)
    result = []
    for indexs in index_list[:-1]:
        result.append(torch.cat([original_tensor[indexs], src], dim=-1))
    result.append(original_tensor[index_list[-1]])
    return torch.cat(result)


def load_data_entries(csv_path, blocklist={'1KBH'}):
    df = pd.read_csv(csv_path)
    df['mutstr'] = df['mutstr'].apply(lambda x: x.replace('_', '') if type(x)==str else x)
    try:
        df = df[~((df['dG'].isna()) & (df['Nkd'].isna()) & (df['log2er'].isna()))]
    except:
        df = df[~df['dG'].isna()]
    df.reset_index(drop=True, inplace=True)
    entries = []

    def _parse_mut(mut_name):
        try:
            wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
            mutseq = int(mut_name[2:-1])
        except:
            import pdb
            pdb.set_trace()
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }

    for i, r in df.iterrows():
        if r['pdb'].upper() in blocklist:
            continue
        if type(r['mutstr']) == float:
            pass
        elif r['mutstr'][0]==',':
            continue
        try:
            entry = {
                'id': i,
                'complex': r['pdb'],
                'mutstr': 'None' if type(r['mutstr']) == float else r['mutstr'],
                'num_muts': 0 if type(r['mutstr']) == float else len(r['mutstr']),
                'pdbcode': f"{r['source']}_{r['pdb']}".upper(),
                'group_ligand': r['ligand'],
                'group_receptor': r['receptor'],
                'dimer': np.float32(len(r['ligand'])==5 and len(r['receptor'])==5),
                'l2andr2': np.float32(len(r['ligand'])<=10 and len(r['receptor'])<=10),
                'mutations': [None] if type(r['mutstr']) == float else list(map(_parse_mut, r['mutstr'].replace(' ','').split(','))),
                'dG': np.float32(r['dG']),
                'labels': r[['dG']].values.astype('float32'),
                'pdb_path': r['pdb_path'],
                'PP_ID': r['pdb']
            }
        except:
            entry = {
                'id': i,
                'complex': r['pdb'],
                'mutstr': 'None' if type(r['mutstr']) == float else r['mutstr'],
                'num_muts': 0 if type(r['mutstr']) == float else len(r['mutstr']),
                'pdbcode': f"{r['source']}_{r['pdb']}".upper(),
                'group_ligand': r['ligand'],
                'group_receptor': r['receptor'],
                'dimer': np.float32(len(r['ligand'])==5 and len(r['receptor'])==5),
                'l2andr2': np.float32(len(r['ligand'])<=10 and len(r['receptor'])<=10),
                'mutations': [None] if type(r['mutstr']) == float else list(map(_parse_mut, r['mutstr'].replace(' ','').split(','))),
                'dG': np.float32(r['dG']),
                'labels': r[['dG']].values.astype('float32'),
                'pdb_path': r['pdb_path'],
                'PP_ID': r['pdb']
            }
        entries.append(entry)

    entries = sorted(entries, key=lambda entry: entry['complex'])

    for i, entry in enumerate(entries):
        entry['id'] = i

    return entries


class MixedDataset(Dataset):

    def __init__(
            self,
            csv_path,
            cache_dir,
            cvfold_index=0,
            num_cvfolds=3,
            split='train',
            split_seed=2024,
            blocklist=frozenset({'1KBH', '4R8I', '1FYT'}),
            transform=None,
            reset=False,
            strict=True,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.transform = transform
        self.blocklist = blocklist
        self.cvfold_index = cvfold_index
        self.num_cvfolds = num_cvfolds
        assert split in ('train', 'val')
        self.split = split
        self.strict = strict
        self.split_seed = split_seed

        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, 'structures.pkl')
        self.structures = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)
            self.entries_full = [e for e in self.entries_full if e['complex'] not in self.blocklist]

        if self.strict:
            complex_to_entries = {}
            for e in self.entries_full:
                if e['PP_ID'] not in complex_to_entries:
                    complex_to_entries[e['PP_ID']] = []
                complex_to_entries[e['PP_ID']].append(e)

            complex_list = sorted(complex_to_entries.keys())
            random.Random(self.split_seed).shuffle(complex_list)


            split_size = math.ceil(len(complex_list) / self.num_cvfolds)
            complex_splits = [
                complex_list[i * split_size: (i + 1) * split_size]
                for i in range(self.num_cvfolds)
            ]
    
            val_split = complex_splits.pop(self.cvfold_index)
            train_split = sum(complex_splits, start=[])

            if self.split == 'val':
                complexes_this = val_split
            else:
                complexes_this = train_split

            entries = []
            for cplx in complexes_this:
                entries += complex_to_entries[cplx]
            self.entries = entries

        else:
            entries = self.entries_full
            random.Random(self.split_seed).shuffle(entries)

            split_size = math.ceil(len(entries) / self.num_cvfolds)
            entries_splits = [
                entries[i * split_size: (i + 1) * split_size]
                for i in range(self.num_cvfolds)
            ]

            val_split = entries_splits.pop(self.cvfold_index)
            train_split = sum(entries_splits, start=[])

            if self.split == 'val':
                entries_this = val_split
            else:
                entries_this = train_split

            self.entries = entries_this

    def _preprocess_entries(self):
        entries = load_data_entries(self.csv_path, self.blocklist)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        return entries

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures()
        else:
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)

    def _preprocess_structures(self):
        structures = {}

        pdb_path_list = list(set([e['pdb_path'] for e in self.entries_full]))
        for pdb_path in tqdm(pdb_path_list, desc='Structures'):
            parser = PDBParser(QUIET=True)
            model = parser.get_structure(None, pdb_path)[0]
            try:
                data, seq_map = parse_biopython_structure(model)
            except:
                import pdb
                pdb.set_trace()
            source = os.path.basename(os.path.dirname(pdb_path))
            pdbcode = os.path.splitext(os.path.basename(pdb_path))[0].upper()
            if '.ENT' in pdbcode or '.ent' in pdbcode:
                pdbcode = pdbcode[:-4]
            pdbcode = f'{source}_{pdbcode}'.upper()
            structures[pdbcode] = (data, seq_map)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        data, seq_map = copy.deepcopy(self.structures[entry['pdbcode']])
        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode', 'dG', 'labels', 'l2andr2', 'dimer'}
        for k in keys:
            try:
                data[k] = entry[k]
            except:
                pass

        group_id = []
        for ch in data['chain_id']:
            if ch in entry['group_ligand']:
                group_id.append(1)
            elif ch in entry['group_receptor']:
                group_id.append(2)
            else:
                group_id.append(0)
        data['group_id'] = torch.LongTensor(group_id)

        if entry['num_muts'] > 0:
            aa_mut = data['aa'].clone()
            for mut in entry['mutations']:
                ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
                if ch_rs_ic not in seq_map: continue
                aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
            data['mut_flag'] = (data['aa'] != aa_mut)
            data['aa'] = aa_mut
        else:
            data['mut_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)

        # labels_mask
        data['labels_mask'] = ~np.isnan(data['labels'])
        data['labels'] = np.nan_to_num(data['labels'])

        idx_ligand = torch.where(data['group_id'] == 1)[0]
        idx_receptor = torch.where(data['group_id'] == 2)[0]
        dist_pair = torch.cdist(data.pos_heavyatom[idx_ligand, 1, :], data.pos_heavyatom[idx_receptor, 1, :])  # 1 is C-alpha atom

        idx_ligand_itf, idx_receptor_itf = torch.where(dist_pair < 10.0)
        idx_ligand_itf = idx_ligand[torch.unique(idx_ligand_itf)]
        idx_receptor_itf = idx_receptor[torch.unique(idx_receptor_itf)]
        idx_itf = torch.cat([idx_ligand_itf, idx_receptor_itf])
        data['itf_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)
        data['itf_flag'][idx_itf] = True

        if self.transform is not None:
            try:
                data_tf = self.transform(data)
                data = data_tf
            except IndexError:
                print(data['pdbcode'])
                print(data['itf_flag'].sum())
                import pdb
                pdb.set_trace()
        
        return data


if __name__=='__main__':
    import tqdm
    import argparse
    import sys
    sys.path.append('../../')
    from BaselineModel.utils.misc import load_config
    from BaselineModel.utils.transforms import get_transform
    
    config, config_name = load_config(config_path='../../baseline_train_config.yml')
    dataset = MixedDataset(
        csv_path=config.data.csv_path,
        cache_dir=config.data.cache_dir,
        num_cvfolds=5,
        cvfold_index=1,
        split='val', 
        strict=config.data.strict,
        split_seed=config.data.split_seed,
        blocklist=config.data.blocklist,
        transform=get_transform(config.data.transform.train)
    )
    blocklist = []
    for i in tqdm.tqdm(range(dataset.__len__())):
        data = dataset.__getitem__(i)
        if data['block']:
            blocklist.append(data['pdbcode'])
        assert ~np.isnan(data['labels']).any(), data['pdbcode']
    print(blocklist)


