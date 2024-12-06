import os
import copy
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from tqdm.auto import tqdm

from BaselineModel.utils.data import PaddingCollate_struc
from BaselineModel.utils.train import *
from BaselineModel.utils.transforms import Compose, SelectAtom, SelectedRegionFixedSizePatch
from BaselineModel.utils.protein.parsers import parse_biopython_structure
from BaselineModel.utils.transforms import _index_select_data
from BaselineModel.models.dg_model import DG_Network

def _parse_mut(mut_name):
    wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
    mutseq = int(mut_name[2:-1])
    return {
        'wt': wt_type,
        'mt': mt_type,
        'chain': mutchain,
        'resseq': mutseq,
        'icode': ' ',
        'name': mut_name
    }


class InferDataset_Struc(Dataset):
    def __init__(self, df=None):
        super().__init__()
        self.entries = self.load_struc_entries(df)
        self.structures = self.load_structures()
        self.transform = Compose([
            SelectAtom('full'),
            SelectedRegionFixedSizePatch('itf_flag', 128)
        ])

    def load_struc_entries(self, summary_filepath):
        file_name, file_extension = os.path.splitext(summary_filepath)
        assert file_extension in ['.xlsx', '.xls',
                                  '.csv'], f"File extention of summary_filepath must in ['.xlsx', '.xls', '.csv'], but the input one is {file_extension}"
        if file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(summary_filepath)
        elif file_extension == '.csv':
            df = pd.read_csv(summary_filepath)
        entries = []

        for i, r in df.iterrows():
            pdb_name = r['PDB']
            ligand_chains = r['ligand']
            receptor_chains = r['receptor']
            dG = r['Binding_affinity'] if 'Binding_affinity' in r else np.nan
            pdb_path = r['pdb_path']
            entry = {
                'id': i,
                'complex': pdb_name + str(r['mutstr']),
                'mutstr': str(r['mutstr']),
                'num_muts': 0 if type(r['mutstr']) == float else 1,
                'pdbcode': pdb_name,
                'group_ligand': ligand_chains,
                'group_receptor': receptor_chains,
                'mutations': [None] if type(r['mutstr']) == float else list(
                    map(_parse_mut, r['mutstr'].replace(' ', '').split(','))),
                'dG': np.float32(dG),
                'pdb_path': pdb_path
            }
            entries.append(entry)
        return entries

    def load_structures(self):
        structures = {}
        pdbcodes = [e['pdbcode'] for e in self.entries]
        pdb_paths = [e['pdb_path'] for e in self.entries]
        for pdbcode, pdb_path in tqdm(zip(pdbcodes, pdb_paths), desc='Structures'):
            if pdbcode in structures:
                continue
            parser = PDBParser(QUIET=True)
            model = parser.get_structure(None, pdb_path)[0]
            data, seq_map = parse_biopython_structure(model)
            structures[pdbcode] = (data, seq_map)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        data, seq_map = copy.deepcopy(self.structures[entry['pdbcode']])
        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode', 'dG'}
        for k in keys:
            try:
                data[k] = entry[k]
            except:
                pass

        data['ID'] = f"{data['pdbcode']}_{data['mutstr']}_{entry['group_receptor']}_{entry['group_ligand']}"

        group_id = []
        for ch in data['chain_id']:
            if ch in entry['group_ligand']:  # 1 is ligand
                group_id.append(1)
            elif ch in entry['group_receptor']:  # 2 is receptor
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

        idx_keep = torch.where(data['group_id'] != 0)[0]
        data = _index_select_data(data, idx_keep)

        idx_ligand = torch.where(data['group_id'] == 1)[0]
        idx_receptor = torch.where(data['group_id'] == 2)[0]
        dist_pair = torch.cdist(data['pos_heavyatom'][idx_ligand, 1, :],
                                data['pos_heavyatom'][idx_receptor, 1, :])  # 1 is CA atom

        idx_ligand_itf, idx_receptor_itf = torch.where(dist_pair < 10.0)
        idx_ligand_itf = idx_ligand[torch.unique(idx_ligand_itf)]
        idx_receptor_itf = idx_receptor[torch.unique(idx_receptor_itf)]
        idx_itf = torch.cat([idx_ligand_itf, idx_receptor_itf])
        data['itf_flag'] = torch.full_like(data['aa'], False, dtype=torch.bool)
        data['itf_flag'][idx_itf] = True
        if data['itf_flag'].sum()==0:
            raise ValueError("No binding site found")

        return self.transform(data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--affinity_file_path', type=str, default='./example_data/example.xlsx')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/xxxx.pt')
    parser.add_argument('--save_path', type=str, default='./example_data/')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint_path, map_location=args.device)
    config = ckpt['config']
    print(config)

    cv_mgr = CrossValidation(
        model_factory=DG_Network,
        config=config,
        num_cvfolds=config['train']['num_cvfolds']
    )
    cv_mgr.load_state_dict(ckpt['model'])
    cv_mgr.to(args.device)

    dataset = InferDataset_Struc(args.affinity_file_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=PaddingCollate_struc(),
        num_workers=0
    )

    result = []
    for batch in tqdm(loader):
        batch = recursive_to(batch, args.device)
        for fold in range(cv_mgr.num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()
            all_dG_true = []
            all_dg_pred = []
            with torch.no_grad():
                pred_dG = model.infer(batch)
            for pdbname, dG_true, dG_pred in zip(batch['ID'], batch['dG'].cpu().tolist(),
                                                 pred_dG.cpu().tolist()):
                result.append({
                    'ID': pdbname,
                    'dG_true': dG_true,
                    'dG_pred': dG_pred,
                })
    result = pd.DataFrame(result)
    final_result = []
    print(result)
    for ID, df in result.groupby('ID'):
        dG_pred = df['dG_pred'].mean()
        dG_true = df['dG_true'].mean()
        final_result.append({
            'ID': ID,
            'dG_true': dG_true,
            'dG_pred': dG_pred,
        })
    final_result = pd.DataFrame(final_result)
    final_result.to_csv(os.path.join(args.save_path, 'predict_dG.csv'))
