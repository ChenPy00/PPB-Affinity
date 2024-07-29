#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os

data_dir = '../PPB-Affinity-DataPrepWorkflow/processed_data'
path_dict = {
    'SKEMPI v2.0': './PDB/SKEMPI v2.0/',
    'PDBbind v2020': './PDB/PDBbind v2020/',
    'SAbDab': './PDB/SAbDab/',
    'ATLAS': './PDB/ATLAS/',
    'Affinity Benchmark v5.5': './PDB/Affinity Benchmark v5.5/',
}
for k,v in path_dict.items(): path_dict[k] = os.path.join(data_dir,v)

case_type = {
    'SKEMPI v2.0': 'upper.pdb',
    'PDBbind v2020': 'lower.ent.pdb',
    'SAbDab': 'lower.pdb',
    'ATLAS': 'upper.pdb',
    'Affinity Benchmark v5.5': 'upper.pdb',
}


def create_pdb_path(inputs):
    pdb_code, source = inputs
    c_t = case_type[source]
    c_t = c_t.split('.')
    suffix = '.' + '.'.join(c_t[1:])
    case = c_t[0]
    file_name = eval(f'pdb_code.{case}()') + suffix
    pdb_path = os.path.join(path_dict[source], file_name)
    if os.path.exists(pdb_path):
        return pdb_path# os.path.realpath(pdb_path)
    else:
        import pdb
        pdb.set_trace()


def mutstr_transform(mutstr):
    if type(mutstr) == str:
        mutstr = mutstr.replace('_', '')
        mutstr = mutstr[1] + mutstr[0] + mutstr[2:]
    else:
        pass
    return mutstr


affinity_data = pd.read_excel('../PPB-Affinity-DataPrepWorkflow/processed_data/PPB-Affinity.xlsx',
                              usecols=['PDB', 'Source Data Set', 'Mutations', 'Ligand Chains', 'Receptor Chains',
                                       'dG(kcal/mol)', 'Subgroup'],
                              dtype={"PDB": str})
affinity_data.rename(columns={'Source Data Set': 'source', 'dG(kcal/mol)': 'dG',
                              'Mutations': 'mutstr', 'PDB': 'pdb',
                              'Ligand Chains': 'ligand', 'Receptor Chains': 'receptor'}, inplace=True)


affinity_data = affinity_data[~(affinity_data.ligand.str.len() + affinity_data.receptor.str.len() > 26)]
affinity_data.reset_index(drop=True, inplace=True)
affinity_data['pdb_path'] = affinity_data[['pdb', 'source']].apply(create_pdb_path, axis=1)

affinity_data['mutstr'] = affinity_data['mutstr'].apply(mutstr_transform)

affinity_data.to_csv('./benchmark.csv')
