# PPB-Affinity

PPB-Affinity: Protein-Protein Binding Affinity dataset for AI-based protein drug discovery

Prediction of protein-protein binding (PPB) affinity plays an important role in large-molecular drug discovery. Deep learning (DL) has been adopted to predict the change of PPB binding affinity upon mutation, but there was a scarcity of studies predicting the PPB affinity itself. The major reason is the paucity of open-source dataset concerning PPB affinity. Therefore, the current study aimed to introduce and disclose a PPB affinity dataset (PPB-Affinity), which will definitely benefit the development of applicable DL to predict the PPB affinity. The PPB-Affinity dataset contains key information such as crystal structures of protein-protein complexes (with or without protein mutation patterns), PPB affinity, receptor protein chain, ligand protein chain, etc. To the best of our knowledge, this is the largest and publicly available PPB-Affinity dataset, which may finally help the industry in improving the screening efficiency of discovering new large-molecular drugs. We also developed a deep-learning benchmark model with this dataset to predict the PPB affinity, providing a foundational comparison for the research community.

## Download Data

You can download data from zenodo https://doi.org/10.5281/zenodo.11070824

### Benchmark File Tree

Files of the dataset are orginized as follows:
- PPB-Affinity.xlsx
- PDB/
  - Affinity Benchmark/
    - file1.pdb
    - file2.pdb
  - ATLAS/
  - PDBbindCN/
  - SAbDab/
  - SKEMPIv2.0/

## Baseline model

1. **Process data**

   ```
   python process_data.py
   ```

   After processing, the csv file "benchmark.csv" will be saved under the running directory

2. **Train the baseline model**

   ```
   python train.py --config ./baseline_train_config.yml
   ```

   After running the script, a folder "log_dir" will be generated, there are the checkpoint, log file and predict file(K-fold)