<!-- #region -->
# PPB-Affinity

[PPB-Affinity: Protein-Protein Binding Affinity dataset for AI-based protein drug discovery](https://www.nature.com/articles/s41597-024-03997-4)

Prediction of protein-protein binding (PPB) affinity plays an important role in large-molecular drug discovery. Deep learning (DL) has been adopted to predict the change of PPB binding affinity upon mutation, but there was a scarcity of studies predicting the PPB affinity itself. The major reason is the paucity of open-source dataset concerning PPB affinity. Therefore, the current study aimed to introduce and disclose a PPB affinity dataset (PPB-Affinity), which will definitely benefit the development of applicable DL to predict the PPB affinity. The PPB-Affinity dataset contains key information such as crystal structures of protein-protein complexes (with or without protein mutation patterns), PPB affinity, receptor protein chain, ligand protein chain, etc. To the best of our knowledge, this is the largest and publicly available PPB-Affinity dataset, which may finally help the industry in improving the screening efficiency of discovering new large-molecular drugs. We also developed a deep-learning benchmark model with this dataset to predict the PPB affinity, providing a foundational comparison for the research community.

## Download Data and Trained Weight

You can download dataset from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.11070823).

You can download trained weight from [Hugging Face](https://huggingface.co/ChenPy00/PPB-Affinity).

### Benchmark File Tree

Files of the download dataset are orginized as follows:
- PPB-Affinity.xlsx
- samples_deleted.zip
- PPB-Affinity-AF.zip
- PDB/
  - Affinity Benchmark v5.5/
    - file1.pdb
    - ...
    - file2.pdb
  - ATLAS/
  - PDBbind v2020/
  - SAbDab/
  - SKEMPI v2.0/


## Baseline model

0. **Prepare environment**

   ```
   conda env create -f environment.yaml -n PPB-Affinity
   conda activate PPB-Affinity
   ```

1. **Process data**

   ```
   python process_data.py
   ```

   In this step, the affinity data in the benchmark will be reorganized into a format suitable for the baseline model, including extracting PDB IDs, data sources, mutation representations (such as RC89P, where R on the C-chain mutates into P), receptor chain IDs, ligand chain IDs, dG values, subgroups, and PDB storage paths. In addition, samples with more than 10 chains will be excluded.

   After processing, the CSV file "benchmark.csv" will be saved in the running directory as follows:

   | pdb  |   source   | mutstr | ligand  | receptor |    dG     | Subgroup |  pdb_path   |
   | :--: | :--------: | :----: | :-----: | :------: | :-------: | :------: | :---------: |
   | 3QIB | SKEMPIv2.0 |  NaN   | A, B, P |   C, D   | -6.634345 | TCR-pMHC | xxx/xxx.pdb |
   | 3QIB | SKEMPIv2.0 | RC89P  | A, B, P |   C, D   | -6.753714 | TCR-pMHC | xxx/xxx.pdb |
   | 3QIB | SKEMPIv2.0 | NC94P  | A, B, P |   C, D   | -5.847532 | TCR-pMHC | xxx/xxx.pdb |

2. **Train the baseline model**

  	Perform five-fold cross-validation on the baseline model using the PPB-Affinity dataset, with data split based on PDB codes.

   ```
   python train.py \
   	--config ./baseline_train_config.yml \
   	--num_workers 4 \
   	--device 'cuda'
   ```

   	After running the script, a folder "log_dir" will be generated, there are the checkpoint, log file and predict file(K-fold)

3. **Run draw.ipynb to draw scatter plots of cross validation.**
    
    You can use the `result.csv` file in the `checkpoints` folder we provided(please see *Download Data and Trained Weight* above) to draw the same picture as in the `CrossValidationFigs` folder

4. **Infer from pdb**

   We have provided an example of inference, please refer to "inference.py" for details and you can complete a simple inference by entering the following command on the terminal:

   ```
   python inference.py \
   	--affinity_file_path ./example_data/example.xlsx \
   	--checkpoint_path ./xxx/checkpoints/xxxxx.pt \
   	--save_path ./example_data
   	--device 'cuda'
   ```

## How do we preprocess the source data-set to get PPB-Affinity？

We provide a code how to preprocess the source dataset to obtain PPB-Affinity. Please visit https://github.com/Huatsing-Lau/PPB-Affinity-DataPrepWorkflow

## Citations
```bibtex
@article{PPB-Affinity2024,
   author = {Liu, Huaqing and Chen, Peiyi and Zhai, Xiaochen and Huo, Ku-Geng and Zhou, Shuxian and Han, Lanqing and Fan, Guoxin},
   title = {PPB-Affinity: Protein-Protein Binding Affinity dataset for AI-based protein drug discovery},
   journal = {Scientific Data},
   volume = {11},
   number = {1},
   pages = {1316},
   ISSN = {2052-4463},
   DOI = {10.1038/s41597-024-03997-4},
   url = {https://doi.org/10.1038/s41597-024-03997-4},
   year = {2024},
   type = {Journal Article}
}
````
<!-- #endregion -->
