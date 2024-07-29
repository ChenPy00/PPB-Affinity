import numpy as np
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt
import re
import os

def rmse_score(x,y):
    return np.sqrt(np.mean((x-y)**2))

save_dir = "./results"
s = 5
alpha = 1#0.5
edgecolor = 'black'
c = 'orange'
linewidths = 0.5

all_df = pd.read_csv('./benchmark.csv', index_col=0)

sabdab_df = all_df[(all_df['source']=='SAbDab')]
skempi_df = all_df[(all_df['source']=='SKEMPI v2.0')]
pdbbind_df = all_df[(all_df['source']=='PDBbind v2020')]
atlas_df = all_df[(all_df['source']=='ATLAS')]
ab_df = all_df[(all_df['source']=='Affinity Benchmark v5.5')]

log_path = './log_dir/baseline_train_config_2024_07_26__18_06_00/'
pred_result = pd.read_csv(os.path.join(log_path, 'checkpoints/results_46000.csv'))  #  the prediction file of the 12000iteration

pred_sabdab = pred_result[pred_result['complex'].isin(sabdab_df.pdb.tolist())]
pred_pdbbind = pred_result[pred_result['complex'].isin(pdbbind_df.pdb.tolist())]
pred_skempi = pred_result[(pred_result['complex'].isin(skempi_df.pdb.tolist())) | (pred_result['complex'].isin([s.lower() for s in skempi_df.pdb.tolist()]))]
pred_atlas = pred_result[pred_result['complex'].isin(atlas_df.pdb.tolist())]
pred_ab = pred_result[pred_result['complex'].isin(ab_df.pdb.tolist())]

TCR_df = all_df[all_df['Subgroup']=='TCR-pMHC']
AB_df = all_df[all_df['Subgroup']=='Antibody-Antigen']
pred_TCR = pred_result[pred_result['complex'].isin(TCR_df.pdb.tolist())]
pred_AB = pred_result[pred_result['complex'].isin(AB_df.pdb.tolist())]

# Prediction scatter plot of SKEMPIv2.0
pearson_all = pred_skempi[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
spearman_all = pred_skempi[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
rmse_all = rmse_score(pred_skempi['dG'],pred_skempi['dG_pred'])
plt.figure(figsize=(3,3),dpi=300)
plt.scatter(pred_skempi.dG.tolist(), pred_skempi.dG_pred.tolist(), s=s, alpha=alpha, edgecolor=edgecolor, c=c, linewidths=linewidths)
plt.title(f'PCC:{pearson_all:.3f}, SRCC:{spearman_all:.3f}, RMSE:{rmse_all:.3F}')
# plt.gca().set_aspect('equal')
plt.xlabel('Experiment')
plt.ylabel('AI prediction')
plt.savefig( os.path.join(save_dir,'./SKEMPI v2.0.tiff') )
plt.show()

# Prediction scatter plot of Sabdab
pearson_all = pred_sabdab[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
spearman_all = pred_sabdab[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
rmse_all = rmse_score(pred_sabdab['dG'],pred_sabdab['dG_pred'])
plt.figure(figsize=(4,4),dpi=300)
plt.scatter(pred_sabdab.dG.tolist(), pred_sabdab.dG_pred.tolist(), s=s, alpha=alpha, edgecolor=edgecolor, c=c, linewidths=linewidths)
plt.title(f'PCC:{pearson_all:.3f}, SRCC:{spearman_all:.3f}, RMSE:{rmse_all:.3F}')
# plt.gca().set_aspect('equal')
plt.xlabel('Experiment')
plt.ylabel('AI prediction')
plt.savefig( os.path.join(save_dir,'./SAbDab.tiff') )
plt.show()

# Prediction scatter plot of PDBbind
pearson_all = pred_pdbbind[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
spearman_all = pred_pdbbind[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
rmse_all = rmse_score(pred_pdbbind['dG'],pred_pdbbind['dG_pred'])
plt.figure(figsize=(4,4),dpi=300)
plt.scatter(pred_pdbbind.dG.tolist(), pred_pdbbind.dG_pred.tolist(), s=s, alpha=alpha, edgecolor=edgecolor, c=c, linewidths=linewidths)
plt.title(f'PCC:{pearson_all:.3f}, SRCC:{spearman_all:.3f}, RMSE:{rmse_all:.3F}')
# plt.gca().set_aspect('equal')
plt.xlabel('Experiment')
plt.ylabel('AI prediction')
plt.savefig( os.path.join(save_dir,'./PDBBind v2020.tiff') )
plt.show()

# Prediction scatter plot of atlas
pearson_all = pred_atlas[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
spearman_all = pred_atlas[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
rmse_all = rmse_score(pred_atlas['dG'],pred_atlas['dG_pred'])
plt.figure(figsize=(4,4),dpi=300)
plt.scatter(pred_atlas.dG.tolist(), pred_atlas.dG_pred.tolist(), s=s, alpha=alpha, edgecolor=edgecolor, c=c, linewidths=linewidths)
plt.title(f'PCC:{pearson_all:.3f}, SRCC:{spearman_all:.3f}, RMSE:{rmse_all:.3F}')
# plt.gca().set_aspect('equal')
plt.xlabel('Experiment')
plt.ylabel('AI prediction')
plt.savefig( os.path.join(save_dir,'./ATLAS.tiff') )
plt.show()

# Prediction scatter plot of AB
pearson_all = pred_ab[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
spearman_all = pred_ab[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
rmse_all = rmse_score(pred_ab['dG'],pred_ab['dG_pred'])
plt.figure(figsize=(4,4),dpi=300)
plt.scatter(pred_ab.dG.tolist(), pred_ab.dG_pred.tolist(), s=s, alpha=alpha, edgecolor=edgecolor, c=c, linewidths=linewidths)
plt.title(f'PCC:{pearson_all:.3f}, SRCC:{spearman_all:.3f}, RMSE:{rmse_all:.3F}')
# plt.gca().set_aspect('equal')
plt.xlabel('Experiment')
plt.ylabel('AI prediction')
plt.savefig( os.path.join(save_dir,'./Affinity Benchmark v5.5.tiff') )
plt.show()

# Prediction scatter plot of whole dataset
pearson_all = pred_result[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
spearman_all = pred_result[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
rmse_all = rmse_score(pred_result['dG'],pred_result['dG_pred'])
plt.figure(figsize=(4,4),dpi=300)
plt.scatter(pred_result.dG.tolist(), pred_result.dG_pred.tolist(), s=s, alpha=alpha, edgecolor=edgecolor, c=c, linewidths=linewidths)
plt.title(f'PCC:{pearson_all:.3f}, SRCC:{spearman_all:.3f}, RMSE:{rmse_all:.3F}')
# plt.gca().set_aspect('equal')
plt.xlabel('Experiment')
plt.ylabel('AI prediction')
plt.savefig( os.path.join(save_dir,'./PPB-Affinity.tiff') )
plt.show()

# Prediction scatter plot of TCR
pearson_all = pred_TCR[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
spearman_all = pred_TCR[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
rmse_all = rmse_score(pred_TCR['dG'],pred_TCR['dG_pred'])
plt.figure(figsize=(4,4),dpi=300)
plt.scatter(pred_TCR.dG.tolist(), pred_TCR.dG_pred.tolist(), s=s, alpha=alpha, edgecolor=edgecolor, c=c, linewidths=linewidths)
plt.title(f'PCC:{pearson_all:.3f}, SRCC:{spearman_all:.3f}, RMSE:{rmse_all:.3F}')
# plt.gca().set_aspect('equal')
plt.xlabel('Experiment')
plt.ylabel('AI prediction')
plt.savefig( os.path.join(save_dir,'./TCR-pMHC.tiff') )
plt.show()

# Prediction scatter plot of Antibody
pearson_all = pred_AB[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
spearman_all = pred_AB[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
rmse_all = rmse_score(pred_AB['dG'],pred_AB['dG_pred'])
plt.figure(figsize=(4,4),dpi=300)
plt.scatter(pred_AB.dG.tolist(), pred_AB.dG_pred.tolist(), s=s, alpha=alpha, edgecolor=edgecolor, c=c, linewidths=linewidths)
plt.title(f'PCC:{pearson_all:.3f}, SRCC:{spearman_all:.3f}, RMSE:{rmse_all:.3F}')
# plt.gca().set_aspect('equal')
plt.xlabel('Experiment')
plt.ylabel('AI prediction')
plt.savefig( os.path.join(save_dir,'./Antibody-Antigen.tiff') )
plt.show()


