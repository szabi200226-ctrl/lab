# %%
#%load_ext autoreload
#autoreload 2

# %%

import numpy as np
import sys
import os
from pathlib import Path

project_root = Path(r"/mnt/d/Szabolcs/test/ceed/")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ceed.models.ceed import CEED
import matplotlib.pyplot as plt
from analysis.projections import learn_manifold_umap, pca_train, pca
import colorcet as cc
import torch

from mpl_toolkits.mplot3d import Axes3D

# %%
# example cell loading a 400 neuron, 200 spike MLP cell type model
celltype_test_data = '/mnt/d/Szabolcs/test/CEED/checkpoint/CEED/CEED/example_datasets/IBL_400neuron_200spike_dataset'


spikes_test = np.load(celltype_test_data + '/spikes_test.npy')[:,0]
labels_test = np.load(celltype_test_data + '/labels_test.npy')

# %% [markdown]
# ### Load a checkpoint into a CEED model

# %%
fc_celltype_ckpt_dir = '/mnt/d/Szabolcs/test/CEED/model_ckpts/400neur_200s_celltype_new'

fc_celltype_ceed_5d = CEED(num_extra_chans=0, out_dim=256, proj_dim=5)
fc_celltype_ceed_5d.load(fc_celltype_ckpt_dir)

# %% [markdown]
# ### Load and Transform

# %%
fc_celltype_ckpt_dir = '/mnt/d/Szabolcs/test/CEED/model_ckpts/400neur_200s_celltype_new››_celltype_fc_ckpt'
fc_transformed_inference_data, fc_inference_labels = fc_celltype_ceed_5d.load_and_transform(celltype_test_data, 
                                                                                            file_split='test')

# %%
#remove all zero spikes from dataset :(
vertical_offset = 0
fc_pca_ceed_emb_nonzero = []
labels_nonzero = []
spikes_nonzero = []
for i, unit_id in enumerate(np.unique(labels_test)):
    unit_ceed_emb = fc_transformed_inference_data[labels_test==unit_id]
    unit_spikes = spikes_test[labels_test==unit_id]
    unit_labels = labels_test[labels_test==unit_id]
    unit_ceed_emb = unit_ceed_emb[np.std(unit_spikes,1)>0]
    unit_labels = unit_labels[np.std(unit_spikes,1)>0]
    unit_spikes = unit_spikes[np.std(unit_spikes,1)>0]
    fc_pca_ceed_emb_nonzero.append(unit_ceed_emb)
    labels_nonzero.append(unit_labels)
    spikes_nonzero.append(unit_spikes)
fc_pca_ceed_emb_nonzero = np.concatenate(fc_pca_ceed_emb_nonzero)
labels_nonzero = np.concatenate(labels_nonzero)
spikes_nonzero = np.concatenate(spikes_nonzero)

fc_pca_ceed_emb, explained_var, fc_pca_ceed = pca(fc_pca_ceed_emb_nonzero, 2)
# fc_umap_ceed_emb = learn_manifold_umap(fc_pca_ceed_emb_nonzero, umap_dim=2)

# %%
embeddings = fc_pca_ceed_emb

fig, axes = plt.subplots(1,1, figsize=(8,8))
unit_ids = [1,2,11,45]
colors = cc.glasbey[:len(unit_ids)]
vertical_offset = 0
for i, unit_id in enumerate(unit_ids):
    unit_ceed_emb = embeddings[labels_nonzero==unit_id]
    unit_spikes = spikes_nonzero[labels_nonzero==unit_id]
    template = np.median(unit_spikes,0)
    template_emb = fc_celltype_ceed_5d.transform(torch.from_numpy(template).float()[None,None,:])
    pc_template_emb = fc_pca_ceed.transform(template_emb[:,None].T)[0]
    # axes[0].plot(unit_spikes.T + vertical_offset, color=colors[i], alpha=.01);
    # axes[0].plot(template.T + vertical_offset, color=colors[i], alpha=1);
    # axes[0].annotate(str(unit_id), xy=(0,vertical_offset+.3))
    vertical_offset += 1.5
    axes.scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.2)
    axes.scatter(pc_template_emb[0], pc_template_emb[1], color=colors[i], alpha=1,marker="^", s=200, label=str(unit_id))
# axes[0].vlines([42], ymax=vertical_offset, ymin= -1.5, ls='--', color='black')
axes.set_xticks([])
axes.set_yticks([])
# plt.legend();

# %%
embeddings = fc_pca_ceed_emb

fig, axes = plt.subplots(1,2, figsize=(12,4))
unit_ids = [8,5,12,17,68]
colors = cc.glasbey[:len(unit_ids)]
vertical_offset = 0
for i, unit_id in enumerate(unit_ids):
    unit_ceed_emb = embeddings[labels_nonzero==unit_id]
    unit_spikes = spikes_nonzero[labels_nonzero==unit_id]
    template = np.median(unit_spikes,0)
    template_emb = fc_celltype_ceed_5d.transform(torch.from_numpy(template).float()[None,None,:])
    pc_template_emb = fc_pca_ceed.transform(template_emb[:,None].T)[0]
    axes[0].plot(unit_spikes.T + vertical_offset, color=colors[i], alpha=.01);
    axes[0].plot(template.T + vertical_offset, color=colors[i], alpha=1);
    axes[0].annotate(str(unit_id), xy=(0,vertical_offset+.3))
    vertical_offset += 1.5
    axes[1].scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.1)
    axes[1].scatter(pc_template_emb[0], pc_template_emb[1], color=colors[i], alpha=1,marker="^", s=200, label=str(unit_id))
axes[0].vlines([42], ymax=vertical_offset, ymin= -1.5, ls='--', color='black')
plt.legend();

# %%
embeddings = fc_umap_ceed_emb

fig, axes = plt.subplots(1,2, figsize=(12,4))
unit_ids = [8,5,12,17,68]
colors = cc.glasbey[:len(unit_ids)]
vertical_offset = 0
for i, unit_id in enumerate(unit_ids):
    unit_ceed_emb = embeddings[labels_nonzero==unit_id]
    unit_spikes = spikes_nonzero[labels_nonzero==unit_id]
    template = np.median(unit_spikes,0)
    template_emb = fc_celltype_ceed_5d.transform(torch.from_numpy(template).float()[None,None,:])
    pc_template_emb = fc_pca_ceed.transform(template_emb[:,None].T)[0]
    axes[0].plot(unit_spikes.T + vertical_offset, color=colors[i], alpha=.01);
    axes[0].plot(template.T + vertical_offset, color=colors[i], alpha=1);
    axes[0].annotate(str(unit_id), xy=(0,vertical_offset+.3))
    vertical_offset += 1.5
    axes[1].scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.1)
    #plot median of umap here because didn't do projection of template
    axes[1].scatter(np.median(unit_ceed_emb,0)[0], np.median(unit_ceed_emb,0)[1], color=colors[i], alpha=1,marker='*', s=200, label=str(unit_id))
plt.legend();

# %% [markdown]
# ### Transform without a data folder

# %%
# same output as two cells above, but takes in actual data
cell_type_inference_data = np.load(os.path.join(celltype_test_data, 'spikes_test.npy'))
print("cell type data:", cell_type_inference_data.shape)
transformed_inference_data = fc_celltype_ceed_5d.transform(cell_type_inference_data)
print(transformed_inference_data.shape)


