# %%
#%load_ext autoreload
#autoreload 2

# %%

import numpy as np
import sys
import os
from pathlib import Path

# ...existing code...
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless environments (avoids Qt errors)
import matplotlib.pyplot as plt
# ...existing code...

project_root = Path(r"/mnt/d/Szabolcs/test/ceed/")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ceed.models.ceed import CEED
import matplotlib.pyplot as plt
from analysis.projections import learn_manifold_umap, pca_train, pca
import colorcet as cc
import torch
import re

from mpl_toolkits.mplot3d import Axes3D

# %%
# example cell loading a 400 neuron, 200 spike MLP cell type model
celltype_test_data = '/mnt/d/Szabolcs/test/CEED/checkpoint/CEED/CEED/example_datasets/IBL_400neuron_200spike_dataset'


spikes_test = np.load(celltype_test_data + '/spikes_test.npy')[:,0]
labels_test = np.load(celltype_test_data + '/labels_test.npy')

# %% [markdown]
# ### Load a checkpoint into a CEED model

# %%
# ...existing code...
fc_celltype_ckpt = '/mnt/d/Szabolcs/test/CEED/model_ckpts/400neur_200s_celltype_new/checkpoint.pth'
fc_celltype_ckpt_dir = '/mnt/d/Szabolcs/test/CEED/model_ckpts/400neur_200s_celltype_new'

# Instantiate CEED with the same hyperparameters used in the original notebook/checkpoint
# (original notebook used out_dim=5 and proj_dim=5)
# Instantiate CEED with the hyperparameters that match the checkpoint
fc_celltype_ceed_5d = CEED(num_extra_chans=0, out_dim=256, proj_dim=5)

# ...existing code...
import torch

def _extract_state_dict_from_ckpt(ckpt):
    # Accept several common checkpoint shapes
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "backbone_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # if dict looks like a state_dict already (string keys -> tensors)
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Couldn't find a state_dict in checkpoint")

def _remap_by_shape(ckpt_sd, model_sd):
    patched = dict(ckpt_sd)  # copy
    model_shapes = {k: v.shape for k, v in model_sd.items()}
    ckpt_shapes = {k: v.shape for k, v in ckpt_sd.items()}
    missing = [k for k in model_shapes.keys() if k not in ckpt_shapes]
    unexpected = [k for k in ckpt_shapes.keys() if k not in model_shapes]
    remap = {}
    used = set()
    for m in missing:
        tgt_shape = tuple(model_shapes[m])
        for u in unexpected:
            if u in used:
                continue
            if tuple(ckpt_shapes[u]) == tgt_shape:
                remap[u] = m
                used.add(u)
                break
    for old, new in remap.items():
        patched[new] = patched.pop(old)
    return patched, remap

# Replace the CEED.load(...) try/except with this robust loader:
try:
    fc_celltype_ceed_5d.load(fc_celltype_ckpt_dir)
    print("loaded checkpoint via CEED.load()")
except Exception as e:
    print("CEED.load() failed:", e)
    print("Attempting manual checkpoint load + safe remapping...")
    ckpt = torch.load(fc_celltype_ckpt, map_location="cpu")
    try:
        sd = _extract_state_dict_from_ckpt(ckpt)
    except ValueError:
        # fallback: if checkpoint file is simply a dict of tensors use it directly
        sd = ckpt if isinstance(ckpt, dict) else {}
    # pick the target state dict from the instantiated model (backbone if present)
    target_module = getattr(fc_celltype_ceed_5d, "model", fc_celltype_ceed_5d)
    target_state = None
    for attr in ("backbone", "net", "encoder"):
        if hasattr(target_module, attr):
            target_state = getattr(target_module, attr).state_dict()
            target_obj = getattr(target_module, attr)
            break
    if target_state is None:
        target_state = fc_celltype_ceed_5d.state_dict()
        target_obj = fc_celltype_ceed_5d

    patched, remap = _remap_by_shape(sd, target_state)
    if remap:
        print("Applied remap (checkpoint_key -> model_key):")
        for o, n in remap.items():
            print(f"  {o} -> {n}")
    else:
        print("No remapping performed (no matching shapes found)")

    # try strict load first, then fallback to strict=False
    try:
        target_obj.load_state_dict(patched)
        print("Loaded patched state_dict with strict=True")
    except Exception as e2:
        print("Strict load failed:", e2)
        target_obj.load_state_dict(patched, strict=False)
        print("Loaded patched state_dict with strict=False")



#fc_celltype_ceed_5d.load(fc_celltype_ckpt_dir)


# %%
cell_data = np.load(os.path.join(celltype_test_data, 'spikes_test.npy'))
print("DEBUG: cell_data.shape:", cell_data.shape)
# Use transform() directly (no re-loading). transform should accept the raw array.
fc_transformed_inference_data = fc_celltype_ceed_5d.transform(cell_data)
fc_inference_labels = np.load(os.path.join(celltype_test_data, 'labels_test.npy'))

# %%
#remove all zero spikes from dataset :(
vertical_offset = 0
fc_pca_ceed_emb_nonzero = []
labels_nonzero = []
spikes_nonzero = []
for i, unit_id in enumerate(np.unique(fc_inference_labels)):
    unit_ceed_emb = fc_transformed_inference_data[fc_inference_labels==unit_id]
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

# ...existing code...

 # ensure embeddings are 2D (n_samples, n_features) for PCA
# ...existing code...
# ensure embeddings are 2D (n_samples, n_features) for PCA
arr = fc_pca_ceed_emb_nonzero
print("DEBUG: embeddings before PCA, shape:", getattr(arr, "shape", None))
if arr.ndim == 3:
    # Option A (recommended): average across the middle axis (temporal/channel aggregation)
    arr2 = arr.mean(axis=1)         # result shape (n_samples, features)
    # Option B (alternative): flatten per-sample
    # arr2 = arr.reshape(arr.shape[0], -1)
elif arr.ndim == 2:
    arr2 = arr
else:
    raise ValueError(f"Unexpected embeddings ndim={arr.ndim}; PCA requires 2D array")

fc_pca_ceed_emb, explained_var, fc_pca_ceed = pca(arr2, 2)
# ...existing code...
# ...existing code...
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


