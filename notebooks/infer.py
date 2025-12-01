# %%
#%load_ext autoreload
#autoreload 2

# %%
import sys
import os
from pathlib import Path
project_root = Path(r"/mnt/d/Szabolcs/test/ceed/")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ceed.models.model_simclr import FullyConnectedEnc
import numpy as np



# ...existing code...
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless environments (avoids Qt errors)
import matplotlib.pyplot as plt
# ...existing code...



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


spikes_test_full = np.load(celltype_test_data + '/spikes_test.npy')
print(f"Original spikes_test shape: {spikes_test_full.shape}")
spikes_test = spikes_test_full[:,0]  # Single channel for backwards compatibility
labels_test = np.load(celltype_test_data + '/labels_test.npy')

# %% [markdown]
# ### Load a checkpoint into a CEED model

# %%

fc_celltype_ckpt_dir = '/mnt/d/Szabolcs/test/CEED/model_ckpts/400neur_200s_celltype_new'

# Model was trained with single channel (121 input size)
fc_celltype_ceed_5d = CEED(num_extra_chans=0, out_dim=256, proj_dim=5)

model = FullyConnectedEnc(fc_depth=10, input_size=121*5)  # 5 channels?
print("Model keys:", list(model.state_dict().keys())[:10])
print("Has fcpart.7?", any("fcpart.7" in k for k in model.state_dict()))
fc_celltype_ceed_5d.load(fc_celltype_ckpt_dir)

fc_transformed_inference_data, fc_inference_labels = fc_celltype_ceed_5d.load_and_transform(celltype_test_data,  
                                                                                            file_split='test')

print(f"fc_transformed_inference_data shape after load_and_transform: {fc_transformed_inference_data.shape}")

# If the output is 3D (N, T, D), we likely need to use it as-is for PCA, or extract a specific dimension
# The model output seems to be (N, 21, 5) where 21 might be sequence length and 5 is the embedding dim

#remove all zero spikes from dataset :(
vertical_offset = 0
fc_pca_ceed_emb_nonzero = []
labels_nonzero = []
spikes_nonzero = []
#spikes_nonzero_full = []  # Keep full multi-channel data
for i, unit_id in enumerate(np.unique(labels_test)):
    unit_ceed_emb = fc_transformed_inference_data[labels_test==unit_id]
    unit_spikes = spikes_test[labels_test==unit_id]
    #unit_spikes_full = spikes_test_full[labels_test==unit_id]
    unit_labels = labels_test[labels_test==unit_id]
    unit_ceed_emb = unit_ceed_emb[np.std(unit_spikes,1)>0]
    unit_labels = unit_labels[np.std(unit_spikes,1)>0]
    #unit_spikes_full = unit_spikes_full[np.std(unit_spikes,1)>0]
    unit_spikes = unit_spikes[np.std(unit_spikes,1)>0]
    fc_pca_ceed_emb_nonzero.append(unit_ceed_emb)
    labels_nonzero.append(unit_labels)
    spikes_nonzero.append(unit_spikes)
    #spikes_nonzero_full.append(unit_spikes_full)
fc_pca_ceed_emb_nonzero = np.concatenate(fc_pca_ceed_emb_nonzero)
labels_nonzero = np.concatenate(labels_nonzero)
spikes_nonzero = np.concatenate(spikes_nonzero)
#spikes_nonzero_full = np.concatenate(spikes_nonzero_full)

# Debug: print shape to verify dimensionality
print(f"Shape of fc_pca_ceed_emb_nonzero before reshape: {fc_pca_ceed_emb_nonzero.shape}")
print(f"First element shape: {fc_transformed_inference_data[0].shape if len(fc_transformed_inference_data) > 0 else 'empty'}")

# Reshape to 2D for PCA (samples, features)
if len(fc_pca_ceed_emb_nonzero.shape) == 3:
    # If shape is (N, 1, D) or (N, D, 1), reshape to (N, D)
    if fc_pca_ceed_emb_nonzero.shape[1] == 1:
        fc_pca_ceed_emb_nonzero = fc_pca_ceed_emb_nonzero[:, 0, :]
    elif fc_pca_ceed_emb_nonzero.shape[2] == 1:
        fc_pca_ceed_emb_nonzero = fc_pca_ceed_emb_nonzero[:, :, 0]
    else:
        # Flatten the last two dimensions: (N, D1, D2) -> (N, D1*D2)
        fc_pca_ceed_emb_nonzero = fc_pca_ceed_emb_nonzero.reshape(fc_pca_ceed_emb_nonzero.shape[0], -1)
    print(f"After reshape: {fc_pca_ceed_emb_nonzero.shape}")
elif len(fc_pca_ceed_emb_nonzero.shape) == 1:
    # If 1D, reshape to (N, 1)
    fc_pca_ceed_emb_nonzero = fc_pca_ceed_emb_nonzero.reshape(-1, 1)
    print(f"After reshape from 1D: {fc_pca_ceed_emb_nonzero.shape}")

fc_pca_ceed_emb, explained_var, fc_pca_ceed = pca(fc_pca_ceed_emb_nonzero, 2)
fc_umap_ceed_emb = learn_manifold_umap(fc_pca_ceed_emb_nonzero, umap_dim=2)

# Print available unit IDs for reference
unique_labels = np.unique(labels_nonzero)
print(f"\nAvailable unit IDs ({len(unique_labels)} total): {unique_labels[:20]}...")  # Show first 20

 # %%
embeddings = fc_pca_ceed_emb

fig, axes = plt.subplots(1,1, figsize=(8,8))
unit_ids = [1,2,11,45]
colors = cc.glasbey[:len(unit_ids)]
vertical_offset = 0
for i, unit_id in enumerate(unit_ids):
    unit_ceed_emb = embeddings[labels_nonzero==unit_id]
    unit_spikes = spikes_nonzero[labels_nonzero==unit_id]
    #unit_spikes_full = spikes_nonzero_full[labels_nonzero==unit_id]
    
    # Skip if no data for this unit
    if len(unit_ceed_emb) == 0:
        print(f"Warning: No data found for unit {unit_id}, skipping")
        continue
    
    # Use full multi-channel data for template to match load_and_transform processing
    #template_full = np.median(unit_spikes_full,0)
    #print(f"Unit {unit_id}: template_full shape: {template_full.shape}")
    
    # Transform expects (N, C, L) format: N samples, C channels, L length
    #template_tensor = torch.from_numpy(template_full).float().unsqueeze(0)
    #print(f"Unit {unit_id}: template_tensor shape: {template_tensor.shape}")
    
    #template_emb = fc_celltype_ceed_5d.transform(template_tensor)
    
    # Convert to numpy if it's a torch tensor
    if isinstance(template_emb, torch.Tensor):
        template_emb = template_emb.cpu().numpy()
    
    print(f"Unit {unit_id}: template_emb shape: {template_emb.shape}, has NaN: {np.isnan(template_emb).any()}")
    if np.isnan(template_emb).any():
        print(f"  NaN locations: {np.where(np.isnan(template_emb))}")
        print(f"  template_full stats: min={template_full.min():.3f}, max={template_full.max():.3f}, mean={template_full.mean():.3f}")
    
    # Reshape template_emb to match PCA input dimensions (flatten to 2D)
    if len(template_emb.shape) == 3:
        # Flatten from (1, 21, 5) to (1, 105)
        template_emb = template_emb.reshape(template_emb.shape[0], -1)
        print(f"Unit {unit_id}: reshaped 3D -> {template_emb.shape}")
    elif len(template_emb.shape) == 2:
        # If it's 2D (e.g., (21, 5)), add batch dimension then flatten
        template_emb = template_emb.reshape(1, -1)
        print(f"Unit {unit_id}: reshaped 2D -> {template_emb.shape}")
    elif len(template_emb.shape) == 1:
        template_emb = template_emb.reshape(1, -1)
        print(f"Unit {unit_id}: reshaped 1D -> {template_emb.shape}")
    
    # Check for NaN - use median of unit's original embeddings and project through PCA
    if np.isnan(template_emb).any():
        print(f"Warning: NaN in template_emb for unit {unit_id}, using median of unit's embeddings")
        # Get the median of this unit's embeddings in the original high-dim space
        unit_emb_highdim = fc_pca_ceed_emb_nonzero[labels_nonzero==unit_id]
        median_emb_highdim = np.median(unit_emb_highdim, axis=0).reshape(1, -1)
        pc_template_emb = fc_pca_ceed.transform(median_emb_highdim)[0]
        print(f"  Using median of {len(unit_emb_highdim)} embeddings")
    else:
        pc_template_emb = fc_pca_ceed.transform(template_emb)[0]
        
    print(f"Unit {unit_id}: Marker at ({pc_template_emb[0]:.3f}, {pc_template_emb[1]:.3f}), cloud center: ({np.mean(unit_ceed_emb[:,0]):.3f}, {np.mean(unit_ceed_emb[:,1]):.3f})")
    
    # axes[0].plot(unit_spikes.T + vertical_offset, color=colors[i], alpha=.01);
    # axes[0].plot(template.T + vertical_offset, color=colors[i], alpha=1);
    # axes[0].annotate(str(unit_id), xy=(0,vertical_offset+.3))
    vertical_offset += 1.5
    axes.scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.2)
    axes.scatter(pc_template_emb[0], pc_template_emb[1], color=colors[i], alpha=1,marker="^", s=200, label=str(unit_id), edgecolors='black', linewidths=1)
# axes[0].vlines([42], ymax=vertical_offset, ymin= -1.5, ls='--', color='black')
axes.set_xticks([])
axes.set_yticks([])
# plt.legend();
plt.savefig('pca_embeddings_simple.png', dpi=150, bbox_inches='tight')
print('Saved: pca_embeddings_simple.png')

# %%
embeddings = fc_pca_ceed_emb

fig, axes = plt.subplots(1,2, figsize=(12,4))
unit_ids = [8,5,12,17,68]
colors = cc.glasbey[:len(unit_ids)]
vertical_offset = 0
for i, unit_id in enumerate(unit_ids):
    unit_ceed_emb = embeddings[labels_nonzero==unit_id]
    unit_spikes = spikes_nonzero[labels_nonzero==unit_id]
    #unit_spikes_full = spikes_nonzero_full[labels_nonzero==unit_id]
    
    # Use full multi-channel data for template to match load_and_transform processing
    #template_full = np.median(unit_spikes_full,0)
    template = np.median(unit_spikes,0)  # Keep single-channel for plotting
    
    # Transform expects (N, C, L) format: N samples, C channels, L length
    #template_tensor = torch.from_numpy(template_full).float().unsqueeze(0)
    template_emb = fc_celltype_ceed_5d.transform(template_tensor)
    
    # Convert to numpy if it's a torch tensor
    if isinstance(template_emb, torch.Tensor):
        template_emb = template_emb.cpu().numpy()
    
    # Reshape template_emb to match PCA input dimensions (flatten to 2D)
    if len(template_emb.shape) == 3:
        # Flatten from (1, 21, 5) to (1, 105)
        template_emb = template_emb.reshape(template_emb.shape[0], -1)
    elif len(template_emb.shape) == 2:
        # If it's 2D (e.g., (21, 5) or (1, 5)), flatten to (1, N*D)
        template_emb = template_emb.reshape(1, -1)
    elif len(template_emb.shape) == 1:
        template_emb = template_emb.reshape(1, -1)
    
    # Convert to numpy if it's a torch tensor
    if isinstance(template_emb, torch.Tensor):
        template_emb = template_emb.cpu().numpy()
    
    # Reshape to 2D if needed
    if len(template_emb.shape) == 3:
        template_emb = template_emb.reshape(template_emb.shape[0], -1)
    elif len(template_emb.shape) == 2 and template_emb.shape[0] != 1:
        template_emb = template_emb.reshape(1, -1)
    
    # Check for NaN - use median of unit's original embeddings and project through PCA
    if np.isnan(template_emb).any():
        print(f"Warning: NaN in template_emb for unit {unit_id}, using median of unit's embeddings")
        unit_emb_highdim = fc_pca_ceed_emb_nonzero[labels_nonzero==unit_id]
        median_emb_highdim = np.median(unit_emb_highdim, axis=0).reshape(1, -1)
        pc_template_emb = fc_pca_ceed.transform(median_emb_highdim)[0]
    else:
        pc_template_emb = fc_pca_ceed.transform(template_emb)[0]
        
    axes[0].plot(unit_spikes.T + vertical_offset, color=colors[i], alpha=.01);
    axes[0].plot(template.T + vertical_offset, color=colors[i], alpha=1);
    axes[0].annotate(str(unit_id), xy=(0,vertical_offset+.3))
    vertical_offset += 1.5
    axes[1].scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.1)
    axes[1].scatter(pc_template_emb[0], pc_template_emb[1], color=colors[i], alpha=1,marker="^", s=200, label=str(unit_id), edgecolors='black', linewidths=1)
axes[0].vlines([42], ymax=vertical_offset, ymin= -1.5, ls='--', color='black')
plt.legend();
plt.savefig('pca_embeddings_with_waveforms.png', dpi=150, bbox_inches='tight')
print('Saved: pca_embeddings_with_waveforms.png')

# %%
embeddings = fc_umap_ceed_emb

fig, axes = plt.subplots(1,2, figsize=(12,4))
unit_ids = [8,5,12,17,68]
colors = cc.glasbey[:len(unit_ids)]
vertical_offset = 0
for i, unit_id in enumerate(unit_ids):
    unit_ceed_emb = embeddings[labels_nonzero==unit_id]
    unit_spikes = spikes_nonzero[labels_nonzero==unit_id]
    #unit_spikes_full = spikes_nonzero_full[labels_nonzero==unit_id]
    
    # Skip if no data for this unit
    if len(unit_ceed_emb) == 0:
        print(f"Warning: No data found for unit {unit_id}, skipping")
        continue
    
    # Use full multi-channel data for template to match load_and_transform processing
    #template_full = np.median(unit_spikes_full,0)
    template = np.median(unit_spikes,0)  # Keep single-channel for plotting
    
    # Transform expects (N, C, L) format: N samples, C channels, L length
    template_tensor = torch.from_numpy(template_full).float().unsqueeze(0)
    template_emb = fc_celltype_ceed_5d.transform(template_tensor)
    
    # Convert to numpy if it's a torch tensor
    if isinstance(template_emb, torch.Tensor):
        template_emb = template_emb.cpu().numpy()
    
    # Reshape template_emb to match PCA input dimensions (flatten to 2D)
    if len(template_emb.shape) == 3:
        # Flatten from (1, 21, 5) to (1, 105)
        template_emb = template_emb.reshape(template_emb.shape[0], -1)
    elif len(template_emb.shape) == 2:
        # If it's 2D (e.g., (21, 5) or (1, 5)), flatten to (1, N*D)
        template_emb = template_emb.reshape(1, -1)
    elif len(template_emb.shape) == 1:
        template_emb = template_emb.reshape(1, -1)
    
    # For UMAP, just use median of the embeddings (can't project template through UMAP after fitting)
    axes[0].plot(unit_spikes.T + vertical_offset, color=colors[i], alpha=.01);
    axes[0].plot(template.T + vertical_offset, color=colors[i], alpha=1);
    axes[0].annotate(str(unit_id), xy=(0,vertical_offset+.3))
    vertical_offset += 1.5
    axes[1].scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.1)
    #plot median of umap here because didn't do projection of template
    axes[1].scatter(np.median(unit_ceed_emb,0)[0], np.median(unit_ceed_emb,0)[1], color=colors[i], alpha=1,marker='*', s=200, label=str(unit_id), edgecolors='black', linewidths=1)
plt.legend();
plt.savefig('umap_embeddings_with_waveforms.png', dpi=150, bbox_inches='tight')
print('Saved: umap_embeddings_with_waveforms.png')

# %% [markdown]
# ### Transform without a data folder

# %%
# same output as two cells above, but takes in actual data
cell_type_inference_data = np.load(os.path.join(celltype_test_data, 'spikes_test.npy'))
print("cell type data:", cell_type_inference_data.shape)
transformed_inference_data = fc_celltype_ceed_5d.transform(cell_type_inference_data)
print(transformed_inference_data.shape)


