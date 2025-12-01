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


spikes_test = np.load(celltype_test_data + '/spikes_test.npy')[:,0]
labels_test = np.load(celltype_test_data + '/labels_test.npy')
print(f"Loaded single-channel spikes_test shape: {spikes_test.shape}")


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
#unit_ids = [1,2,11,45]
unit_ids = [2,11,12,45,50]
colors = cc.glasbey[:len(unit_ids)]
vertical_offset = 0
for i, unit_id in enumerate(unit_ids):
    unit_ceed_emb = embeddings[labels_nonzero==unit_id]
    unit_spikes = spikes_nonzero[labels_nonzero==unit_id]
    
    if len(unit_ceed_emb) == 0:
        print(f"Warning: No data found for unit {unit_id}, skipping")
        continue
    
    # Use single-channel template (shape: (121,))
    template = np.median(unit_spikes, 0)
    print(f"Unit {unit_id}: template shape: {template.shape}")
    
    # Transform expects (N, 1, L) for single-channel: (batch, channel, length)
    template_tensor = torch.from_numpy(template).float().unsqueeze(0).unsqueeze(0)
    print(f"Unit {unit_id}: template_tensor shape: {template_tensor.shape}")
    
    template_emb = fc_celltype_ceed_5d.transform(template_tensor)
    
    if isinstance(template_emb, torch.Tensor):
        template_emb = template_emb.cpu().numpy()
    
    print(f"Unit {unit_id}: template_emb shape: {template_emb.shape}, has NaN: {np.isnan(template_emb).any()}")
    
    # The transform may output (T, D) or (1, T, D) or (D,). We need (1, T*D) for PCA.
    # Match what was done for fc_pca_ceed_emb_nonzero (which was reshaped to (N, 105))
    original_shape = template_emb.shape
    if len(template_emb.shape) == 3:
        # (1, T, D) -> flatten to (1, T*D)
        template_emb = template_emb.reshape(1, -1)
    elif len(template_emb.shape) == 2:
        # (T, D) -> flatten to (1, T*D)
        template_emb = template_emb.reshape(1, -1)
    elif len(template_emb.shape) == 1:
        # (D,) -> (1, D) - but this may still be wrong if it should be (1, T*D)
        # Check if this matches PCA input dim
        template_emb = template_emb.reshape(1, -1)
    
    print(f"Unit {unit_id}: after reshape {original_shape} -> {template_emb.shape}")
    
    # Verify dimensions match PCA expectations
    expected_features = fc_pca_ceed.n_features_in_
    if template_emb.shape[1] != expected_features:
        print(f"Warning: template_emb has {template_emb.shape[1]} features but PCA expects {expected_features}")
        print(f"This suggests the single-template transform output differs from batch transform.")
        print(f"Falling back to median of unit embeddings.")
        # Fallback: use median of unit's high-dim embeddings
        unit_emb_highdim = fc_pca_ceed_emb_nonzero[labels_nonzero==unit_id]
        template_emb = np.median(unit_emb_highdim, axis=0).reshape(1, -1)

    if np.isnan(template_emb).any():
        print(f"Warning: NaN in template_emb for unit {unit_id}, using median of unit's embeddings")
        unit_emb_highdim = fc_pca_ceed_emb_nonzero[labels_nonzero==unit_id]
        median_emb_highdim = np.median(unit_emb_highdim, axis=0).reshape(1, -1)
        pc_template_emb = fc_pca_ceed.transform(median_emb_highdim)[0]
    else:
        pc_template_emb = fc_pca_ceed.transform(template_emb)[0]
        
    print(f"Unit {unit_id}: Marker at ({pc_template_emb[0]:.3f}, {pc_template_emb[1]:.3f})")
    
    axes.scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.2)
    axes.scatter(pc_template_emb[0], pc_template_emb[1], color=colors[i], alpha=1,
                marker="^", s=200, label=str(unit_id), edgecolors='black', linewidths=1)



axes.set_xticks([])
axes.set_yticks([])
plt.savefig('pca_embeddings_simple.png', dpi=150, bbox_inches='tight')
print('Saved: pca_embeddings_simple.png')



# ...existing code...

# %%
embeddings = fc_pca_ceed_emb

fig, axes = plt.subplots(1,2, figsize=(12,4))
unit_ids = [2,11,12,45,50]
colors = cc.glasbey[:len(unit_ids)]
vertical_offset = 0
for i, unit_id in enumerate(unit_ids):
    unit_ceed_emb = embeddings[labels_nonzero==unit_id]
    unit_spikes = spikes_nonzero[labels_nonzero==unit_id]
    
    if len(unit_ceed_emb) == 0:
        print(f"Warning: No data found for unit {unit_id}, skipping")
        continue
    
    # Use single-channel template
    template = np.median(unit_spikes, 0)
    
    # Transform expects (N, 1, L) for single-channel
    template_tensor = torch.from_numpy(template).float().unsqueeze(0).unsqueeze(0)
    template_emb = fc_celltype_ceed_5d.transform(template_tensor)
    
    if isinstance(template_emb, torch.Tensor):
        template_emb = template_emb.cpu().numpy()
    
    # Reshape to 2D
    original_shape = template_emb.shape
    if len(template_emb.shape) == 3:
        template_emb = template_emb.reshape(1, -1)
    elif len(template_emb.shape) == 2:
        template_emb = template_emb.reshape(1, -1)
    elif len(template_emb.shape) == 1:
        template_emb = template_emb.reshape(1, -1)
    
    print(f"Unit {unit_id}: after reshape {original_shape} -> {template_emb.shape}")
    
    # ADD THIS DIMENSION CHECK (same as first cell):
    expected_features = fc_pca_ceed.n_features_in_
    if template_emb.shape[1] != expected_features:
        print(f"Warning: template_emb has {template_emb.shape[1]} features but PCA expects {expected_features}")
        print(f"Falling back to median of unit embeddings.")
        unit_emb_highdim = fc_pca_ceed_emb_nonzero[labels_nonzero==unit_id]
        template_emb = np.median(unit_emb_highdim, axis=0).reshape(1, -1)
    
    # Handle NaN
    if np.isnan(template_emb).any():
        print(f"Warning: NaN in template_emb for unit {unit_id}, using median")
        unit_emb_highdim = fc_pca_ceed_emb_nonzero[labels_nonzero==unit_id]
        median_emb_highdim = np.median(unit_emb_highdim, axis=0).reshape(1, -1)
        pc_template_emb = fc_pca_ceed.transform(median_emb_highdim)[0]
    else:
        pc_template_emb = fc_pca_ceed.transform(template_emb)[0]
        
    axes[0].plot(unit_spikes.T + vertical_offset, color=colors[i], alpha=.01)
    axes[0].plot(template.T + vertical_offset, color=colors[i], alpha=1)
    axes[0].annotate(str(unit_id), xy=(0,vertical_offset+.3))
    vertical_offset += 1.5
    axes[1].scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.1)
    axes[1].scatter(pc_template_emb[0], pc_template_emb[1], color=colors[i], alpha=1,
                   marker="^", s=200, label=str(unit_id), edgecolors='black', linewidths=1)

axes[0].vlines([42], ymax=vertical_offset, ymin=-1.5, ls='--', color='black', alpha=0.5, linewidth=1)
axes[0].set_xlabel('Sample (timepoint)', fontsize=11)
axes[0].set_ylabel('Amplitude (offset per unit)', fontsize=11)
axes[0].set_title('Spike Waveforms', fontsize=12)
axes[0].set_ylim(-1.5, vertical_offset)  # Set y limits for better view
axes[0].grid(alpha=0.3, axis='x')

# Adjust right panel (embeddings)
axes[1].set_xlabel('PC1', fontsize=11)
axes[1].set_ylabel('PC2', fontsize=11)
axes[1].set_title('PCA Embeddings', fontsize=12)
axes[1].grid(alpha=0.3)

plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('pca_embeddings_with_waveforms.png', dpi=150, bbox_inches='tight')
print('Saved: pca_embeddings_with_waveforms.png')


embeddings = fc_umap_ceed_emb

fig, axes = plt.subplots(1,2, figsize=(12,4))
unit_ids = [2,11,12,45,50]
colors = cc.glasbey[:len(unit_ids)]
vertical_offset = 0
for i, unit_id in enumerate(unit_ids):
    unit_ceed_emb = embeddings[labels_nonzero==unit_id]
    unit_spikes = spikes_nonzero[labels_nonzero==unit_id]
    
    # Skip if no data for this unit
    if len(unit_ceed_emb) == 0:
        print(f"Warning: No data found for unit {unit_id}, skipping")
        continue
    
    # Use single-channel template (shape: (121,))
    template = np.median(unit_spikes, 0)
    
    # Plot waveforms
    axes[0].plot(unit_spikes.T + vertical_offset, color=colors[i], alpha=.01)
    axes[0].plot(template.T + vertical_offset, color=colors[i], alpha=1)
    axes[0].annotate(str(unit_id), xy=(0,vertical_offset+.3))
    vertical_offset += 1.5
    
    # For UMAP, use median of embeddings (can't project new samples through fitted UMAP)
    axes[1].scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.1)
    median_emb = np.median(unit_ceed_emb, 0)
    axes[1].scatter(median_emb[0], median_emb[1], color=colors[i], alpha=1,
                   marker='*', s=200, label=str(unit_id), 
                   edgecolors='black', linewidths=1)

axes[0].vlines([42], ymax=vertical_offset, ymin=-1.5, ls='--', color='black', alpha=0.5, linewidth=1)
axes[0].set_xlabel('Sample (timepoint)', fontsize=11)
axes[0].set_ylabel('Amplitude (offset per unit)', fontsize=11)
axes[0].set_title('Spike Waveforms', fontsize=12)
axes[0].set_ylim(-1.5, vertical_offset)
axes[0].grid(alpha=0.3, axis='x')

# Adjust right panel
axes[1].set_xlabel('UMAP1', fontsize=11)
axes[1].set_ylabel('UMAP2', fontsize=11)
axes[1].set_title('UMAP Embeddings', fontsize=12)
axes[1].grid(alpha=0.3)

plt.legend(loc='best', fontsize=10)
plt.tight_layout()
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

