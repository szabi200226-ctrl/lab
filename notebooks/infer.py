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
    
    axes.scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.3)
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
    axes[1].scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.3)
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
    axes[1].scatter(unit_ceed_emb[:,0], unit_ceed_emb[:,1], color=colors[i], alpha=.3)
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


# same output as two cells above, but takes in actual data
cell_type_inference_data = np.load(os.path.join(celltype_test_data, 'spikes_test.npy'))
print("cell type data:", cell_type_inference_data.shape)
transformed_inference_data = fc_celltype_ceed_5d.transform(cell_type_inference_data)
print(transformed_inference_data.shape)


# ...existing code...

# %%
# Automatic cell type discovery using GMM clustering (CEED's approach)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

print("\n" + "="*60)
print("AUTOMATIC CELL TYPE DISCOVERY WITH GMM")
print("="*60)

# Use the 5D embeddings (before PCA) for clustering
embeddings_5d = fc_pca_ceed_emb_nonzero  # Shape: (N, 105) after flattening

# Scan 1-10 clusters and select best by BIC
n_clusters_range = range(2, 11)
bic_scores = []
silhouette_scores = []
gmm_models = []

for n_clusters in n_clusters_range:
    gmm = GaussianMixture(n_components=n_clusters, 
                          covariance_type='full',
                          random_state=42,
                          n_init=10)
    gmm.fit(embeddings_5d)
    bic = gmm.bic(embeddings_5d)
    bic_scores.append(bic)
    gmm_models.append(gmm)
    
    # Silhouette score (alternative metric)
    labels_pred = gmm.predict(embeddings_5d)
    sil_score = silhouette_score(embeddings_5d, labels_pred)
    silhouette_scores.append(sil_score)
    
    print(f"n_clusters={n_clusters}: BIC={bic:.1f}, Silhouette={sil_score:.3f}")

# Find elbow in BIC curve
bic_scores = np.array(bic_scores)
bic_diffs = np.diff(bic_scores)
bic_diffs2 = np.diff(bic_diffs)  # Second derivative
elbow_idx = np.argmax(bic_diffs2) + 1  # +1 for diff offset, +1 for range start
n_clusters_elbow = list(n_clusters_range)[elbow_idx]

# Use minimum BIC as final choice
n_clusters_best = list(n_clusters_range)[np.argmin(bic_scores)]

print(f"\n*** Best by BIC minimum: {n_clusters_best} clusters ***")
print(f"*** Best by BIC elbow: {n_clusters_elbow} clusters ***")

# Use BIC minimum (standard approach)
best_gmm = gmm_models[np.argmin(bic_scores)]
celltype_labels_gmm = best_gmm.predict(embeddings_5d)

# Create cell type names based on waveform characteristics
celltype_names_gmm = {}
for ct_id in np.unique(celltype_labels_gmm):
    ct_spikes = spikes_nonzero[celltype_labels_gmm == ct_id]
    ct_template = np.median(ct_spikes, 0)
    
    # Calculate spike width (trough to peak)
    trough_idx = np.argmin(ct_template)
    peak_idx = np.argmax(ct_template)
    spike_width = abs(peak_idx - trough_idx)
    
    # Classify as narrow or broad
    if spike_width < 15:
        celltype_names_gmm[ct_id] = f'Type {ct_id}: Narrow-spiking'
    elif spike_width < 25:
        celltype_names_gmm[ct_id] = f'Type {ct_id}: Medium-spiking'
    else:
        celltype_names_gmm[ct_id] = f'Type {ct_id}: Broad-spiking'

print(f"\nDiscovered cell types:")
for ct_id, ct_name in celltype_names_gmm.items():
    count = np.sum(celltype_labels_gmm == ct_id)
    pct = 100 * count / len(celltype_labels_gmm)
    print(f"  {ct_name}: {count} spikes ({pct:.1f}%)")

# %%
# Plot BIC curve and silhouette scores
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: BIC curve
axes[0].plot(list(n_clusters_range), bic_scores, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(n_clusters_best, color='red', linestyle='--', linewidth=2, 
               label=f'Best (BIC min): {n_clusters_best}')
axes[0].axvline(n_clusters_elbow, color='orange', linestyle='--', linewidth=2, 
               label=f'Elbow: {n_clusters_elbow}')
axes[0].set_xlabel('Number of clusters', fontsize=12)
axes[0].set_ylabel('BIC (lower is better)', fontsize=12)
axes[0].set_title('GMM Model Selection: BIC Curve', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Right: Silhouette scores
axes[1].plot(list(n_clusters_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].axvline(n_clusters_best, color='red', linestyle='--', linewidth=2, 
               label=f'BIC choice: {n_clusters_best}')
axes[1].set_xlabel('Number of clusters', fontsize=12)
axes[1].set_ylabel('Silhouette score (higher is better)', fontsize=12)
axes[1].set_title('Cluster Quality: Silhouette Score', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gmm_model_selection.png', dpi=150, bbox_inches='tight')
print('Saved: gmm_model_selection.png')

# %%
# Visualize discovered cell types
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

unique_celltypes_gmm = np.unique(celltype_labels_gmm)
colors_celltypes = cc.glasbey[:len(unique_celltypes_gmm)]

# Top left: Cell type waveforms overlaid
for i, ct_id in enumerate(unique_celltypes_gmm):
    ct_spikes = spikes_nonzero[celltype_labels_gmm == ct_id]
    ct_name = celltype_names_gmm.get(ct_id, f'Type {ct_id}')
    
    sample_idx = np.random.choice(len(ct_spikes), size=min(100, len(ct_spikes)), replace=False)
    axes[0, 0].plot(ct_spikes[sample_idx].T, color=colors_celltypes[i], alpha=0.02)
    
    template = np.median(ct_spikes, 0)
    axes[0, 0].plot(template, color=colors_celltypes[i], linewidth=3, label=ct_name)

axes[0, 0].axvline(42, color='black', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Sample (timepoint)', fontsize=11)
axes[0, 0].set_ylabel('Amplitude', fontsize=11)
axes[0, 0].set_title('Discovered Cell Type Waveforms', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3, axis='x')

# Top right: Cell type waveforms stacked
vertical_offset = 0
for i, ct_id in enumerate(unique_celltypes_gmm):
    ct_spikes = spikes_nonzero[celltype_labels_gmm == ct_id]
    ct_name = celltype_names_gmm.get(ct_id, f'Type {ct_id}')
    
    sample_idx = np.random.choice(len(ct_spikes), size=min(50, len(ct_spikes)), replace=False)
    axes[0, 1].plot(ct_spikes[sample_idx].T + vertical_offset, 
                   color=colors_celltypes[i], alpha=0.05, linewidth=0.5)
    
    template = np.median(ct_spikes, 0)
    axes[0, 1].plot(template + vertical_offset, color=colors_celltypes[i], linewidth=2.5)
    axes[0, 1].annotate(ct_name, xy=(5, vertical_offset + 0.5), fontsize=9, fontweight='bold')
    vertical_offset += 2.5

axes[0, 1].axvline(42, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Sample (timepoint)', fontsize=11)
axes[0, 1].set_ylabel('Amplitude (offset per type)', fontsize=11)
axes[0, 1].set_title('Stacked Cell Type Waveforms', fontsize=12, fontweight='bold')
axes[0, 1].set_ylim(-1, vertical_offset)
axes[0, 1].grid(alpha=0.3, axis='x')

# Bottom left: PCA embeddings colored by discovered cell types
for i, ct_id in enumerate(unique_celltypes_gmm):
    ct_emb = fc_pca_ceed_emb[celltype_labels_gmm == ct_id]
    ct_name = celltype_names_gmm.get(ct_id, f'Type {ct_id}')
    axes[1, 0].scatter(ct_emb[:, 0], ct_emb[:, 1], 
                      color=colors_celltypes[i], alpha=0.4, s=15, label=ct_name)
    centroid = np.median(ct_emb, axis=0)
    axes[1, 0].scatter(centroid[0], centroid[1], 
                      color=colors_celltypes[i], marker='*', s=500, 
                      edgecolors='black', linewidths=2, zorder=10)

axes[1, 0].set_xlabel('PC1', fontsize=11)
axes[1, 0].set_ylabel('PC2', fontsize=11)
axes[1, 0].set_title('PCA: GMM-Discovered Cell Types', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# Bottom right: UMAP embeddings colored by discovered cell types
for i, ct_id in enumerate(unique_celltypes_gmm):
    ct_emb = fc_umap_ceed_emb[celltype_labels_gmm == ct_id]
    ct_name = celltype_names_gmm.get(ct_id, f'Type {ct_id}')
    axes[1, 1].scatter(ct_emb[:, 0], ct_emb[:, 1], 
                      color=colors_celltypes[i], alpha=0.4, s=15, label=ct_name)
    centroid = np.median(ct_emb, axis=0)
    axes[1, 1].scatter(centroid[0], centroid[1], 
                      color=colors_celltypes[i], marker='*', s=500, 
                      edgecolors='black', linewidths=2, zorder=10)

axes[1, 1].set_xlabel('UMAP1', fontsize=11)
axes[1, 1].set_ylabel('UMAP2', fontsize=11)
axes[1, 1].set_title('UMAP: GMM-Discovered Cell Types', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gmm_discovered_celltypes.png', dpi=150, bbox_inches='tight')
print('Saved: gmm_discovered_celltypes.png')

# ...existing code...