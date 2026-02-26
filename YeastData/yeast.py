############################################################
# Yeast Data Processing
#
# Step 1: Load yeast pooled-seq frequency data
# Step 2: Construct frequency tensor and save conforming our model output
# Step 3: Generate haplotype matrix based on initial frequencies
# Step 4: Convert haplotype matrix to MIMICREE-style input format
############################################################


############################################################
# Step 1: Load yeast frequency data (from processed .df file)
############################################################
import pandas as pd
import numpy as np
## Yeast pooled frequency data
yeast = pd.read_pickle('yeast.df')

# Save a subset (first 500 SNPs) for benchmark usage
yeast.iloc[0:500,:].to_pickle(
    'yeast_500.df'
)

# Reshape structure:
# SNP x Replicate x Timepoint x (allele_count, total_count)
yeast = yeast.values.reshape(1000, 12, 4, 2)

# Convert counts to allele frequencies
freq = lambda x: x[:,:,:,0] / x[:,:,:,1]
freq_yeast = freq(yeast)

# Subset:
# - first 500 SNPs
# - first 10 replicates
# - first 3 timepoints
#freq_yeast = freq_yeast[:140, [1,5], :3]
#freq_yeast = freq_yeast[140:280, [1,5], :3]
#freq_yeast = freq_yeast[280:420, [1,5], :3]
#freq_yeast = freq_yeast[420:560, [1,5], :3]
#freq_yeast = freq_yeast[560:700, [1,5], :3]
#freq_yeast = freq_yeast[700:840, [1,5], :3]
#freq_yeast = freq_yeast[840:1000, [1,5], :3]
############################################################
# Step 2: Save reshaped frequency input for our model inference
############################################################

final_yeast = []

for i in range(2):
    # Flatten SNP x time into a single vector per replicate
    # Insert number of timepoints (3) at position 0
    final_yeast.append(
        np.insert(
            np.around(freq_yeast[:,i,:].flatten(), 3),
            0,
            3
        )
    )
#np.savez('yeast_140_2_3', realobs=final_yeast)
#np.savez('yeast_141_280_2_3', realobs=final_yeast)
#np.savez('yeast_281_420_2_3', realobs=final_yeast)
#np.savez('yeast_421_560_2_3', realobs=final_yeast)
#np.savez('yeast_561_700_2_3', realobs=final_yeast)
#np.savez('yeast_701_840_2_3', realobs=final_yeast)
#np.savez('yeast_841_1000_2_3', realobs=final_yeast)


# ############################################################
# # Step 3: Generate haplotype matrix from initial frequencies
# ############################################################
#
# # Use allele frequency at replicate 0, timepoint 0
# initial_freq = freq_yeast[:,0,0]
#
# # Construct binary haplotype matrix:
# # 500 SNPs x 1000 individuals
# yeast_hp = np.zeros((500,1000))
#
# for i in range(len(initial_freq)):
#     pi = initial_freq[i]
#     yeast_hp[i,] = np.random.choice(
#         [0, 1],
#         size=1000,
#         p=[1-pi, pi]
#     )
#
# # Save simulated haplotype matrix
# pd.DataFrame(yeast_hp).astype(int).to_csv(
#     'yeast_1000_500_10_3',
#     header=None,
#     sep='\t',
#     mode='a',
#     index=False
# )
#
#
# ############################################################
# # Step 4: Convert haplotype matrix into CLEAR-style input
# ############################################################
#
# aa = pd.read_csv(
#     "../input/real_ha_1000",
#     header=None,
#     sep='\t'
# )
#
# hp_str = pd.read_csv(
#     "../input/yeast_2000.txt",
#     header=None,
#     sep='\t'
# )
#
# hp_str = pd.DataFrame(hp_str)
#
# # Convert 0/1 to nucleotide encoding
# hp_str = hp_str.replace(1, 'A')
# hp_str = hp_str.replace(0, 'C')
#
# # Concatenate SNP row into haplotype string
# hp_str['new'] = hp_str.apply(''.join, axis=1)
# hp_str = hp_str['new']
#
# total = []
#
# # Split into 2-mer segments
# for line in hp_str:
#     sep = [line[i:i+2] for i in range(0, len(line), 2)]
#     total.append(sep)
#
# # Split into single nucleotide segments
# for line in hp_str:
#     sep = [line[i] for i in range(2000)]
#     total.append(sep)
#
# hp_str = pd.DataFrame(total)
#
# # Add structural annotation columns
# hp_str.index = np.arange(1, len(hp_str) + 1)
# hp_str.reset_index(level=0, inplace=True)
#
#
# hp_str['chromo'] = '2L'
# hp_str['ref'] = 'C'
#
# cols = hp_str.columns.tolist()
# cols = [cols[-2]] + [cols[0]] + [cols[-1]] + cols[1:-2]
# hp_str = hp_str[cols]
#
# anc = [x for x in ['C/A'] for _ in range(len(hp_str))]
#
# index = np.where((shape_sub1[:,0] > 0.1) & (shape_sub1[:,0] < 0.9))[0]
# index = aa.iloc[:,0]
#
# hp_str.insert(3, 'anc', anc)
#
# # Filter by selected SNP indices
# hp_str = hp_str.filter(items=index-1, axis=0)
# hp_str = hp_str.loc[hp_str.iloc[:,1].isin(index)]
#
# hp_str.to_csv(
#     '../input/sim_yeast_2000',
#     header=None,
#     sep='\t',
#     mode='a',
#     index=False
# )
