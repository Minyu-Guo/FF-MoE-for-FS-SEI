# Feature-Family-wise-Mixture-of-Experts-for-Robust-FS-SEI
This letter proposes a robust FS-SEI method based on FF-MoE. The proposed method explicitly decomposes RF representations into physically meaningful feature families and performs adaptive weighting for each expert. This structured fusion improves both generalization and interpretability under limited supervision.

## Dataset
The datasets (download links given below) are described in the paper titled Distinguishable IQ Feature Representation for Domain-Adaptation Learning of WiFi Device Fingerprints in IEEE Transactions on Machine Learning in Communications and Networking, which include 4 Scenario and we adopt Scenario 2: Cross-Day Wireless Scenario. Link: https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/Stable-WiFi-Dataset/Wireless-Dataset/

## Code Procedure
### 1、Data Preprocess
----------------------------------------------------------
First, the raw HDF5 packet files are read from the Stable WiFi dataset. Then, the raw I/Q signals are preprocessed and reorganized into a unified format, followed by the extraction of multi-domain feature families for different experts, including time-domain, frequency-domain, time-frequency, and instantaneous-domain representations. Finally, the processed signal segments, image-based representations, and handcrafted features are jointly saved into MAT files (HDF5/v7.3 format) compatible with FF-MoE for subsequent expert pretraining and gate-guided fusion training.  We name this script data_preprocess_stable_wifi.py.
#### Dataset assumptions (from the official release note)
1) Each HDF5 file contains a dataset named 'data'.
2) The shape of 'data' is (N, 50340), where N is the number of packets in that file.
3) For each packet row:
   - first 25170 values: I samples
   - next 25170 values: Q samples
4) Sample rate defaults to 45e6.
#### Design choice used here
To keep the train/export/downstream few-shot scripts compatible and avoids exploding the dataset size, each packet is treated as one candidate sample. Since each packet is much longer than the frame length used by the current FF-MoE pipeline, one fixed sub-window is selected from each packet (max-energy / center / random), and that sub-window is used to build:
  - X_time (17)
  - X_freq (5)
  - X_tf   (5)
  - X_inst (12)
  - featureMatrix (39)
  - specTensor
  - occTensor

### 2、Single expert pretrain and weight inject
----------------------------------------------------------
Each domain-specific expert is pretrained independently by training on the corresponding input derived from the training set, and the best-performing checkpoint is saved for the subsequent fusion stage. This script is single_expert_train_v2.py. The inject script is inject_experts_into_moe.py.
#### Single expert pretrain
We train the four experts(time/freq/tf/inst) individually and save their optimal parameters as the corresponding best_xxx.pth files respectively for the subsequent fusion stage.
#### Weight inject
In this stage we load optimal parameters (best_xxx.pth) of the four individual experts into the corresponding expert branches of FF-MoE respectively. Subsequently, an MoE checkpoint with injected expert weights (stored as a pure \texttt{state_dict}) is saved for the following gate-and-fusion training stage.

### 3、Model Training
----------------------------------------------------------
After the pretrained expert parameters are loaded into the corresponding FF-MoE branches to initialize the expert modules, which the expert weights are frozen. The gating network is then trained to adaptively assign expert weights and fusion the output of different exports for final prediction. Finally, the best FF-MoE parameters on the validation set are selected for subsequent testing or few-shot evaluation. This script is train_moe_tfcnn_supcon_explain_closedloop_v2.py

### 4、Few-shot evaluation
----------------------------------------------------------
After training, the script export_hfusion_supcon_closedloop_v2_consistent.py is used to export the learned FF-MoE representations into a unified feature file for downstream few-shot evaluation. This step converts the trained model outputs into consistent embeddings and metadata (e.g., labels and file identifiers), which avoids repeated full forward inference during episodic evaluation and ensures compatibility with the existing few-shot evaluation protocol. Based on the exported feature file, the downstream script (downstream_fssei_fewshot_SNR.py) is then applied to constract episodic tasks under different shot settings and to assess the final FS-SEI performance in a consistent and reproducible manner.

## Visual
To intuitively observe the model's performance across different environments, this project also provides corresponding visualization and plotting scripts, which can be found in the "visual" directory. In addition, the project includes two sets of experiments: comparative experiments and ablation studies. The comparative experiments benchmark the model against FineZero, TWC CNN, and GLFormer under identical experimental settings, with the relevant scripts stored in the "const_experiment" folder. For ablation studies, a single-expert retention mode is supported. The relevent interface is provided in the downstream script, and this mode can be directly enabled by modifying the configuration in the corresponding script.

