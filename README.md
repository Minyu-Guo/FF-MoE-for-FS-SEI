### Feature-Family-wise-Mixture-of-Experts-for-Robust-FS-SEI
This letter proposes a robust FS-SEI method based on FF-MoE. The proposed method explicitly decomposes RF representations into physically meaningful feature families and performs adaptive weighting for each expert. This structured fusion improves both generalization and interpretability under limited supervision.

## Dataset
The datasets (download links given below) are described in the paper titled Distinguishable IQ Feature Representation for Domain-Adaptation Learning of WiFi Device Fingerprints in IEEE Transactions on Machine Learning in Communications and Networking, which include 4 Scenario and we adopt Scenario 2: Cross-Day Wireless Scenario. Link: https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/Stable-WiFi-Dataset/Wireless-Dataset/

## Code Procedure
1、Data Preprocess
First, we read the original HDF5 packet files from the Stable WiFi dataset and convert them directly into an FF-MoE-compatible MAT (HDF5/v7.3-style) file.
# Dataset assumptions (from the official release note)
----------------------------------------------------
1) Each HDF5 file contains a dataset named 'data'.
2) The shape of 'data' is (N, 50340), where N is the number of packets in that file.
3) For each packet row:
   - first 25170 values: I samples
   - next 25170 values: Q samples
4) Sample rate defaults to 45e6.

# Design choice used here
-----------------------
To keep the train/export/downstream few-shot scripts compatible and avoids exploding the dataset size, each packet is treated as one candidate sample. Since each packet is much longer than the frame length used by the current FF-MoE pipeline, one fixed sub-window is selected from each packet (max-energy / center / random), and that sub-window is used to build:
  - X_time (17)
  - X_freq (5)
  - X_tf   (5)
  - X_inst (12)
  - featureMatrix (39)
  - specTensor
  - occTensor

2、
