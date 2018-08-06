# DeepMIIL
3D-volume Conv Pix2Pix cGans for PET/MR

Based on pix2pix by Isola et al.
Directed conversion of affinelayer's tensorflow implementation.
https://github.com/affinelayer/pix2pix-tensorflow

To run
-Input folder: String that points to the directory containing the 3D volumes. Each volume should be represented by its own sub-folder containin all the individual slices. Each slice of the volume should be named numerically based on their order. [IE: 1.png, 2.png, 3.png, ..., xxx.png]

# Autov: MSE and MS-SIM accuracy benchmarks. 
autov compile (temp makefile replacement): g++ autov.c++ -o autov `pkg-config --cflags opencv`  `pkg-config --cflags --libs opencv`

TODO: generate a error-heatmap of the attenuation mapping
