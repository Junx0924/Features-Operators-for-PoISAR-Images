1 Handcrafted_Features:\
1) Calculate handcrafted features from polsar files, store them in HDF5 file
2) Apply KNN to classifiy these features.
3) Generate the colormap of class results, and the classification overall accuracy.
4) Reduce the dimension of random selected feature groups, then plot them by class in python
5) Generate feature maps for each dimension of feature groups

Dependencies: MPEG-7, T-SNE, HDF5 (installed by Anaconda)

How-to-use:
main.exe \<ratFolder\> \<labelFolder\> \<Hdf5File\> \<featureName\> \<filterSize\> \<patchSize\>

\<filterSize\> choose from: 0,5,7,9,11

\<featureName\> choose from: mp, decomp, color, texture, polstatistic, ctelements

mp stands for: morphological profile features

decomp stands for: target decomposition features

color stands for: MPEG-7 CSD,DCD and HSV features

texture stands for: GLCM and LBP features

polstatistic stands for: the statistic of polsar parameters

ctelements stands for: the 6 upcorner elements of covariance and coherence matrix
