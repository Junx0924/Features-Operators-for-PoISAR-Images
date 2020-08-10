1 Handcrafted_Features:\
Calculate handcrafted features from polsar files, store them in HDF5 file, then apply KNN to classifiy these features.
Generate the colormap of class results, and the classification overall accuracy.
Reduce the dimension of random selected feature groups, then plot them by class in python
Generate feature maps for each dimension of feature groups

Dependencies: MPEG-7, T-SNE, HDF5 (installed by Anaconda)

Main functions:
1) ober::caculFeatures (hdf5file, feature_name, filterSize, patchSize, batchSize)
2) Utils::classifyFeaturesML(hdf5file, feature_name, "opencvKNN", 80, filterSize, patchSize, batchSize)
3) Utils::featureDimReduction(hdf5file, feature_name, filterSize, patchSize, batchSize):\
Reduced the feature dimension by T-SNE, dump the first batch to txt file for plotting
4) Utils::generateFeatureMap(hdf5file, feature_name, filterSize, patchSize, batchSize):\
Generate feature maps for each feature group

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
