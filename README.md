1) Handcrafted_Features:

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

Main function descriptions:
ober::caculFeatures (hdf5file, feature_name, filterSize, patchSize, batchSize)
	Generate handcrafted features and store to hdf5 file.

Utils::classifyFeaturesML(hdf5file, feature_name, "opencvKNN", 80, filterSize, patchSize, batchSize);
Classify the features and store the class results to hdf5 file.

Utils::featureDimReduction(hdf5file, feature_name, filterSize, patchSize, batchSize);
reduced the feature dimension by T-SNE
 dump the first batch to txt file for plotting

Utils::generateFeatureMap(hdf5file, feature_name, filterSize, patchSize, batchSize);
Generate feature maps for each feature group
