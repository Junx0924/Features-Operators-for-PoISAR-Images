Done:
* target decomposition (Krogager, Pauli, Huynen, Freeman-Durden, Yamaguchi4, Cloude-Pottier)
* MP ( opening-closing by reconstruction)
* ELBP
* GLCM ( contrast, entrophy, homogenity, energy)
* MPEG_7 CSD, DCD ( used MPEG_7 libraries by Muhammet Bastan, www.cs.bilkent.edu.tr/~bilmdg/bilvideo-7/Software.html)
* Local statistic ( min, max, mean, std, median) of polsar parameters
* Data pipeline of Sen12ms
* Data pipeline of Oberpfaffenhofen
* KNN - right now tested with Obf datasets. To be extended to Sen12ms


Notes:
KNNTest in KNN.h Function : classifies the test samples based on training data Arguments: Trained samples, labels and testing samples and labels

DivideTrainTestData in main.cpp Function: Splits the training and test data sets Arguments: Data samples

calculatePredictionAccuracy in Performance.cpp Function: Calculates accuracy percentage Arguments: Predicted and true labels

Feature.cpp Contains feature extractors
