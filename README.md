# Iranian-Traditional-Music-Dastgah-Classifier

Iranian traditional music is distinguished by its seven primary Dastgahs: Chahargah, Homayoun, Mahour, Segah, Shour, Nava, and Rast-Panjgah. Within the scope of this project, we constructed a classifier aimed at discerning the Dastgah to which each music track is affiliated. Additionally, we applied clustering techniques to group similar music tracks based on their audio features. This project also encompassed tasks such as gathering suitable music tracks, preprocessing audio signals, extracting relevant features, and assembling a notably informative dataset.

This project has several parts:

## Data Cleaning
- Remove data with incorrect formatting
- Convert data to the WAV format
- Split audio files into segments of uniform length

## Feature Extraction
In this project, we have selected features from three categories: temporal features, spectral features, and harmonic features.
We employed two feature extraction methods: the first method (Feature Set 1) utilizes all three feature types, while the second method (Feature Set 2) exclusively utilizes spectral and temporal features.

## Feature Selection
Dimension reduction in audio files aims to trim feature count while retaining crucial information and eliminating redundancy. The methods implemented in this project to reduce the dimensions of the feature vector are:
- Principal Component Analysis
- Linear Discriminant Analysis
- Forward Selection
- Backward Elimination


## Classification
Following the splitting of the dataset into training and testing sets, we proceed to standardize them and subsequently apply the following machine learning algorithms for classification:
- K-Nearest Neighbors
- Support Vector Machine
- Extreme Gradient Boosting
- Multilayer Perceptron
- Logistic Regression
- Long Short-Term Memory
