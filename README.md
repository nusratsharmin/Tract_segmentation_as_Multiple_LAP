Tract_segmentation_as_Multiple_LAP

Overview:
This is a set of files for segmenting the tract of interest from the whole brain tractogram, which is presented in "White Matter Tract Segmentation as Multiple Linear Assignment Problems". It allows you to do the tract segmentation based on some prior knowledge (segmented tract from the different brain). Hence the inputs of the experiment are the segmented example tracts and the whole brain tractogram.    

The main file for running the experiment is "segmentation_as_NN_and_lap.py", which takes one test tractogram, T_A and three example tract of Uncinate fasciculus (uf.left) tracts. The code will save the segmented tract in ".trk" of the uf.left from T_A. Moreover, it will visualize the segmented tract. 


Requirements:

numpy
dipy
sklearn
joblib (optional)





