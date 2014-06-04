Introduction
------------

This is an implementation of GBH descriptor with Fisher vector encoding. The details about GBH can be found at my phd thesis here: http://www.site.uottawa.ca/~fshi098/ 

If you use our code, please cite our CVPR'13 paper / phd thesis. To run the code, you need to have opencv library.
To run the code with Fisher vector, you need to download  VLFeat library here: http://www.vlfeat.org/, with proper citation.
If you use the svm in our code (can be found here: https://github.com/fshi/actionMBH), you should cite libsvm paper here: http://www.csie.ntu.edu.tw/~cjlin/libsvm/. You should also read its COPYRIGHT: http://www.csie.ntu.edu.tw/~cjlin/libsvm/COPYRIGHT.


COPYRIGHT
---------

The code is only for research purpose. Any usage on commerical purpose should have author's acknowldege.


Usage
-----

Dataset subdirectory::

     To run the program, you need to have the dataset videos, and replace (const string dirName = "C:\\dataSets\\hmdb51\\";) with your dataset directory. For simplicity, the videos from the same class are stored in a subdirectory. One class, one subdirectory, marked with 1,2,3...

Visual studio project properties::

     If you have compiler error, try to change the VS project properties as: Configuration properties->General->Character Set->Not Set. If using vs2012 and having compiler error of "itoa...", you can set: property pages->Configuration Properties->C/C++->General->SDL checks->No(/sdl-).
   
If you have questions, contact me at: fshi98_at_gmail_com  
     
Parameters::

     You need to include “MBH_parameters_input.txt” file inside your working directory as input parameters.

Pre-computed GMM (k=128) and pca projection matrix::

	You can use the pre-computed GMMs and pca matrix for the attached parameter “MBH_parameters_input.txt”. If you change the parameters, it is important to recompute the GMMs and pca with attached project "getGMM". The pre-computed GMM files can be found inside the sub-directory: "\computed GMMs". "gmmResults0.yml" is for root, "gmmResults1.yml" is for parts, and "pca_Mat.yml" is for PCA. You need to copy them into your working directory.



Steps to do testing/training:

1.  To run the program, you need to have the dataset videos, and replace (const string dirName = "C:\\dataSets\\hmdb51\\";) with your dataset directory. For simplicity, the videos from the same class are stored in a subdirectory. One class, one subdirectory, marked with 1,2,3...
2. You need to include “MBH_parameters_input.txt” file inside your working directory as input parameters.
3. You can learn GMMs and pca projection matrix by running "getGMM.exe". However, you can also use the pre-computed GMM files inside the sub-directory: "\computed GMMs". 
4. Next step is to run "getTrainTestData.exe". After running the *getTrainTestData*, you should get the files representing each video file with bag-of-features representation (one .dat file for one video clip). I assume that you are testing with HMDB51 dataset, and all the .dat files are stored inside the sub-directory, such as random1, random2,..., random51.

The other steps shold be same as in this project: 
https://github.com/fshi/actionMBH





