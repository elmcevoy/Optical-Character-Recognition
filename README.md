# Optical-Character-Recognition

Usage Instructions:
This pipeline can be executed with the "run.py" script.  This script will load in and process 5 images, and will generate an output of 5 images with bounding boxes and digit classification displayed.   This script requires the use of trained weights, which can be found at the following location:

https://drive.google.com/drive/folders/1pkFPTbXy2yXQb4siWjkS_Ajc1Ae-LKnc?usp=drive_link

The following directory structure must be created to run this pipeline:
	At the same level where the "run.py" script is located on your computer, create 3 folders.  
	One folder should be named "houses" (no quotation).
	The other folder should be named "graded_images" (no quotation).
	The third folder should be named "supporting_files" (no quotation).
	See attached Screenshot for a visualization of this file structure, if needed.


Inside the "graded_images" folder, create another folder called "cnn_model" (no quotation).  The trained weights downloaded from Google Drive (.pth file) should be inserted into this "cnn_model" folder.  Nothing else should belong to this "cnn_model" folder.  After the "run.py" script is exectuted, the 5 output images will be saved and written to this "graded_images" folder.

Inside the "houses" folder, the following housing data files should be inserted: 
House1.png, House2.webp, House3.webp, House4.webp, House5.png

Inside the "supporting_files" folder, the following files should be inserted:
	train.py, video_run.py

The "supporting_files" folder contains supporting files and code that was used in this project.  The train.py file is a script of the code that was used to train the 3 classifier models on Google Colab, as well as generate the graphs and tables contained in the Report.  The video_run.py script is the script of the code that was used to generate the video.  The video pipeline is an adaptation of run.py that is formatted to accept frames of a video as input and generate a video as output.

To view the output of the video, the file can be found at the following location on Google Drive:

https://drive.google.com/drive/folders/1pkFPTbXy2yXQb4siWjkS_Ajc1Ae-LKnc?usp=drive_link


---------------------------------------------------------

Python Packages Instructions:

The "run.py" file makes use of the following Python packages and their versions:

- python=3.7.11
- numpy=1.17.4
- scipy=1.3.1
- pytorch=1.10
- torchvision=0.11.1
- opencv-python==4.1.2.30

These should be compatible with the environment defined by the cv_proj.yml file.

---------------------------------------------------------

Image Data Information:

The 5 images that were chosen for the Image Classification Task are found in the "houses" directory.  These 5 images are photographs of houses that were individually downloaded from zillow.com.  

---------------------------------------------------------

Information about Supporting files:

There are 2 files contained in the folder "supporting_files".

The train.py file is a python script that was used to generate the weights of the CNN classifiier.  There are 3 models trained in this file (and are described in full detail in the Report).  The Custom CNN model corresponds to the code that is executable.  The code that is commented out was used for training both of the VGG-16 models (untrained, and pre-trained versions).  This code was executed on a Google Colab instance, and later converted to the train.py script contained in this directory.  The executable code in this file will generate the trained weights for the Custom CNN model.  A GPU is recommended for this task.

The video_run.py file is a python script that extends the run.py script, so as to generate a video .mp4 file from individual frames of a recorded video.  The original recorded video was taken on a Samsung cellphone of a house, and used to generate the video.mp4 file (see Google Drive link above for file location).  

