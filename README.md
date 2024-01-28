
# Microscope Cell Image Processing

This project utilizes a series of image processing techniques to process and analyze microscopic cell images.

## Overview
In this project, a cell image is taken and subjected to a series of operations:

- Sharpening Filter: The image is first processed with a sharpening filter. This enhances the edges in the image, making the cells more distinct.

- Adaptive Thresholding: The sharpened image is then converted to a binary image using adaptive thresholding. This separates the cells from the background.

- Morphological Opening: The binary image is processed with a morphological opening operation. This removes small noise in the image and separates the cells that are close to each other or not well defined.

- Gaussian Blur: The image is then blurred using a Gaussian blur. This reduces high-frequency noise in the image, making the cells easier to detect.

- Canny Edge Detection: The blurred image is processed with the Canny edge detection algorithm. This detects the edges of the cells, which can be used to determine the shape and location of each cell.

- Dilation: The edge-detected image is then dilated. This makes the cells larger and easier to detect.

- Connected Components Analysis: Finally, the dilated image is analyzed using connected components analysis. This labels each cell in the image and provides statistics about each cell, such as its area and centroid.

- Kmeans Algorithm: KMeans clustering algorithm uses the fixed centers of the components and divides them into a certain number of clusters for us

These operations result better information about the cells in the image, including their locations and shapes.

## Tech-stack
- Python
- OpenCV
- Scikit-Image
- Scikit-Learn

## Model

All details regarding to model and results are covered in the Jupyter notebook file named [model.ipynb](https://github.com/waldvoid/spotifyProject/blob/main/microscope-cell-image-processing/model.ipynb)

## Start Using

To run this project, you first need to install the following Python libraries:

### Step 1: Open Git Bash or Terminal
Open Git Bash on Windows or Terminal on Mac/Linux.

### Step 2: Clone Command
Paste the following command into Git Bash or Terminal:

`git clone https://github.com/waldvoid/microscope-cell-image-processing`

### Step 3: Navigate to Project Directory
Use the following command to navigate to the downloaded repository's directory:

`cd microscope-cell-image-processing`

### Step 4: Run the Project
Install the necessary dependencies to run the project:

`pip install -r requirements.txt`

Navigate to the application directory and run python file

`python model.py`

![Microscope Cell Image Processing](https://i.imgur.com/Q99xg6K.png)
