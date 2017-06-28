##Writeup Template
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Writeup_images/vehicle.png "This is a typical vehicle"
[image2]: ./Writeup_images/non_vehicle.png
[image3]: ./Writeup_images/Car_post_HOG.png
[image4]: ./Writeup_images/non_vehicle_post_HOG.png
[image5]: ./Writeup_images/sliding_window_smaller.png
[image6]: ./Writeup_images/sliding_window_larger.png
[image7]: ./Writeup_images/slide_product_1.png
[image8]: ./Writeup_images/slide_product_2.png
[image9]: ./Writeup_images/slide_product_3.png
[image10]: ./Writeup_images/heatmap/heat_1.png
[image11]: ./Writeup_images/heatmap/frame_1.png
[image12]: ./Writeup_images/heatmap/heat_2.png
[image13]: ./Writeup_images/heatmap/frame_2.png
[image14]: ./Writeup_images/heatmap/heat_3.png
[image15]: ./Writeup_images/heatmap/frame_3.png
[image16]: ./Writeup_images/heatmap/heat_4.png
[image17]: ./Writeup_images/heatmap/frame_4.png
[image18]: ./Writeup_images/heatmap/final_labels.png
[image19]: ./Writeup_images/heatmap/final_picture.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook. It is composed of the function **get_hog_features**, which takes in as input the image to produce the HOG, the number of oriented gradient bins, the number of pixels per cell in the histogram, and the number of cells per block normalization. As will be expanded later, block normalization was not used for the final implementation. 

The actual extraction of HOG features from training images is done in cell 7 when the function **extract_features** is called on each training image. **Extract_features** calls **get_hog_features**. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

Here is an example of visualized results of taking the final HOG features I used from two pictures.


![alt text][image3]
![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

I initially chose parameters based off of examples taken from Udacity's online lessons (found as the default falues in the **single_img_features** function). These include having 9 orientation bins in the HOG and having 8x8 pixel size cells in 64x64 size-adjusted windows (thus, having (64/8)x(64/8) = 8x8 cells per window). 2x2 cells per block normaliztion was used, as well as use of only the 1st  channel of YCrCb. 

I chose to take features from pictures converted to the YCrCb color space  because, when first testing a linear SVM on just color histograms of the pictures, I received the greatest accuracy from using YCrCb when compared to HSV and RGB. I arbitrarily decided to use  the same colorspace for  the HOG as I did for the color histogram.

However, after testing a linear SVM's ability to learn from the features, I found that results left much to be desired. An early change was to use all 3 channels of YCrCb for training, which easily improved the accuracy of the SVM. After additional testing, it became obvious that the training was much too slow, especially with the use of 3 channels. The HOG alone was using 5292 features, not including the color histogram also being used.

I first cut down on the number of features being used  by decreasing block normalization down to 1 cell per block. Block normalization had no discernable improvement on car identification accuracy, and anyways, I've never quite understood why it would be very helpful for a learner. I also decreased the number of orientations down to 6 and decreased the number of blocks in 1/4 by increasing the number of pixels per cell to 16. None of these changes drastically decreased accuracy (which is actually at 100% for the testing data). The number of features used for  the HOG is now 320, or less than 10% of the first amount.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM with a radial basis function (rbf) kernel. The software for doing so can be found in the 10th cell of the Ipython notebook, in the line:

clf.fit(scaled_features,y_train)

The clf variable is an SVC that uses the default values from Scikit for its training values (C = 1, gamma = 1/n_features). When evaluating a given set of features, I used predict_proba (instead  of a binary classifier), so that I could have an intuitive threshold to adjust for the classifier's performance.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I limited my search to the bottom 370 and rightmost 680 pixels of the image, assuming that the car would always be filming from the leftmost lane and that we aren't dealing with any steep hills (or cars in the sky). The functions for the sliding window are defined in cells 14 (**search_windows**), 19 (**video_heatmap**), and 1 (**slide_window**).

I chose two scales, one smaller sliding window of size 64x64 and one larger sliding window of 128x128 (as defined in **video_heatmap**). The smaller sliding window size  was picked with a larger overlap of 75% both vertically and horizontally as it slides along the picture. This was chosen because it allowed for tighter box thresholds around the cars. Later on, it will be explained that heatmaps were used  to combine windows. It proved to be easier to combine multiple, smaller windows than fewer, larger windows. The smaller windows also proved to be better for detecting distant cars, which shrink with distance.

The 128x128 windows were used  because  the smaller windows  proved to be poor in detecting cars near the margins. One of my future goals is to find out why the smaller windows performed  poorly at the tresholds.
  
![alt text][image5]
![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus created histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image7]
![alt text][image8]
![alt text][image9]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./full_project_video_output_v6.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap for every 5 frames and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are four frames and their corresponding heatmaps:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all four frames after removing all points labeled less than 4 times:
![alt text][image18]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image19]



---

###Discussion

####1. Briefly discuss any insights learned and problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

So I took two things out of this project above all else: one, always check if you're expected input (.png) is not of an unexpected filetype (.jpg). I think I have a handful of more grey hairs due to that one. Second, time spent waiting for a slow training algorithm is time lost from experimenting with your parameters. Real-time training is a valuable tool if you can achieve it with comparable accuracy to slow training.

Anyways, what of the problems? I won't lie, there are a few. My biggest frustration is the small blips of false positive labels I continue to get. They last for only a fraction of a second, but they still are an obvious fault of the software. I've tried a few methods to rid myself of them, including limiting the number of new car labels that appear unless if on the edge, but it's been difficult to certify which labels are  the correct ones that match the previous group and which are  the faulty ones. I could increase the number of frames used per heat map (I currently group every 4 frames into 1 heatmap), and that would probably help, but it would also make the software more glitchy and less smooth. I will continue meditating over different solutions.

Another problem is how the learner behaves with cars on the edges of the picture. It seems that both the small and large sliding windows have trouble recognizing cars on the side of the window. This obviously may have something to do with the missing fraction of the car offscreen, but that seems to be contradicted by the proof of sliding windows that capture only half of the cars midscreen. I suspect that my implementation of the algorithm isn't sliding toward the very end of the window, but I haven't found any proof  of that theory in my code nor the libraries I use. I will continue searching down this path. Cars cruising on the boundaries would be a likely source of error.

The third problem the software has is that the bounding boxes of each car oscillate between too big and too small for the cars they follow. One obvious solution is to constrain the size of labels to a minimum and maximum, but since further cars can get a pretty small window size, that doesn't prove to be very helpful. Would would probably be an improvement is to automatically adjust all labels to rectangular shapes based on size: the smaller the label, the more squarish it should be, and the larger the label, the more rectangular it should be. I will continue improvements for shape along this direction.

If you have any questions or comments on this software, please feel free to email me at jao2154@columbia.edu. Also, if you'd like any of the training data mentioned in the code, let me know. Have a good day!
