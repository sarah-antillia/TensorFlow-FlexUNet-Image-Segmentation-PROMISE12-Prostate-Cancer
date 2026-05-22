<h2>TensorFlow-FlexUNet-Image-Segmentation-PROMISE12-Prostate-Cancer (2026/05/23)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>PROMISE12-Prostate-Cancer-Breast-US (Benign and Malignant)</b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>), and a 512x512 pixels PNG
 <a href="https://drive.google.com/file/d/1WohQwykERmJVYUgsEXajfMUZsltAUQp5/view?usp=sharing">
Augmented-PROMISE12-Prostate-Cancer-ImageMask-Dataset.zip</a>, which was derived by us from <br><br>
<a href="https://zenodo.org/records/8026660">
<b>PROMISE12: Data from the MICCAI Grand Challenge: Prostate MR Image Segmentation 2012
</b>
</a>
<br><br>
<hr>
<b>Actual Image Segmentation for PROMISE12-Prostate-Cancer Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to 
the ground truth masks.
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/images/1007_33.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/masks/1007_33.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test_output/1007_33.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/images/1020_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/masks/1020_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test_output/1020_15.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/images/1036_11.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/masks/1036_11.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test_output/1036_11.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://zenodo.org/records/8026660">
<b>PROMISE12: Data from the MICCAI Grand Challenge: Prostate MR Image Segmentation 2012
</b>
</a>
<br>
<br>
The following explanation was taken above zenodo web site.
<br><br>
<b>Description</b><br>
This repository contains all data associated with the 
'Prostate MR Image Segmentation'-challenge 2012 on https://promise12.grand-challenge.org/.<br>
 The goal of this challenge was to compare interactive and (semi)-automatic segmentation algorithms 
 for MRI of the prostate. 
<br><br>
<b>Citation</b><br>
Geert Litjens, Bram van Ginneken, Henkjan Huisman, Wendy van de Ven, Caroline Hoeks, Dean Barratt,<br>
& Anant Madabhushi. (2023). <br>
PROMISE12: Data from the MICCAI Grand Challenge: Prostate MR Image Segmentation 2012 [Data set].<br>
In Medical Image Analysis (Updated the zip files, fixed some issues., <br>
Vol. 18, Number 2, pp. 359?373). <br>
Zenodo. https://doi.org/10.5281/zenodo.8026660
<br><br>
<b>License Text</b><br>
You can use this data for your research. If you do that it is mandatory to cite<br>
<a href="https://doi.org/10.1016/j.media.2013.12.002">https://doi.org/10.1016/j.media.2013.12.002</a>.
<br>
<br>
<h3>
2 PROMISE12-Prostate-Cancer ImageMask Dataset
</h3>
<h3>2.1 PROMISE12-Prostate-Cancer ImageMask Dataset</h3>
 If you would like to train this PROMISE12-Prostate-Cancer Segmentation model by yourself,
 please download the dataset from the google drive  
 <a href="https://drive.google.com/file/d/1WohQwykERmJVYUgsEXajfMUZsltAUQp5/view?usp=sharing">
Augmented-PROMISE12-Prostate-Cancer-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<br>
<pre>
./dataset
└─PROMISE12-Prostate-Cancer
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>PROMISE12-Prostate-Cancer Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/PROMISE12-Prostate-Cancer_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for the
 training set of our segmentation model.
<br>
<br>
<h3>2.2 Derivation of PROMISE12-Prostate-Cancer ImageMask Dataset</h3>
The folder of the original dataset is the following.<br>
<pre>
./8026660
└training_data
    ├─Case00.mhd
    ├─Case00.raw
    ├─Case00_segmentation.mhd
    ├─Case00_segmentation.raw
...
    ├─Case49.mhd
    ├─Case49.raw
    ├─Case49_segmentation.mhd
    └─Case49_segmentation.raw
</pre>
We generated a 512x512 pixels PNG master ImageMask dataset from all pairs of images (<b>Case*.mhd/Case*.raw</b>)
 and their corresponding segmentations (<b>Case*_segmentation.mhd/Case*_segmentation.raw</b>), and then our augmented 
 dataset from the master by using the following image deformation tools:<br>
<a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a><br>
<a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a> <br>
<br>
<h3>2.3 Train Sample Images and Masks</h3>
<b>Train sample images</b><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train sample masks</b><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained PROMISE12-Prostate-Cancer TensorFlowFlexUNet Model by using the 
<a href="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers = 8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 2
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specifed rgb color map dict for PROMISE12-Prostate-Cancer 1+1 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;PROMISE12-Prostate-Cancerrgb color map dict for 1+1 classes.
;                      Cancer:red
rgb_map = {(0,0,0):0, (255, 0, 0):1 }
</pre>
<b>Infer section</b><br>
<pre>
[infer] 
images_dir    = "./mini_test/images/"
output_dir    = "./mini_test_output/"
</pre>
<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test/images</b> folder specified in <b>infer</b> section. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<!--
<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
 -->
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 18,19,20)</b><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was terminated at epoch 40.<br><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/train_console_output_at_epoch40.png" width="1024" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer</b> folder,
and run the following bat file to evaluate TensorFlowUNet model for PROMISE12-Prostate-Cancer.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/evaluate_console_output_at_epoch40.png" width="1024" height="auto">
<br><br>Image-Segmentation-PROMISE12-Prostate-Cancer

<a href="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this <b>PROMISE12-Prostate-Cancer/test</b> was very low 
and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0077
dice_coef_multiclass,0.9967
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for PROMISE12-Prostate-Cancer.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of PROMISE12-Prostate-Cancer Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar 
to the ground truth masks.
<br><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/images/1007_20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/masks/1007_20.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test_output/1007_20.png" width="320" height="auto"></td>

</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/images/1007_34.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/masks/1007_34.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test_output/1007_34.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/images/1014_0.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/masks/1014_0.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test_output/1014_0.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/images/1022_16.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/masks/1022_16.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test_output/1022_16.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/images/1029_16.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/masks/1029_16.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test_output/1029_16.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/images/1036_11.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test/masks/1036_11.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/PROMISE12-Prostate-Cancer/mini_test_output/1036_11.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Evaluation of prostate segmentation algorithms for MRI: The PROMISE12 challenge</b><br>
Geert Litjens, Robert Toth, Wendy van de Ven, Caroline Hoeks, Sjoerd Kerkstra, Bram van Ginneken,<br> 
Graham Vincent, Gwenael Guillard, Neil Birbeck, Jindang Zhang, Robin Strand, Filip Malmberg, <br>
Yangming Ou, Christos Davatzikos, Matthias Kirschner, Florian Jung, Jing Yuan, Wu Qiu, Qinquan Gao,<br>
 Philip “Eddie” Edwards, Bianca Maan, Ferdinand van der Heijden, Soumya Ghose, Jhimli Mitra,<br>
  Jason Dowling, Dean Barratt, Henkjan Huisman, Anant Madabhushi<br>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841513001734?via%3Dihub">
https://www.sciencedirect.com/science/article/abs/pii/S1361841513001734?via%3Dihub</a>
<br><br>

<b>2. TensorFlow-FlexUNet-Image-Segmentation-Prostate158-Prostate-Tumor-T2</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Prostate158-Prostate-Tumor-T2">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Prostate158-Prostate-Tumor-T2</a>
<br><br>

<b>3. TensorFlow-FlexUNet-Image-Segmentation-Prostate-MRI</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Prostate-MRI">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Prostate-MRI</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model</a>
<br><br>

