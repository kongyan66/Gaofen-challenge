# Ai-Competition-Tool

The sixth place winning solution (7/147) in the Object Recognition in  Sar Images, 2021 Gaofen Challenge on Automated High-Resolution Earth Observation Image Interpretation. 


## Members
[Caihua kong](https://github.com/kongyan66),  [Longgang Dai](https://github.com/dailonggang), [Zhentao Fan](https://github.com/zt-fan), [Xiang Chen](https://github.com/cschenxiang).


## Solution

* **On-line date augmentation**  
  We use random combination of affine transformation, flip, scaling, optical distortion for data augmentation.

* **Multi-scale training and testing**    
  The training images are resized into sizes of 600, 800, and 1024 for training and testing.


* **Lower confidence**  
  Set the output threshold into 0.005.
