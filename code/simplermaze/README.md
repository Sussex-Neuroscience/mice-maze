# Welcome to the tactile experiment component of the aMAZEing maze



For posthoc analysis, we are currently relying on DeepLabCut output, which is not particularly robust. 
We are working on data labeling for robustness, and will compare performance of [Yolov8](https://yolov8.com/), [STPoseNet](https://github.com/lvrgb777/STPoseNet/tree/master) (check out their [paper](https://www.sciencedirect.com/science/article/pii/S2589004224009945)!), and [DLC](https://github.com/DeepLabCut/DeepLabCut) (check out their [paper](https://www.nature.com/articles/s41596-019-0176-0)!) for accuracy. 

**For newer versions of the script** , <a href="simplermaze.py">simplermaze.py</a> outputs already a video of the full session, and a subdirectory containing the segments of the individual trials, together with a summary of the trial segments and frames ranges (that can then be used to extrapolate the trajectories per trials with the chosen keypoint estimator tool).

**For older versions of the script/data**, <a href="trial_segmentation_video_analysis.ipynb">trial_segmentation_video_analysis.ipynb</a> will handle the segmentation by reprocessing the videos. 

There are different ways to **plot this data**, right now we are using <a href="make_3d_dlc_plot_trajectories.py">make_3d_dlc_plot_trajectories.py</a>. This might change or we might make a new version when the newer versions of the script will be running , but that's a job for future us, isn't it?

Now, important. To calculate the speed, we need to convert the euclidean distance calculated in pixels per frame, to cm/s. The script should handle this automatically as long as you enter the **px/cm** as one of the arguments. To find out how many px/cm, please run <a href="measure_pixel_per_cm.py">measure_pixel_per_cm.py</a>. 