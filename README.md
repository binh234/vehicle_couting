# Vehicles Counting
An application with a user-friendly interface, easy-to-use using OpenCV, YoloV3 to detect and count the number of vehicles passed through a line in the video, live cam, or stream. This app runs in multi-threading for different videos and streams.

*Note*: You should have a GPU with [NVIDIA driver](https://www.nvidia.co.uk/Download/index.aspx?lang=en-uk), [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit) and [CuDNN](https://developer.nvidia.com/rdp/cudnn-download) to run faster with YoloV3

## Clone project
```git
git clone https://github.com/binh234/vevhicle_counting.git
```

## Dependencies
To install required dependencies run:
```python
pip install -r requirements.txt
```

## Usage

```python
python main.py
```

### References
*GUI*: [github.com/flytocc/pyqt5-cv2-multithreaded](https://github.com/flytocc/pyqt5-cv2-multithreaded)  
*YoloV3*: [pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)  
*SORT*: [github.com/abewley/sort](https://github.com/abewley/sort)  

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }
