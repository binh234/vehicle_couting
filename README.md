# Vehicles Counting
Simple project using OpenCV, YoloV3 to count number of vehicles passed through a line in video.

*Note*: You should have a GPU with [NVIDIA driver](https://www.nvidia.co.uk/Download/index.aspx?lang=en-uk), [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit) and [CUDNN](https://developer.nvidia.com/rdp/cudnn-download) to run faster with YoloV3

## Clone project
```git
git clone https://github.com/binh234/vevhicle_counting.git
```

## Usage

```python
python main.py
```

*GUI*: [PyQt5](https://github.com/flytocc/pyqt5-cv2-multithreaded)  
*Detection*: [YOLOV3-Tiny](https://pjreddie.com/darknet/yolo/)  
*Tracking*: OpenCV
