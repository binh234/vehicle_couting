# Vehicles Counting
An application with a user-friendly interface, easy-to-use using OpenCV, YoloV3 to detect and count the number of vehicles passed through a line in the video, live cam, or stream. This app runs in multi-threading for different videos and streams.

*Note*: You should have a GPU with [NVIDIA driver](https://www.nvidia.co.uk/Download/index.aspx?lang=en-uk), [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit) and [CUDNN](https://developer.nvidia.com/rdp/cudnn-download) to run faster with YoloV3

## Clone project
```git
git clone https://github.com/binh234/vevhicle_counting.git
```

## Install libraries
```python
pip install -r requirements.txt
```

## Usage

```python
python main.py
```

*GUI*: [PyQt5](https://github.com/flytocc/pyqt5-cv2-multithreaded)  
*Detection*: [YOLOV3-Tiny](https://pjreddie.com/darknet/yolo/)  
*Tracking*: OpenCV
