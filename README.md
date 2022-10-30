# events_cone_detector

This is a implementation of object detection using events data. 
This is a practice from the AMZ racing team ETHz to see if it is possible to use events camera 
as a possible sensor setup in the future.

For the introduction of events camera and realated paper you can check [here](https://rpg.ifi.uzh.ch/research_dvs.html).

In this work we use yolov5 as the object detection model since its fast inference speed satisfy our real-time requirements.


## Some results
### Result of Events Yolov5 detection

<img src="./asset/rgb.gif" alt="drawing" width="600"/>  
Input 
<img src="./asset/Events_input.gif" alt="drawing" width="600"/>  
Events time surface
<img src="./BNN_seg_asset/CamVid/Events_detection.gif" alt="drawing" width="600"/> 
Detection result

