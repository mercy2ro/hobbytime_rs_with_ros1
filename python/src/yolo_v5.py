import pyrealsense2 as rs
import numpy as np
import cv2
import yolov5

#ready for yolov5
# load pretrained model
model = yolov5.load('../include/yolov5/yolov5s.pt')

# or load custom model
#model = yolov5.load('../include/yolov5/train/best.pt')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#d435i
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        color_colormap_dim = color_image.shape

        img = color_image
        results = model(img, augment=True)

        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4] # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        
        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('frame', np.squeeze(results.render()))
        #cv2.imshow('RealSense', results)
        #cv2.imshow('RealSense', images)
        cv2.waitKey(1)



finally:

    # Stop streaming
    pipeline.stop()
