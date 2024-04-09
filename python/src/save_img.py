import cv2
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
k_in = ""
cnt = 0
image_dir = "../data/colmap/"

while True:
  try:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
  
    if k_in == "s":
      cnt += 1
      cv2.imwrite(image_dir + "colmap" + str(cnt) + ".png", color_image)

  except Exception as e:
      print(e)
  k_in = input("input")
  if k_in == "q":
    break
