#%%

import numpy as np
import cv2 as cv



cap = cv.VideoCapture(3) #change this port number depending upon your USB 0,1,2,3 ...

if not cap.isOpened():
    print("Cannot open camera")
    exit()

img_counter = 1
while True:
     # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret: 
        break

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) & 0xFF == ord('S'):
            # S pressed
            img_name_color = "color_img{}.jpg".format(img_counter)
            #img_name_depth = "depth_img{}.jpg".format(img_counter)
            cv.imwrite(img_name_color, frame)
            #cv2.imwrite(img_name_depth, depth_image)
            print("{} written!".format(img_name_color))
            #print("{} written!".format(img_name_depth))
            img_counter += 1
#            for y in range(480):
#                for x in range(640):
#                    dist = depth_frame.get_distance(x, y)
#                    print(dist)
    elif cv.waitKey(1) & 0xFF == ord('q'):
          cv.destroyAllWindows()
          break

      
    if cv.waitKey(1) == ord('q'):
        break
#print(frame.shape)
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
    
#%%
# # ## License: Apache 2.0. See LICENSE file in root directory.
# # ## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

# # ###############################################
# # ##      Open CV and Numpy integration        ##
# # ###############################################

# import pyrealsense2 as rs
# import numpy as np
# import cv2

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()

# # Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))

# found_rgb = False
# for s in device.sensors:
#     if s.get_info(rs.camera_info.name) == 'RGB Camera':
#         found_rgb = True
#         break
# if not found_rgb:
#     print("The demo requires Depth camera with Color sensor")
#     exit(0)

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

# try:
#     while True:

#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue

#         # Convert images to numpy arrays
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

#         depth_colormap_dim = depth_colormap.shape
#         color_colormap_dim = color_image.shape

#         # If depth and color resolutions are different, resize color image to match depth image for display
#         if depth_colormap_dim != color_colormap_dim:
#             resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
#             images = np.hstack((resized_color_image, depth_colormap))
#         else:
#             images = np.hstack((color_image, depth_colormap))

#         # Show images
#         cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('RealSense', images)
#         cv2.waitKey(1)

# finally:

#     # Stop streaming
#     pipeline.stop() 
    
    
    
# %%
