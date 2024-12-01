Clone the Git repo:
   git clone https://github.com/ajitjha14/Zips-buttons-identification-in-textiles.git
Install Ubuntu 22.04.
Install ROS 2 Humble.
Install the necessary Interbotix packages following the Trossen Robotics documentation.
Launch the robot description using the following command:
   ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx250s
Run the script to detect and interact with bounding boxes of detected objects:
   python3 Button_Boundingbox_final.py
For more information about the robot and nessicary package check the Trossen Robotics documentation:
https://docs.trossenrobotics.com/interbotix_xsarms_docs/
