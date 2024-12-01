Zips and Buttons Identification in Textiles  
======================================
This project focuses on detecting zips and buttons in textiles using a robotic arm and computer vision techniques. Follow the steps below to set up and run the project.  
Steps to run tbe project:
1.Clone the Git repo:  
   git clone https://github.com/ajitjha14/Zips-buttons-identification-in-textiles.git  
2. Install Ubuntu 22.04 native, Interbotix packages might not work correctly on virtual machine.  
3. Install ROS 2 Humble.  
4. Install the necessary Interbotix packages following the Trossen Robotics documentation.  
5. Launch the robot description using the following command:  
   ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wx250s  
6. Run the script to detect and interact with bounding boxes of detected objects:  
   python3 Button_Boundingbox_final.py  
     
For more information about the robot and nessicary package check the Trossen Robotics documentation:

https://docs.trossenrobotics.com/interbotix_xsarms_docs/  
