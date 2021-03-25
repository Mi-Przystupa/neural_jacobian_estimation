# Analyzing Neural Jacobian Methods in Applications of Visual Servoing and Kinematic Control

This repository contains all the code we used in our submission to ICRA 2021

This repository is "as is" for now, but will be updated coming soon to be a bit more readable and easier to use for those interested.


# Stuff from Old Repository

This repo is a copy of the necessary portions to run our implementations, because the original repository has some stuff we wanted to keep private

## Runnning Ros code in existing Catkin_workspace

To run the ros stuff you need 3 terminals doing 3 different things:

In Terminal 1: just run "roscore":
```
roscore
```
In Terminal 2: 
```
roslaunch kortex_driver kortex_driver.launch gripper:=robotiq_2f_85 start_rviz:=false ip_address:=192.168.1.12
```
the stuff after .launch are optional arguments [details here](https://github.com/Kinovarobotics/ros_kortex/tree/kinetic-devel/kortex_driver)

In Terminal 3:
do the rosrun command after building in a catkin workspace
```
catkin_make # in your workspace
rosrun <name of codebase> <script to run> 

```
if after running catkin_make and trying to run files you find that you still cannot. Make sure the python script is set to be executable. So something like this:
```
chmod +x <file-you-want-to-run.py>
```

# Optional for Running camera

to run camera you can use following:

```
rosrun UncalibratedVisualServoingLearning camera_publisher.py 
```
