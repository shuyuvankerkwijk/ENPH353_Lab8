# Cross-Entropy Inverted Pendulum
Using deep learning and cross-entropy reinforcement learning to train a cartpole to stay upright!

Note: 'gym-gazebo' is an OpenAI gym extension for using Gazebo

### Build and install gym-gazebo and then launch gazebo cartpole

In the root directory of the repository:

```bash
sudo pip install -e .
```

To run the cartpole environment go to directory where gym-gazebo is contained, then run:
```
source enph353_gym-gazebo/gym_gazebo/envs/ros_ws/devel/setup.bash
cd enph353_gym-gazebo/examples/gazebo_cartpole  
python gazebo_cartpole_v0.py
```


We recommend creating an alias to kill those processes.

```bash
echo "alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'" >> ~/.bashrc
```
