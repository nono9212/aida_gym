"""This file implements the gym environment of aida.
"""
import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
from aida_env import bullet_client
from aida_env import aida
import os
import pybullet_data

NUM_SUBSTEPS = 5
NUM_MOTORS = 12
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class AidaBulletEnv(gym.Env):
  """The gym environment for aida.
  It simulates the locomotion of aida, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far aida walks in 1000 steps and penalizes the energy
  expenditure.
  """
  metadata = {
      "render.modes": ["human", "rgb_array"],
      "video.frames_per_second": 50
  }
  def __init__(self,
               commands,   ##(position) commands is an array of [x,y] position
               ##(velocity) commands is an array of [duration,[vx,vy,vz,ax,ay,az]]
               commandtype="position",
               area = "plane",
               urdf_root=pybullet_data.getDataPath(),
               action_repeat=1,
               height_weight=100,
               default_reward = 1.0,
               orientation_weight = 1.0,
               direction_weight = 1.0,
               speed_weight = 1.0,
               type_weight = 10,
               distance_limit=float("inf"),
               observation_noise_stdev=0.0,
               self_collision_enabled=True,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,#not needed to be true if accurate motor model is enabled (has its own better PD)
               leg_model_enabled=False,
               accurate_motor_model_enabled=True,
               motor_kp=0.01,
               motor_kd=1.0,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               hard_reset=True,
               on_rack=False,
               render=False,
               kd_for_pd_controllers=0.3):
               #env_randomizer=aida_env_randomizer.aidaEnvRandomizer()):
    """Initialize aida gym environment.
    Args:
      urdf_root: The path to the urdf data folder.
      action_repeat: The number of simulation steps before actions are applied.
      distance_weight: The weight of the distance term in the reward.
      energy_weight: The weight of the energy term in the reward.
      shake_weight: The weight of the vertical shakiness term in the reward.
      drift_weight: The weight of the sideways drift term in the reward.
      height_weight: The weight of the height of the body in term of reward
      (to avoir walking on knees)
      distance_limit: The maximum distance to terminate the episode.
      observation_noise_stdev: The standard deviation of observation noise.
      self_collision_enabled: Whether to enable self collision in the sim.
      motor_velocity_limit: The velocity limit of each motor.
      pd_control_enabled: Whether to use PD controller for each motor.
      leg_model_enabled: Whether to use a leg motor to reparameterize the action
        space.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_kp: proportional gain for the accurate motor model.
      motor_kd: derivative gain for the accurate motor model.
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction()aida.py for more
        details.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place aida back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place aida on rack. This is only used to debug
        the walking gait. In this mode, aida's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
      kd_for_pd_controllers: kd value for the pd controllers of the motors
      env_randomizer: An EnvRandomizer to randomize the physical properties
        during reset().
    """
    self._time_step = 0.01
    self._action_repeat = action_repeat
    self._num_bullet_solver_iterations = 300
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._observation = []
    self._env_step_counter = 0
    self._is_render = render
    self._last_base_position = [0, 0, 0]
    self._height_weight = height_weight
    self._type_weight = type_weight
    self._distance_limit = distance_limit
    self._observation_noise_stdev = observation_noise_stdev
    self._action_bound = 1
    self._pd_control_enabled = pd_control_enabled
    self._leg_model_enabled = leg_model_enabled
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._motor_kp = motor_kp
    self._motor_kd = motor_kd
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self._cam_dist = 3.0
    self._cam_yaw = 0
    self._cam_pitch = -30
    self._hard_reset = True
    self._kd_for_pd_controllers = kd_for_pd_controllers
    self._last_frame_time = 0.0
    self._commands = commands
    self._commandtype = commandtype
    self._area = area
    self._default_reward = default_reward
    self._currentObjective = 0
    self._orientation_weight = orientation_weight
    self._direction_weight = direction_weight
    self._direction_weight = direction_weight
    self._speed_weight = speed_weight
	
    self._rewardLineID = -1
    self._shapeID = -1

    #self._env_randomizer = env_randomizer
    # PD control needs smaller time step for stability.
    if pd_control_enabled or accurate_motor_model_enabled:
      self._time_step /= NUM_SUBSTEPS
      self._num_bullet_solver_iterations /= NUM_SUBSTEPS
      self._action_repeat *= NUM_SUBSTEPS

    if self._is_render:
      self._pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = bullet_client.BulletClient()
    
    self._seed()
    self.reset()
    observation_high = (
        self.aida.GetObservationUpperBound() + OBSERVATION_EPS)
    observation_low = (
        self.aida.GetObservationLowerBound() - OBSERVATION_EPS)
    action_dim = NUM_MOTORS
    action_high = np.array([self._action_bound] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(observation_low, observation_high)
    self.viewer = None
    self._hard_reset = hard_reset  # This assignment need to be after reset()
    self.render = self._render
    self.step = self._step

 # def set_env_randomizer(self, env_randomizer):
 #   self._env_randomizer = env_randomizer

  def configure(self, args):
    self._args = args

  def reset(self):
      return self._reset()

  def get_commands(self):
      return self._commands

  def _reset(self):
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      path = aida.getDataPath() + "/urdf/" + self._area +".urdf"

      self._pybullet_client.loadURDF(path)
      #self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)

      self._pybullet_client.setGravity(0, 0, -10)
      acc_motor = self._accurate_motor_model_enabled
      motor_protect = self._motor_overheat_protection
      self.aida = (aida.Aida(
          pybullet_client=self._pybullet_client,
          time_step=self._time_step,
          motor_velocity_limit=self._motor_velocity_limit,
          pd_control_enabled=self._pd_control_enabled,
          accurate_motor_model_enabled=acc_motor,
          motor_kp=self._motor_kp,
          motor_kd=self._motor_kd,
          torque_control_enabled=self._torque_control_enabled,
          motor_overheat_protection=motor_protect,
          on_rack=self._on_rack,
          kd_for_pd_controllers=self._kd_for_pd_controllers))
    else:
      self.aida.Reset(reload_urdf=False)

    #if self._env_randomizer is not None:
    #  self._env_randomizer.randomize_env(self)

    self._env_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._pybullet_client.resetDebugVisualizerCamera(
        self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
    if not self._torque_control_enabled:
      for _ in range(100):
        if self._pd_control_enabled or self._accurate_motor_model_enabled:
          self.aida.ApplyAction(-0.5 * np.ones(12))
        self._pybullet_client.stepSimulation()
    ret = self._noisy_observation()
    self._currentObjective = 0
    self.aida.setTarget(self._commands[self._currentObjective])
    pybullet.removeUserDebugItem(self._shapeID)
    self._shapeID = pybullet.addUserDebugLine(self._commands[self._currentObjective]+[0],self._commands[self._currentObjective]+[1],lineColorRGB=[0,1,0.5], lineWidth=10)
    for i in range(50):
        self._step([-1,-1,1,  0,0,1,  -1,-1,1,  0,0,1])  # init
        
    return ret

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _transform_action_to_motor_command(self, action):
    if self._leg_model_enabled:
      for i, action_component in enumerate(action):
        if not (-self._action_bound - ACTION_EPS <= action_component <=
                self._action_bound + ACTION_EPS):
          raise ValueError(
              "{}th action {} out of bounds.".format(i, action_component))
      action = self.aida.ConvertFromLegModel(action)
    return action

  def _step(self, action):
    """Step forward the simulation, given the action.
    Args:
      action: A list of desired motor angles for eight motors.
    Returns:
      observations: The angles, velocities and torques of all motors.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.
    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    if self._is_render:
      # Sleep, otherwise the computation takes less time than real time,
      # which will make the visualization like a fast-forward video.
      time_spent = time.time() - self._last_frame_time
      self._last_frame_time = time.time()
      time_to_sleep = self._action_repeat * self._time_step - time_spent
      if time_to_sleep > 0:
        time.sleep(time_to_sleep)
      base_pos = self.aida.GetBasePosition()
      self._pybullet_client.resetDebugVisualizerCamera(
          self._cam_dist, self._cam_yaw, self._cam_pitch, base_pos)
    action = self._transform_action_to_motor_command(action)
    for _ in range(self._action_repeat):
      self.aida.ApplyAction(action)
      self._pybullet_client.stepSimulation()

    self._env_step_counter += 1
    reward = self._reward()
    done = self._termination()
    return np.array(self._noisy_observation()), reward, done, {}

  def _render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos = self.aida.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._pybullet_client.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def get_aida_motor_angles(self):
    """Get the aida's motor angles.
    Returns:
      A numpy array of motor angles.
    """
    return np.array(
        self._observation[MOTOR_ANGLE_OBSERVATION_INDEX:
                          MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS])

  def get_aida_motor_velocities(self):
        """Get the aida's motor velocities.
        Returns:
        A numpy array of motor velocities.
        """
        return np.array(
            self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX:
                            MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS])

  def get_aida_motor_torques(self):
    """Get the aida's motor torques.
    Returns:
      A numpy array of motor torques.
    """
    return np.array(
        self._observation[MOTOR_TORQUE_OBSERVATION_INDEX:
                          MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS])

  def get_aida_base_orientation(self):
    """Get the aida's base orientation, represented by a quaternion.
    Returns:
      A numpy array of aida's orientation.
    """
    return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

  def is_fallen(self):
    """Decide whether the aida has fallen.
    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the aida is considered fallen.
    Returns:
      Boolean value that indicates whether the aida has fallen.
    """
    orientation = self.aida.GetBaseOrientation()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    local_up = rot_mat[6:]
    pos = self.aida.GetBasePosition()
    return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or
            pos[2] < 0.13)
    
  def _termination(self):
    position = self.aida.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    return self.is_fallen() or distance > self._distance_limit

  def _reward(self):
    current_base_position = self.aida.GetBasePosition()
    distToTarget = self.aida.distToTarget()
    if(np.linalg.norm(distToTarget) < 0.4):
        self._currentObjective = (self._currentObjective+1)%len(self._commands)

        self.aida.setTarget(self._commands[self._currentObjective])
        pybullet.removeUserDebugItem(self._shapeID)
        self._shapeID = pybullet.addUserDebugLine(self._commands[self._currentObjective]+[0],self._commands[self._currentObjective]+[1],lineColorRGB=[0,1,0.5], lineWidth=10)
        #if(self._is_render):
            #self.drawTarget(self.aida._targetPoint)
    

    height_reward = np.exp(-((current_base_position[2]-0.7)**2)/0.15)
    
    orientation = self.aida.GetBaseOrientation()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    local_up = rot_mat[6:]
    orientation_reward = np.exp(-((np.dot(np.asarray([0, 0, 1]), np.asarray(local_up))-1)**2)/0.005)
    
    dirTo = distToTarget 
    dirTo /= np.linalg.norm(dirTo)
    actualDir = rot_mat[:2]
    actualDir /= np.linalg.norm(actualDir)
    direction_reward = np.exp(-((np.dot(actualDir, dirTo)-1)**2)/0.5)
    x = np.dot(np.array(self.aida.GetBaseLinearVelocity()[0:2]),dirTo)
    speed_reward = np.arctan(3*x)/np.pi+0.5


    reward = self._default_reward + self._height_weight*height_reward + self._orientation_weight*orientation_reward + self._direction_weight*direction_reward + self._speed_weight*speed_reward

    reward /=(self._default_reward+self._height_weight+self._orientation_weight+self._orientation_weight+self._direction_weight+self._speed_weight)
    #pybullet.removeUserDebugItem(self._rewardLineID)
    #self._rewardLineID = pybullet.addUserDebugLine([0,0,0],[0,0,reward],lineColorRGB=[0.5,1,0], lineWidth=10)
    
    return reward



  def _get_observation(self):
    self._observation = self.aida.GetObservation()
    return self._observation

  def _noisy_observation(self):
    self._get_observation()
    observation = self._observation
    observation = np.array(observation)
    if self._observation_noise_stdev > 0:
      observation += (np.random.normal(
          scale=self._observation_noise_stdev, size=observation.shape) *
                      self.aida.GetObservationUpperBound())

    return observation
