## About
This is my implementation in TensorFlow of the [Advantage Actor Critic (A2C)] algorithm, a reinforcement learning algorithm that can learn to control high-dimensional continuous action spaces to maximize long-term reward in the agent's environment. Scroll down for a video of the results.

## Algorithm
The agent is controlled by a deep convolutional policy network μ(s), which maps states to a specific action. This is updated to maximize the expected return predicted by the action-value network Q(s,a). In each state s<sub>t</sub> the agent takes action a<sub>t</sub> and receives a scalar reward r<sub>t</sub> whilst transitioning to state s<sub>t+1</sub>. The state-action value can then be learned by the recursive relationship:

<p align="center">Q<sup>μ</sup>(s<sub>t</sub>,a<sub>t</sub>) = <i>E</i><sub>(s<sub>t+1</sub>∼E)</sub>[r(s<sub>t</sub>,a<sub>t</sub>) + γQ<sup>μ</sup>(s<sub>t+1</sub>, μ(s<sub>t+1</sub>))]</p>

## Implementation Details
* Action is repeated for 3 simulation timesteps, to allow the agent to infer velocities from frame differences.
* 3 convolutional layers shared by both critic and actor networks.
* 2 fully-connected layers of 500 nodes per critic and actor network.
* Batch normalization before the Relu activation of every hidden layer.
* Linear output layer for critic, and tanh/softmax activations for actor output.
* Target networks are used to give a stable target Q-value, which prevents the value network from diverging.
* Minibatches are sampled from experience replay memory which is stored in a circular buffer of recorded sequences.
* Experience replay sampling is prioritized according to TD error, so that learning is focused on samples with the most unexpected return value.
* Sequences can either be created by following the policy, or supplied by human expert demonstration to bootstrap the learning process.

## Results
On a GTX 1070, the following policy is learnt in the OpenAI environment CarRacing-v0:
![CarRacing-v0](https://raw.githubusercontent.com/michaelrgb/bin/master/CarRacing-v0_93b51c7399d0b11080f9a2245533646498312406.gif)

## Future Work
I am experimenting with adding self-attention modules after the convolutional layers. Self-attention will allow the network to learn [global relationships](https://arxiv.org/abs/1806.01830) between entities in the scene, as opposed to just local coincidence detectors learnt by purely convolutional networks which only give translation invariance. This could allow the RL agent to better generalize to changes in the environment.

## Dependencies:
```bash
pip install gym # includes CarRacing-v0
pip install tensorflow-gpu
```

## Usage
Learning is performed on the server, which accepts a client connection for each agent. First we start the server:
```bash
export ENV=CarRacing-v0
python rltf.py
tensorboard --logdir=/tmp/tf
```
Then we start one client instance that is on-policy, and one that takes an exploratory policy:
```bash
python rltf.py --inst 1 & python rltf.py --inst 2 --sample_action 0.1
```

## Optional environments
```bash
# FlappyBird requires PLE and gym-ple:
export ENV=FlappyBird-v0
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .
git clone https://github.com/lusob/gym-ple.git
cd gym-ple/
pip install -e .
```
```bash
export ENV=AntBulletEnv-v0
pip install pybullet
```

