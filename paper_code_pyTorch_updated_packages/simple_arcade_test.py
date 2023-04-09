import gymnasium as gym
from agents import *
from config import *
from envs import *
import random
env_id = "BreakoutNoFrameskip-v4"
# env = gym.make(env_id, render_mode="human")
# input_size = env.observation_space.shape  # 4
# output_size = env.action_space.n - 1  # 2
input_size = 4
output_size = 1

is_render = True
model_path = 'models/{}.model'.format(env_id)

# Load config values
use_cuda = False
use_gae = default_config.getboolean('UseGAE')
use_noisy_net = default_config.getboolean('UseNoisyNet')
lam = float(default_config['Lambda'])
num_worker = 1
num_step = int(default_config['NumStep'])
ppo_eps = float(default_config['PPOEps'])
epoch = int(default_config['Epoch'])
mini_batch = int(default_config['MiniBatch'])
batch_size = int(num_step * num_worker / mini_batch)
learning_rate = float(default_config['LearningRate'])
entropy_coef = float(default_config['Entropy'])
gamma = float(default_config['Gamma'])
clip_grad_norm = float(default_config['ClipGradNorm'])
# Create agent and load model file from disk
agent = ICMAgent(
    input_size,
    output_size,
    num_worker,
    num_step,
    gamma,
    lam=lam,
    learning_rate=learning_rate,
    ent_coef=entropy_coef,
    clip_grad_norm=clip_grad_norm,
    epoch=epoch,
    batch_size=batch_size,
    ppo_eps=ppo_eps,
    use_cuda=use_cuda,
    use_gae=use_gae,
    use_noisy_net=use_noisy_net
)
#agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
# Wrap environment in some basic I/O functions
#env = AtariEnvironmentSimple(env_id, is_render)
env = gym.make(env_id, render_mode="human")
env.close()
env.reset()
states = np.zeros([num_worker, 4, 84, 84])
max_steps = 1000  # A few seconds of playtime when rendering
for i in range(max_steps):
    # Get next action
    #actions, value, policy = agent.get_action(np.float32(states) / 255.)
    action = random.randint(0,3)
    # Run environment for a single step
    #history, reward, force_done, done, log_reward = env.step(action)
    s, reward, terminated, truncated, info = env.step(action)
    # Mirror process with multiple workers even though there is only one here
    # next_states = []
    # next_states.append(history.reshape([4, 84, 84]))
    # next_states = np.stack(next_states)
    # states = next_states[:, :, :, :]
