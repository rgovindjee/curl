import gymnasium
import PyFlyt.gym_envs # noqa

env = gymnasium.make(
  "PyFlyt/QuadX-Hover-v0",
  flight_dome_size = 3.0,
  max_duration_seconds = 1000.0,
  angle_representation = "quaternion",
  agent_hz = 40,
  render_mode = "human"
)
observation, info = env.reset()

for _ in range(1000000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

if terminated or truncated:
    observation, info = env.reset()

env.close()
