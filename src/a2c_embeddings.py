from torch import nn
import torch as th
from gym import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

"""
To use: pass the ActorCriticIcmPolicy class to the A2C constructor, along with policy kwargs
containing the feature kwarg for the ICM embeddings.
"""


class IcmCnn(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    :param icm_embeddings: If not None, use this as the embedding layer. Must match input and
        output dimensions from Nature CNN paper.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = False,
        icm_embeddings=None,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        if icm_embeddings is not None:
            self.model = icm_embeddings
        else:
            print(f"Constructing ICM CNN with {n_input_channels} input channels")
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32,
                          kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with th.no_grad():
                n_flatten = self.cnn(th.as_tensor(
                    observation_space.sample()[None]).float()).shape[1]

            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim), nn.ReLU())
            self.model = nn.Sequential(self.cnn, self.linear)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.model(observations)
