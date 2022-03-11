REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .noise_parallel_runner import NoiseParallelRunner
REGISTRY["noise_parallel"] = NoiseParallelRunner

from .curiosity_episode_runner import CuriosityEpisodeRunner
REGISTRY["curiosity_episode"] = CuriosityEpisodeRunner