import time
from typing import Any, Dict

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from .base_sim import BaseSimulator


def init_channel(config: Dict[str, Any]) -> None:
    """
    Initialize the communication channel for simulator/robot communication.

    Args:
        config: Configuration dictionary containing DOMAIN_ID and optionally INTERFACE
    """
    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])


class SimulatorFactory:
    """Factory class for creating different types of simulators."""

    @staticmethod
    def create_simulator(config: Dict[str, Any], env_name: str = "default", **kwargs):
        """
        Create a simulator based on the configuration.

        Args:
            config: Configuration dictionary containing SIMULATOR type
            env_name: Environment name
            **kwargs: Additional keyword arguments for specific simulators
        """
        simulator_type = config.get("SIMULATOR", "mujoco")
        if simulator_type == "mujoco":
            return SimulatorFactory._create_mujoco_simulator(config, env_name, **kwargs)
        else:
            print(
                f"Warning: Invalid simulator type: {simulator_type}. "
                "If you are using run_sim_loop, please ignore this warning."
            )
            return None

    @staticmethod
    def _create_mujoco_simulator(config: Dict[str, Any], env_name: str = "default", **kwargs):
        """Create a MuJoCo simulator instance."""
        camera_configs = kwargs.pop("camera_configs", {})
        if len(camera_configs) > 0:
            print(f"Debug: SimulatorFactory received {len(camera_configs)} camera config(s)")
        
        env_kwargs = dict(
            onscreen=kwargs.pop("onscreen", True),
            offscreen=kwargs.pop("offscreen", False),
            camera_configs=camera_configs,
        )
        return BaseSimulator(config=config, env_name=env_name, **env_kwargs)

