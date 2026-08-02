from enum import Enum
from typing import Union
import numpy as np

import jax.numpy as jnp
import jax_dataclasses as jdc

from lotf import LOTF_PATH



class RefTrajNames(Enum):
    CIRCLE = "circle"
    FIG8 = "fig8"
    STAR = "star"


### csv file paths ###
CIRCLE_CSV = LOTF_PATH + "/objects/ref_traj_files/circle.csv"
FIG8_CSV = LOTF_PATH + "/objects/ref_traj_files/fig8.csv"
STAR_CSV = LOTF_PATH + "/objects/ref_traj_files/star.csv"


@jdc.pytree_dataclass
class ReferenceTraj:
    """
    Represents a reference trajectory.
    Use `from_name` to load a predefined reference trajectory or
    `from_csv` to load a reference trajectory from a csv file.

    >>> ref_traj_obj = ReferenceTraj.from_name(RefTrajNames.CIRCLE)
    """

    ref_traj: jnp.array
    num_waypoints: int
    pos_bounds: jnp.array
    vel_bounds: jnp.array

    @classmethod
    def from_name(cls, name: Union[str, RefTrajNames]) -> "ReferenceTraj":

        if isinstance(name, RefTrajNames):
            name = name.value

        if name == RefTrajNames.CIRCLE.value:
            return ReferenceTraj.from_csv(CIRCLE_CSV)
        elif name == RefTrajNames.FIG8.value:
            return ReferenceTraj.from_csv(FIG8_CSV)
        elif name == RefTrajNames.STAR.value:
            return ReferenceTraj.from_csv(STAR_CSV)
        else:
            raise ValueError(f"Unknown track name: {name}")

    @classmethod
    def from_csv(cls, path: str) -> "ReferenceTraj":

        ref_traj = jnp.array(np.loadtxt(path))
        assert ref_traj.shape[1] == 30, f"Expected 30 columns in trajectory, got {ref_traj.shape[1]}"
        num_waypoints = ref_traj.shape[0]

        # compute position and velocity bounds
        pos_bounds = jnp.array([
            jnp.min(ref_traj[:, TrajColumns.POS.slice], axis=0),
            jnp.max(ref_traj[:, TrajColumns.POS.slice], axis=0)
        ])
        vel_bounds = jnp.array([
            jnp.min(ref_traj[:, TrajColumns.VEL.slice], axis=0),
            jnp.max(ref_traj[:, TrajColumns.VEL.slice], axis=0)
        ])

        # noinspection PyArgumentList
        return cls(
            ref_traj,
            num_waypoints,
            pos_bounds,
            vel_bounds
        )

    @classmethod
    def default_traj(cls) -> "ReferenceTraj":
        cls.from_name(RefTrajNames.CIRCLE)



class TrajColumns(Enum):
    TIME = (0, 1)
    POS = (1, 4)
    QUAT = (4, 8)
    VEL = (8, 11)
    OMEGA = (11, 14)
    ACC = (14, 17)
    ALPHA = (17, 20)
    COMMANDS = (20, 24)
    JERK = (24, 27)
    SNAP = (27, 30)

    @property
    def start(self):
        return self.value[0]

    @property
    def end(self):
        return self.value[1]

    @property
    def slice(self):
        return slice(self.start, self.end)
