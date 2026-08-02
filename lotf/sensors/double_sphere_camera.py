from enum import Enum
from typing import Union
import yaml

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax.scipy.spatial.transform import Rotation

from lotf.utils.pytrees import field_jnp, CustomPyTree
from lotf import LOTF_PATH


class CameraNames(Enum):
    """Enumeration of available camera configuration names"""
    EXAMPLE_CAM = "example_cam"


EXAMPLE_CAM = LOTF_PATH + "/sensors/camera_files/example_cam.yaml"


@jdc.pytree_dataclass
class CameraState(CustomPyTree):
    """
    Represents the extrinsic state of a camera.
    
    Attributes:
        p_CW: world position in camera frame
        R_CW: rotation matrix from world to camera frame
    """
    p_CW: jnp.ndarray = field_jnp([0.0, 0.0, 0.0])
    R_CW: jnp.ndarray = field_jnp(jnp.eye(3))


class DoubleSphereCamera:
    """
    Implementation of the double sphere camera model for wide-angle lenses.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        alpha: float,
        xi: float,
        width: int,
        height: int,
    ):
        """Initializes camera with intrinsic parameters and resolution"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.alpha = alpha
        self.xi = xi
        self.width = width
        self.height = height

        # default pitch angle
        self.pitch = 30.

    @property
    def pitch(self):
        """Pitch angle in degrees"""
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        """Sets pitch and updates the camera-to-body rotation matrix"""
        self._pitch = value
        # camera points in x direction of quad frame
        rot_CprimeB = Rotation.from_euler(
            "XYZ", jnp.array([90, 0, 90]), degrees=True
        )
        # apply pitch rotation
        rot_cam = Rotation.from_euler("Y", jnp.array(self._pitch), degrees=True)
        # combine rotations for camera to body frame
        self.rot_CB = rot_CprimeB * rot_cam

    @classmethod
    def from_name(cls, name: Union[str, CameraNames]) -> "DoubleSphereCamera":
        """Instantiates a camera using a predefined name or enum"""
        if isinstance(name, CameraNames):
            name = name.value

        if name == "example_cam":
            return cls.from_yaml(EXAMPLE_CAM)
        else:
            raise ValueError(f"Unknown camera name: {name}")

    @classmethod
    def from_yaml(cls, path: str) -> "DoubleSphereCamera":
        """Loads camera configuration from a yaml file"""
        with open(path) as stream:
            try:
                config = yaml.safe_load(stream)
                return cls.from_dict(config)
            except yaml.YAMLError as exc:
                raise exc

    @classmethod
    def from_dict(cls, config: dict) -> "DoubleSphereCamera":
        """Parses configuration dictionary into camera parameters"""
        return cls(
            xi=config["cam0"]["intrinsics"][0],
            alpha=config["cam0"]["intrinsics"][1],
            fx=config["cam0"]["intrinsics"][2],
            fy=config["cam0"]["intrinsics"][3],
            cx=config["cam0"]["intrinsics"][4],
            cy=config["cam0"]["intrinsics"][5],
            width=config["cam0"]["resolution"][0],
            height=config["cam0"]["resolution"][1],
        )

    def project_points(
        self, points: jax.Array, camera_state: CameraState
    ) -> jax.Array:
        """
        Projects 3d world points onto the 2d image plane.

        Args:
            points: Nx3 array of points in world frame
            camera_state: current extrinsic state of the camera
        Returns:
            Nx3 array (u, v, mask) where mask indicates if projection is valid
        """
        # transform points to camera coordinates
        rot_CW = Rotation.from_matrix(camera_state.R_CW)
        points_C = rot_CW.apply(points) + camera_state.p_CW

        # calculate norms and intermediate points for double sphere model
        d1 = jnp.linalg.norm(points_C, axis=1)
        points_C_zxi = points_C.at[:, 2].add(d1 * self.xi)
        d2 = jnp.linalg.norm(points_C_zxi, axis=1)

        # calculate divisor for projection
        div = self.alpha * d2 + (1 - self.alpha) * points_C_zxi[:, 2]

        # compute projected pixel coordinates
        u = self.fx * (points_C[:, 0] / div) + self.cx
        v = self.fy * (points_C[:, 1] / div) + self.cy

        # determine validity bounds based on alpha
        w1 = jax.lax.select(
            self.alpha <= 0.5,
            self.alpha / (1 - self.alpha),
            (1 - self.alpha) / self.alpha,
        )
        # calculate max angle/distance validity
        w2 = (w1 + self.xi) / jnp.sqrt(2 * w1 * self.xi + self.xi**2 + 1)
        
        predicates = jnp.array(
            [
                # check if point is in front of the camera
                points_C[:, 2] > 0,
                # check if projection is valid for the model geometry
                points_C[:, 2] > -w2 * d1,
                # check if the point falls within image boundaries
                u >= 0,
                u < self.width,
                v >= 0,
                v < self.height,
            ]
        )
        # aggregate all validity checks
        valid = jnp.all(predicates, axis=0)

        # combine projected points and validity flag into Nx3
        projected_points = jnp.column_stack((u, v, valid.astype(float)))

        return projected_points

    def update_pose(
        self, state: CameraState, p_WB: jax.Array, R_WB: jax.Array
    ) -> CameraState:
        """
        Updates the camera extrinsic state based on the body pose in world frame.
        """
        rot_WB = Rotation.from_matrix(R_WB)
        # get rotation from world to body
        rot_BW = rot_WB.inv()
        # chain rotations to get world to camera
        rot_CW = self.rot_CB * rot_BW
        # transform world origin to camera frame
        p_CW = rot_CW.apply(-p_WB)

        # return updated camera state dataclass
        state_new = state.replace(p_CW=p_CW, R_CW=rot_CW.as_matrix())

        return state_new

    def project_points_with_pose(
        self, points: jax.Array, p_WB: jax.Array, R_WB: jax.Array
    ) -> jax.Array:
        """
        Convenience method to project points given the body's world pose.
        """
        # initialize temporary identity state
        state = CameraState(p_CW=jnp.zeros(3), R_CW=jnp.eye(3))
        # calculate extrinsics from body pose
        state_new = self.update_pose(state, p_WB, R_WB)
        # project using calculated extrinsics
        return self.project_points(points, state_new)
