"""
Inverse kinematics (IK): "I want the torso tilted like THIS, the elbows HERE and the hands
HERE pointing THIS way. Which joint angles do that?"

One solver handles the whole upper body: 3 waist joints + 7 joints in each arm = 17 joints.
It tracks several "tasks" at the same time:

    task                         what is compared                          weight (config.py)
    torso orientation            robot torso rotation vs. your torso       IK_TORSO_WEIGHT
    left / right elbow position  robot elbow vs. elbow target              IK_ELBOW_WEIGHT
    left / right wrist position  robot wrist vs. wrist target              IK_WRIST_WEIGHT
    left / right wrist rotation  robot hand rotation vs. your palm         IK_WRIST_ROTATION_WEIGHT

Method (damped least squares), repeated a few times every frame:
    1. For every task, measure the error (how far off we are)
    2. Ask MuJoCo for that task's Jacobian J: how much it moves when each joint moves a little
    3. Stack all the tasks (multiplied by their weights) into one big J and one big error e
    4. Solve for the small joint change dq that shrinks all the errors as much as possible:
           dq = (J^T J + damping^2 I)^-1 J^T e
    5. Apply dq (but never past a joint limit)
"""
import mujoco
import numpy as np

import config

SIDES = ("left", "right")
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM_JOINTS = (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
)


def orientation_error(target_rot, current_rot):
    """
    The rotation (as axis * angle, world frame) that turns current_rot into target_rot.
    Its length is how many radians we are off by.
    """
    target_quat = np.zeros(4)
    current_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, target_rot.flatten())
    mujoco.mju_mat2Quat(current_quat, current_rot.flatten())

    current_conj = np.zeros(4)
    mujoco.mju_negQuat(current_conj, current_quat)
    error_quat = np.zeros(4)
    mujoco.mju_mulQuat(error_quat, target_quat, current_conj)

    error = np.zeros(3)
    mujoco.mju_quat2Vel(error, error_quat, 1.0)
    return error


class UpperBodyIK:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)   # our own scratch copy, so IK never disturbs the simulation

        self.joint_names = list(WAIST_JOINTS) + [f"{side}_{j}_joint" for side in SIDES for j in ARM_JOINTS]
        joint_ids = [model.joint(name).id for name in self.joint_names]
        self.qpos_ids = model.jnt_qposadr[joint_ids]
        self.dof_ids = model.jnt_dofadr[joint_ids]
        self.lower = model.jnt_range[joint_ids, 0]
        self.upper = model.jnt_range[joint_ids, 1]

        self.torso_id = model.body("torso_link").id
        self.elbow_site = {side: model.site(f"{side}_elbow_site").id for side in SIDES}
        self.wrist_site = {side: model.site(f"{side}_hand_ik_site").id for side in SIDES}

        self.q = np.zeros(len(self.joint_names))   # current answer (next frame starts from it)
        self.errors = {}                           # last error of every task, for printing/tests

        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros((3, model.nv))

    # ---------------------------------------------------------------- Jacobians
    def _site_jacobian(self, site_id):
        mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, site_id)
        return self.jac_pos[:, self.dof_ids].copy(), self.jac_rot[:, self.dof_ids].copy()

    def _body_rotation_jacobian(self, body_id):
        mujoco.mj_jacBody(self.model, self.data, self.jac_pos, self.jac_rot, body_id)
        return self.jac_rot[:, self.dof_ids].copy()

    # ---------------------------------------------------------------- solve
    def solve(self, sim_qpos, targets):
        """
        sim_qpos: the simulation's current qpos (the rest of the robot stays as it is)
        targets:  RobotTargets from the Retargeter (world frame)
        returns:  dict joint name -> angle, for the 17 waist + arm joints
        """
        m, d = self.model, self.data
        d.qpos[:] = sim_qpos

        for _ in range(config.IK_ITERATIONS):
            d.qpos[self.qpos_ids] = self.q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)

            J_rows, error_rows = [], []

            def add_task(name, J, error, weight):
                self.errors[name] = np.linalg.norm(error)
                J_rows.append(weight * J)
                error_rows.append(weight * error)

            # ---- torso orientation (only the waist joints can change it) ----
            torso_error = orientation_error(targets.torso_rot, d.xmat[self.torso_id].reshape(3, 3))
            add_task("torso", self._body_rotation_jacobian(self.torso_id), torso_error, config.IK_TORSO_WEIGHT)

            for side in SIDES:
                # ---- elbow position ----
                site = self.elbow_site[side]
                J_pos, _ = self._site_jacobian(site)
                add_task(f"{side}_elbow", J_pos, targets.elbow_pos[side] - d.site_xpos[site],
                         config.IK_ELBOW_WEIGHT)

                # ---- wrist position and rotation ----
                site = self.wrist_site[side]
                J_pos, J_rot = self._site_jacobian(site)
                add_task(f"{side}_wrist", J_pos, targets.wrist_pos[side] - d.site_xpos[site],
                         config.IK_WRIST_WEIGHT)
                rot_error = orientation_error(targets.wrist_rot[side], d.site_xmat[site].reshape(3, 3))
                add_task(f"{side}_wrist_rotation", J_rot, rot_error, config.IK_WRIST_ROTATION_WEIGHT)

            J = np.vstack(J_rows)
            e = np.concatenate(error_rows)

            # ---- damped least squares ----
            n = len(self.q)
            dq = np.linalg.solve(J.T @ J + config.IK_DAMPING ** 2 * np.eye(n), J.T @ e)

            # don't take giant steps
            biggest = np.max(np.abs(dq))
            if biggest > config.IK_MAX_STEP:
                dq *= config.IK_MAX_STEP / biggest

            # never go past the joint limits
            self.q = np.clip(self.q + dq, self.lower, self.upper)

        return dict(zip(self.joint_names, self.q))

    def reset(self):
        self.q[:] = 0.0
