"""Scene + trajectory visualizer using paradex ViserViewer.

Shows scene_cfg (cuboids, meshes), the planned trajectory, and grasp hand poses.
No dependency on rsslib — uses paradex.visualization directly.
"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from paradex.visualization.visualizer.viser import ViserViewer

from autodex.utils.path import urdf_path
from autodex.utils.conversion import cart2se3
import os

_ASSET_ROOT = os.path.join(os.path.expanduser("~"), "shared_data", "AutoDex", "content", "assets", "robot")

HAND_URDF = {
    "allegro": (
        os.path.join(_ASSET_ROOT, "allegro_description", "xarm_allegro.urdf"),
        os.path.join(_ASSET_ROOT, "allegro_description", "allegro_hand_description_right.urdf"),
    ),
    "inspire": (
        os.path.join(_ASSET_ROOT, "inspire_description", "xarm_inspire.urdf"),
        os.path.join(_ASSET_ROOT, "inspire_description", "inspire_hand_right.urdf"),
    ),
    "inspire_left": (
        os.path.join(_ASSET_ROOT, "inspire_left_description", "xarm_inspire_left.urdf"),
        os.path.join(_ASSET_ROOT, "inspire_description", "inspire_hand_left.urdf"),
    ),
}


class ScenePlanVisualizer(ViserViewer):
    """Visualize scene_cfg + PlanResult in viser.

    Usage:
        vis = ScenePlanVisualizer(scene_cfg, plan_result)
        vis.start_viewer(use_thread=True)
        # ... later
        vis.start_viewer(use_thread=False)  # blocks
    """

    def __init__(self, scene_cfg, plan_result=None, port=8080, hand="allegro"):
        super().__init__(port_number=port)
        self.port = port
        self.scene_cfg = scene_cfg
        self.plan_result = plan_result
        self._urdf_full, self._urdf_hand = HAND_URDF.get(hand, HAND_URDF["allegro"])

        self._add_scene()

        if plan_result is not None and plan_result.success:
            self._add_trajectory(plan_result)

    def _pose7d_to_se3(self, pose_list):
        se3 = np.eye(4)
        se3[:3, 3] = pose_list[:3]
        wxyz = pose_list[3:7]
        se3[:3, :3] = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
        return se3

    def _add_scene(self):
        # Cuboids
        for name, data in self.scene_cfg.get("cuboid", {}).items():
            dims = data["dims"]
            pose_se3 = self._pose7d_to_se3(data["pose"])
            box = trimesh.creation.box(extents=dims)
            self.add_object(f"cuboid_{name}", box, pose_se3)
            self.change_color(f"cuboid_{name}", [0.5, 0.5, 0.5, 0.4])

        # Meshes
        for name, data in self.scene_cfg.get("mesh", {}).items():
            pose_se3 = self._pose7d_to_se3(data["pose"])
            mesh_path = data["file_path"]
            # NOTE: do NOT substitute raw_mesh — its origin may differ from
            # the planning simplified.obj, making the obj appear offset from
            # where the planner thinks it is. Use scene_cfg's mesh as-is.
            # process=False preserves materials/texture in trimesh load.
            try:
                mesh = trimesh.load(mesh_path, process=False)
            except Exception:
                mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            self.add_object(f"mesh_{name}", mesh, pose_se3)
            # Only override color for non-target meshes (obstacles).
            # Target keeps its native texture/material.
            if name != "target":
                self.change_color(f"mesh_{name}", [0.6, 0.4, 0.2, 0.5])

    def _add_trajectory(self, result):
        """Add trajectory robot + grasp hand + auto-playing playback."""
        # Full robot trajectory
        self.add_robot("traj_robot", self._urdf_full)
        traj = result.traj

        # Grasp hand at wrist (target — green, semi-transparent)
        urdf_hand = self._urdf_hand
        self.add_robot("grasp_hand", urdf_hand, pose=result.wrist_se3)
        self.robot_dict["grasp_hand"].update_cfg(result.grasp_pose)
        self.change_color("grasp_hand", [0.0, 1.0, 0.0, 0.6])

        # Register the trajectory with the base-class player so the
        # built-in "Playing" checkbox auto-advances through frames.
        # add_player() creates gui_timestep/gui_playing/gui_framerate.
        self.add_player()
        self.add_traj("plan", {"traj_robot": traj})
        self.update_scene(0)   # snap robot to start of trajectory

    def start_viewer(self, use_thread=False):
        self.add_frame("base_frame", np.eye(4))
        self._running = True
        if use_thread:
            import threading
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
        else:
            self._loop()

    def stop_viewer(self):
        """Signal the loop thread to exit, then close the viser server."""
        self._running = False
        import time
        time.sleep(0.05)   # let loop thread observe the flag
        try:
            self.server.stop()
        except Exception:
            pass

    def _loop(self):
        """Auto-advance trajectory when gui_playing is True (base class update()).

        update() already paces itself via gui_framerate, so don't add extra
        sleeps here — that would make playback jerky.
        """
        import time
        print(f"Visualizer running at http://localhost:{self.port}")
        while getattr(self, "_running", True):
            try:
                if hasattr(self, "gui_playing"):
                    self.update()       # paces at 1/gui_framerate.value
                else:
                    time.sleep(0.1)
            except RuntimeError as e:
                # Server closed mid-update — exit loop cleanly.
                if "Event loop is closed" in str(e):
                    return
                raise

    def add_candidates(self, wrist_se3, grasp_pose, filtered, ik_failed=None):
        """Show candidate hands with slider.
        Red = filtered (backward/collision), Yellow = IK-failed (passed filter
        but wrist unreachable), Green = fully valid (filter+IK ok).
        ``ik_failed`` is optional; if None, treated as all-False (= 2-color)."""
        self._cand_wrist = wrist_se3
        self._cand_grasp = grasp_pose
        self._cand_filtered = filtered
        n = len(wrist_se3)
        if ik_failed is None:
            import numpy as _np
            ik_failed = _np.zeros(n, dtype=bool)
        self._cand_ik_failed = ik_failed

        urdf_hand = self._urdf_hand
        self.add_robot("cand_hand", urdf_hand)
        self.robot_dict["cand_hand"].set_visibility(False)

        n_filtered = int(sum(1 for i in range(n) if filtered[i]))
        n_ikfail = int(sum(1 for i in range(n)
                            if (not filtered[i]) and ik_failed[i]))
        n_valid = n - n_filtered - n_ikfail

        with self.server.gui.add_folder("Candidates"):
            self.server.gui.add_text(
                "Stats",
                initial_value=(f"Green(valid+IK ok): {n_valid} | "
                               f"Yellow(IK fail): {n_ikfail} | "
                               f"Red(filtered): {n_filtered} | Total: {n}"),
                disabled=True,
            )
            self._cand_slider = self.server.gui.add_slider(
                "Candidate #", min=0, max=n - 1, step=1, initial_value=0,
            )
            self._cand_label = self.server.gui.add_text(
                "Status", initial_value="", disabled=True,
            )

        self._update_candidate(0)

        @self._cand_slider.on_update
        def _on_cand(_):
            self._update_candidate(int(self._cand_slider.value))

    def _update_candidate(self, idx):
        pose = self._cand_wrist[idx]
        self.robot_dict["cand_hand"].set_visibility(True)
        self.robot_dict["cand_hand"]._visual_root_frame.position = pose[:3, 3]
        self.robot_dict["cand_hand"]._visual_root_frame.wxyz = R.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self.robot_dict["cand_hand"].update_cfg(self._cand_grasp[idx])

        if self._cand_filtered[idx]:
            status = "FILTERED"
            color = [1, 0, 0, 0.6]            # red
        elif self._cand_ik_failed[idx]:
            status = "IK_FAIL"
            color = [1, 1, 0, 0.6]            # yellow
        else:
            status = "VALID"
            color = [0, 1, 0, 0.6]            # green
        self.change_color("cand_hand", color)
        self._cand_label.value = f"#{idx}: {status}"

    def add_frame(self, name, pose):
        self.server.scene.add_frame(
            f"/frames/{name}",
            position=pose[:3, 3],
            wxyz=R.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]],
            axes_length=0.1,
            axes_radius=0.003,
        )