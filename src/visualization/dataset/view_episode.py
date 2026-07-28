"""Browse recorded episode trajectories interactively (bodex-viewer style).

Pick an episode from dropdowns (set / object / timestamp) and scrub through the
actual recorded motion. Two xarm_allegro robots play in sync, one per stream:

    action    : arm = action,  hand = action    (commanded)
    position  : arm = state,   hand = state      (measured)

Uses the video-synced, URDF-joint-order streams under ``{trial}/arm`` and
``{trial}/hand`` (produced by ``src/dataset/sync_arm_hand.py``); arm and hand
share the same frame index, so no per-viewer resampling/remapping is needed.

Run in the ``mingi`` conda env:
    python src/visualization/dataset/view_episode.py
"""

import json
import os

import numpy as np

from paradex.visualization.visualizer.viser import ViserViewer

ROOTS = {
    "clean": "/home/mingi/shared_data/autodex_dataset/selected_100",
    "corl": "/home/mingi/shared_data/autodex_dataset/corl_selected_100",
    "wireout": "/home/mingi/shared_data/autodex_dataset/selected_100_wireout",
}
SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
URDF = ("/home/mingi/shared_data/AutoDex/content/assets/robot/"
        "allegro_description/xarm_allegro.urdf")

COLOR_ACTION = [90, 200, 120]    # green  (commanded)
COLOR_POSITION = [220, 100, 90]  # red    (measured)

# allegro rest/init pose in URDF (joint_0.0..15.0) order — the hand holds this
# before the approach command (recorded scrambled, so we substitute it back).
ALLEGRO_INIT_URDF = np.array(
    [0.0, 1.5707, 0.0, 0.0, 0.0, 1.5707, 0.0, 0.0,
     0.0, 1.5707, 0.0, 0.0, 1.24565697, 0.05513508, 0.23153956, -0.02217758])


class EpisodeViewer(ViserViewer):
    def __init__(self):
        super().__init__()
        self.add_robot("action", URDF)
        self.add_robot("position", URDF)
        for name, col in (("action", COLOR_ACTION), ("position", COLOR_POSITION)):
            self.change_color(name, col)  # viewer-level: (robot_name, rgb 0-255)

        with self.server.gui.add_folder("Episode Viewer"):
            self.root_sel = self.server.gui.add_dropdown(
                "Set", options=list(ROOTS), initial_value="wireout")
            self.obj_sel = self.server.gui.add_dropdown("Object", options=[], initial_value="")
            self.ts_sel = self.server.gui.add_dropdown("Episode", options=[], initial_value="")
            self.show_action = self.server.gui.add_checkbox("action (green)", initial_value=True)
            self.show_position = self.server.gui.add_checkbox("position (red)", initial_value=True)
            self.info = self.server.gui.add_text("Info", initial_value="", disabled=True)

        self.gui_playing.value = True

        @self.root_sel.on_update
        def _(_e): self._on_root()

        @self.obj_sel.on_update
        def _(_e): self._on_obj()

        @self.ts_sel.on_update
        def _(_e): self._load_episode()

        @self.show_action.on_update
        def _(_e): self.robot_dict["action"].set_visibility(self.show_action.value)

        @self.show_position.on_update
        def _(_e): self.robot_dict["position"].set_visibility(self.show_position.value)

        self._on_root()

    def _root(self):
        return ROOTS[self.root_sel.value]

    def _dirs(self, path):
        if not os.path.isdir(path):
            return []
        return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))

    def _on_root(self):
        objs = self._dirs(self._root())
        self.obj_sel.options = objs or ["(none)"]
        if objs:
            self.obj_sel.value = objs[0]
        self._on_obj()

    def _on_obj(self):
        ts = self._dirs(os.path.join(self._root(), self.obj_sel.value))
        self.ts_sel.options = ts or ["(none)"]
        if ts:
            self.ts_sel.value = ts[0]
        self._load_episode()

    def _load_episode(self):
        self.clear_traj()
        obj, ts = self.obj_sel.value, self.ts_sel.value
        trial = os.path.join(self._root(), obj, ts)
        # Synced, video-frame-aligned, URDF-joint-order streams (see
        # src/dataset/sync_arm_hand.py). arm and hand share the same index.
        need = ("arm/state.npy", "arm/action.npy", "hand/state.npy", "hand/action.npy")
        if not all(os.path.exists(os.path.join(trial, p)) for p in need):
            self.info.value = f"{obj}/{ts}: no synced arm/hand (run sync_arm_hand)"
            return

        # arm/action.npy is the cartesian command (lift frames = wrist_se3, |q|>2pi);
        # arm/action_qpos.npy is the IK-fixed joint qpos (lift converted to joints)
        # -- read that so the action robot shows the real commanded lift.
        aq_path = os.path.join(trial, "arm", "action_qpos.npy")
        arm_act = np.load(aq_path if os.path.exists(aq_path)
                          else os.path.join(trial, "arm", "action.npy"))   # commanded (F,6)
        arm_pos = np.load(os.path.join(trial, "arm", "state.npy"))    # measured  (F,6)
        h_act = np.load(os.path.join(trial, "hand", "action.npy"))    # commanded (F,16) URDF order
        h_pos = np.load(os.path.join(trial, "hand", "state.npy"))     # measured  (F,16) URDF order

        # ----- action robot: clean queue poses (position robot stays raw) -----
        arm_act = arm_act.copy()
        h_act = h_act.copy()
        # hand: pre-command blank frames hold allegro_init in JS order (scrambled
        # by ACT_MAP) -> overwrite the constant prefix with the correct init.
        chg = np.where(np.abs(np.diff(h_act, axis=0)).max(1) > 0.02)[0]
        if len(chg):
            h_act[: chg[0] + 1] = ALLEGRO_INIT_URDF
        # arm: action_qpos = IK(wrist) during lift -> garbage branch/failure
        # frames (|q|>2pi). Replace those with the measured joints (clean).
        bad = (np.abs(arm_act) > 2 * np.pi).any(1)
        arm_act[bad] = arm_pos[bad]

        q_action = np.concatenate([arm_act, h_act], axis=1)     # (F, 22)
        q_position = np.concatenate([arm_pos, h_pos], axis=1)

        self.add_traj("episode", {"action": q_action, "position": q_position})

        rj = os.path.join(SRC_ROOT, obj, ts, "result.json")
        meta = json.load(open(rj)) if os.path.exists(rj) else {}
        wire = "WIREOUT" if self.root_sel.value == "wireout" else ""
        self.info.value = (f"{obj}/{ts} | grasp={meta.get('scene_info')} "
                           f"| success={meta.get('success')} {wire} "
                           f"| {len(q_action)} frames")


if __name__ == "__main__":
    EpisodeViewer().start_viewer()
