#!/usr/bin/env python3
"""Interactive viser viewer for IK reachability vs the table.

Pick a scene's covering grasp and an (x, yaw) placement; the object is placed on
the table, IK is solved (with the table cuboid), and the robot is shown at the
solution. Text reports IK success and the robot's lowest z vs the table top so
you can SEE which part collides with the table.

    python src/visualization/reach_viser.py --obj pepsi --scene shelf/9 --port 8080
"""
import argparse, sys
import numpy as np, torch, trimesh, yourdfpy
import viser, viser.extras

sys.path.insert(0, "/home/mingi/AutoDex")
import src.visualization.exp as exp
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose


def box_mesh(dims, pose_xyzw):
    b = trimesh.creation.box(extents=dims)
    T = np.eye(4); T[:3, 3] = pose_xyzw[:3]
    b.apply_transform(T)
    return b


def spheres_mesh(centers, radii):
    """One combined mesh of actual icospheres (real radii) for the given spheres."""
    if len(centers) == 0:
        return None
    parts = []
    for cc, rr in zip(centers, radii):
        s = trimesh.creation.icosphere(subdivisions=1, radius=float(rr))
        s.apply_translation(cc)
        parts.append(s)
    return trimesh.util.concatenate(parts)


class ReachViser:
    def __init__(self, obj, hand, scene, port):
        self.obj, self.hand = obj, hand
        self.server = viser.ViserServer(port=port)
        self.pl = GraspPlanner(hand=hand)
        d = exp.compute_coverage_data(self.pl, obj, hand)
        self.va, self.gm, self.sm = d["valid_array"], d["grasp_meta"], d["scene_meta"]
        self.wo, self.preg, self.reach = d["wrist_obj"], d["pregrasp"], d["reach"]  # SAVED reach map
        self.tab = exp.load_tabletop_poses(obj)
        st, sid = scene.split("/")
        si = [i for i, s in enumerate(self.sm) if s[0] == st and str(s[1]) == sid][0]
        self.pose = self.sm[si][2]
        self.cov = [int(g) for g in np.where(self.va[si])[0]]

        # build the two IK solvers ONCE (table + free) and reuse — do NOT re-init
        # cuRobo every slider change (that floods warnings and is slow).
        self.dev = self.pl._tensor_args.device
        self.INIT = torch.tensor(self.pl._init_state, dtype=torch.float32, device=self.dev)
        self.pl._init_ik_solver(_to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}}),
                                use_cuda_graph=False)
        self.solver_table = self.pl._ik_solver
        self.pl._ik_solver = None
        self.pl._init_ik_solver(_to_curobo_world({"mesh": {}, "cuboid": {}}), use_cuda_graph=False)
        self.solver_free = self.pl._ik_solver

        # RobotWorld for the FULL robot's collision spheres (what cuRobo actually
        # checks against the table) -- mesh can look clear while a sphere dips in.
        from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
        from curobo.geom.types import WorldConfig
        rwc = RobotWorldConfig.load_from_config(
            self.pl._robot_cfg,
            WorldConfig.from_dict({"cuboid": {"table": exp.TABLE_CUBOID}, "mesh": {}}),
            collision_activation_distance=0.0, tensor_args=self.pl._tensor_args)
        self.rw = RobotWorld(rwc)

        # robot
        self.urdf = yourdfpy.URDF.load(str(exp.URDF_BY_HAND[hand]))
        self.vurdf = viser.extras.ViserUrdf(self.server, self.urdf, root_node_name="/robot")
        self.jnames = [j.name for j in self.urdf.actuated_joints]

        # table TOP surface as a workspace-sized semi-transparent sheet at
        # z=TABLE_SURFACE_Z (the collision-relevant plane). IK still uses the real
        # full TABLE_CUBOID; this is just so you can SEE the table top and what
        # dips below it. Two layers: a clear sheet + a thin solid rim for contrast.
        # the REAL collision cuboid: real z-extent (0.2 thick, top at TABLE_SURFACE_Z,
        # bottom at top-0.2), x/y clipped to the workspace so it's not a 2x3 wall.
        # Semi-transparent so you can see the robot dip INTO the slab. A solid thin
        # rim marks the exact table-top plane.
        cx, cy, cz = exp.TABLE_CUBOID["pose"][:3]
        dx, dy, cth = exp.TABLE_CUBOID["dims"]
        top = cz + cth / 2
        self.server.scene.add_box("/table_cuboid", color=(150, 110, 70),
                                  dimensions=(dx, dy, cth), position=(cx, cy, cz),
                                  opacity=0.5)
        print(f"TABLE_CUBOID real: x={cx-dx/2:.2f}..{cx+dx/2:.2f} y={cy-dy/2:.2f}..{cy+dy/2:.2f} "
              f"z={cz-cth/2:.3f}..{top:.3f}  (robot base at x=0)", flush=True)

        # object mesh
        self.obj_v = np.asarray(trimesh.load(
            str(exp.MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"), process=False).vertices, np.float32)
        self.obj_f = np.asarray(trimesh.load(
            str(exp.MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"), process=False).faces)
        self.obj_handle = None

        with self.server.gui.add_folder("Reach @ table"):
            self.g_dd = self.server.gui.add_dropdown(
                "grasp", options=[f"{i}:{self.gm[g][2]}" for i, g in enumerate(self.cov)])
            self.x_sl = self.server.gui.add_slider("x", 0, len(exp.X_GRID) - 1, 1, 2)
            self.y_sl = self.server.gui.add_slider("yaw idx", 0, len(exp.YAW_GRID) - 1, 1, 0)
            self.use_table = self.server.gui.add_checkbox("IK uses table", True)
            self.txt = self.server.gui.add_text("status", "")
        for h in (self.g_dd, self.x_sl, self.y_sl, self.use_table):
            h.on_update(lambda _: self.update())
        self.update()

    def _ik(self, W, solver):
        r = solver.solve_batch(_to_curobo_pose(W[None], self.dev),
                               retract_config=self.INIT.unsqueeze(0))
        ok = bool(r.success.view(-1)[0])
        arm = r.solution.cpu().numpy().reshape(-1)[:6]
        return ok, arm

    def update(self):
        gi = int(self.g_dd.value.split(":")[0]); g = self.cov[gi]
        xi, yi = int(self.x_sl.value), int(self.y_sl.value)
        x = float(exp.X_GRID[xi]); yaw = float(exp.YAW_GRID[yi])
        T = exp._place_T(self.tab[self.pose], x, yaw)
        W = T @ self.wo[g]
        ok_t, arm_t = self._ik(W, self.solver_table)   # reused solvers, no re-init
        ok_f, arm_f = self._ik(W, self.solver_free)
        arm = arm_t if self.use_table.value else arm_f
        ok = ok_t if self.use_table.value else ok_f
        saved = bool(self.reach[g, xi, yi])            # SAVED reach map value
        # object on table
        ov = self.obj_v @ T[:3, :3].T + T[:3, 3]
        if self.obj_handle is None:
            self.obj_handle = self.server.scene.add_mesh_simple(
                "/object", ov, self.obj_f, color=(60, 120, 255), opacity=1.0)
        else:
            self.obj_handle.vertices = ov
        # show robot at the solution. If table-IK succeeded, show THAT (clear config).
        # If table-IK failed, show the FREE IK so you can SEE the config penetrating
        # the table (that's WHY table-IK rejected it).
        if ok_t:
            use_arm, showing = arm_t, "table-IK ✓ (clear)"
        elif ok_f:
            use_arm, showing = arm_f, "FREE-IK (table-IK FAILED — this config hits table)"
        else:
            use_arm, showing = None, "no IK at all"
        if use_arm is not None:
            cfg = np.concatenate([use_arm, self.preg[g]])[:len(self.jnames)]
            self.vurdf.update_cfg(np.asarray(cfg, np.float32))
            self.urdf.update_cfg(dict(zip(self.jnames, cfg)))
            sc = self.urdf.scene
            zmin = min(
                (trimesh.transformations.transform_points(np.asarray(gg.vertices), sc.graph.get(nm)[0])[:, 2].min())
                for nm, gg in sc.geometry.items())
            # cuRobo collision spheres at this config (what IK actually checks)
            qt = torch.tensor(cfg[None], dtype=torch.float32, device=self.dev)
            sph = self.rw.get_kinematics(qt).link_spheres_tensor[0].detach().cpu().numpy()  # (N,4)
            sph = sph[sph[:, 3] > 1e-6]
            c, rad = sph[:, :3], sph[:, 3]
            cx, cy, cz_ = exp.TABLE_CUBOID["pose"][:3]
            dx, dy, dz = exp.TABLE_CUBOID["dims"]
            lo = np.array([cx - dx / 2, cy - dy / 2, cz_ - dz / 2])
            hi = np.array([cx + dx / 2, cy + dy / 2, cz_ + dz / 2])
            cl = np.clip(c, lo, hi)                                # closest point on the box
            dist = np.linalg.norm(c - cl, axis=1)
            hit = dist < rad                                       # sphere intersects the cuboid
            mh = spheres_mesh(c[hit], rad[hit])                   # actual spheres, real radii
            mc = spheres_mesh(c[~hit], rad[~hit])
            if mh is not None:
                self.server.scene.add_mesh_simple("/sph_hit", mh.vertices, mh.faces,
                                                  color=(255, 30, 30), opacity=0.95)
            else:
                self.server.scene.add_mesh_simple("/sph_hit", np.zeros((3, 3)),
                                                  np.array([[0, 1, 2]]), color=(255, 30, 30))
            if mc is not None:
                self.server.scene.add_mesh_simple("/sph_clear", mc.vertices, mc.faces,
                                                  color=(0, 200, 0), opacity=0.4)
            nhit = int(hit.sum())
        else:
            zmin = float("nan"); nhit = -1
        self.txt.value = (f"SHOWING: {showing}  |  saved_reach={saved} "
                          f"liveIK table={ok_t} free={ok_f}  |  spheres hitting table: {nhit} "
                          f"(red) | mesh min-z={zmin:.3f} vs table top {exp.TABLE_SURFACE_Z:.3f}")
        print(self.txt.value, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pepsi")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--scene", default="shelf/9")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    ReachViser(args.obj, args.hand, args.scene, args.port)
    print(f"[reach_viser] http://localhost:{args.port}", flush=True)
    import time
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
