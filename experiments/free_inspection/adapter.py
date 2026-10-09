"""Independent joint-space view search with XYZ motion and opaque-cup occlusion.

Predicted apple poses are assigned only to disposable planning MjData. Actual
fruit moves exclusively through ContactTransport's existing contact physics.
"""
from itertools import product
import math

import numpy as np

from aple.execution.goal_builder import Pose3D, compose, inverse
from aple.execution.inspect_skill import _axis_angle, _multiply
from aple.hal.in_hand_simulation import CentralInspectionAdapter
from aple.hal.simulation_optimization import native_route_clear
from aple.hal.simulation_timing import stage_timer
from aple.inspection.coloration import fibonacci_sphere, params_from_config
from aple.inspection.in_hand_provider import pose_to_matrix
from aple.execution.motion_executor import InspectionPhaseResult, ViewVerdict


class FreeInspectionAdapter(CentralInspectionAdapter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.search_reports=[]
        self.selected=[]

    def matrices(self,joints):
        from piper_robot import kinematics
        geometry=kinematics.geometry()
        transforms=kinematics.link_transforms(np.asarray(joints),geometry)[:,geometry.tip_index].copy()
        transforms[:,:3,3]+=transforms[:,:3,:3]@np.array(self._cfg['workcell']['tcp_offset_m'])
        return transforms

    def candidates(self):
        t=self._transport
        settings=self._cfg['free_inspection']
        grid=np.array([(*q,0.) for q in product(*settings['joint_samples_rad'])])
        zero=self.matrices(grid)
        quarter=grid.copy();quarter[:,5]=math.pi/2
        ninety=self.matrices(quarter)
        a,b=zero[:,2,1],ninety[:,2,1]
        root=np.arctan2(-a,b)
        variants=np.stack([root,root+math.pi,root-math.pi],axis=1)
        limits=np.array(self._cfg['safety']['joint_limits_rad'])
        score=np.where((variants>=limits[5,0])&(variants<=limits[5,1]),abs(variants),np.inf)
        grid[:,5]=variants[np.arange(len(grid)),np.argmin(score,axis=1)]
        matrices=self.matrices(grid)
        relative=np.array(t.apple_in_tcp.position_m)
        centers=matrices[:,:3,3]+matrices[:,:3,:3]@relative
        world=np.array(self._cfg['safety']['workspace_aabb_m'])
        valid=((grid>=limits[:,0])&(grid<=limits[:,1])).all(axis=1)
        valid&=((matrices[:,:3,3]>=world[:,0])&(matrices[:,:3,3]<=world[:,1])).all(axis=1)
        valid&=centers[:,2]>=settings['min_apple_z_m']
        valid&=abs(matrices[:,2,1])<=settings['jaw_vertical_tolerance']
        camera=np.array(self._cfg['workcell']['T_base_camera'])
        points=(centers-camera[:3,3])@camera[:3,:3]
        fx,fy,cx,cy=self._cfg['simulation']['intrinsics']
        h,w=self._cfg['simulation']['image_size']
        radius=self._cfg['simulation']['apples'][t.active]['radius_m']
        projected=fx*radius/np.maximum(points[:,2],1e-6)
        pixels=points[:,:2]/np.maximum(points[:,2,None],1e-6)*[fx,fy]+[cx,cy]
        valid&=points[:,2]>radius
        valid&=projected>=settings['min_projected_radius_px']
        valid&=((pixels-projected[:,None]>0)&(pixels+projected[:,None]<[w,h])).all(axis=1)
        current=pose_to_matrix(t.tool_pose())
        self.relative_rotation=current[:3,:3].T@t.data.xmat[t.ids[t.active]].reshape(3,3)
        rotations=matrices[:,:3,:3]@self.relative_rotation
        directions=np.einsum('nji,nj->ni',rotations,camera[:3,3]-centers)
        directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-9)
        indices=np.flatnonzero(valid)
        if not len(indices):
            raise RuntimeError('No joint candidate fits camera/workspace/jaw constraints')
        # Keep diverse surface viewing directions, without scoring synthetic colors.
        seed=indices[np.argmax(directions[indices,2])]
        chosen=[seed]
        distance=np.linalg.norm(directions[indices]-directions[seed],axis=1)
        for _ in range(min(settings['max_candidates'],len(indices))-1):
            index=indices[np.argmax(distance)]
            if distance.max()<1e-6:
                break
            chosen.append(index)
            distance=np.minimum(distance,np.linalg.norm(directions[indices]-directions[index],axis=1))
        return grid[chosen],dict(raw_candidates=len(grid),geometric_candidates=int(valid.sum()),
                                 direction_candidates=len(chosen))

    def visibility(self,q,probe,surface):
        t=self._transport
        for name,value in zip(t.arm,q):
            probe.qpos[t.joints.qadr[name]]=value
        t.mj.mj_kinematics(t.model,probe)
        rotation=probe.xmat[t.native_tip].reshape(3,3)
        tcp=probe.xpos[t.native_tip]+rotation@np.array(self._cfg['workcell']['tcp_offset_m'])
        center=tcp+rotation@np.array(t.apple_in_tcp.position_m)
        apple_rotation=rotation@self.relative_rotation
        body=t.ids[t.active]
        address=t.model.jnt_qposadr[t.model.body_jntadr[body]]
        probe.qpos[address:address+3]=center
        quat=np.empty(4)
        t.mj.mju_mat2Quat(quat,apple_rotation.reshape(9))
        probe.qpos[address+3:address+7]=quat
        t.mj.mj_forward(t.model,probe)
        params=params_from_config(self._cfg)
        radius=t.cfg['simulation']['apples'][t.active]['radius_m']
        normals=surface@apple_rotation.T
        points=center+normals*radius
        origin=np.array(t.cfg['workcell']['T_base_camera'])[:3,3]
        rays=points-origin
        rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        visible=(-normals*rays).sum(axis=1)>=params.cos_theta_min
        gid=np.empty(1,np.int32)
        for i in np.flatnonzero(visible):
            distance=t.mj.mj_ray(t.model,probe,origin,rays[i],None,1,-1,gid)
            if distance<0 or t.model.geom_bodyid[gid[0]]!=body:
                visible[i]=False
        return visible

    def safe_path(self,start,goal):
        t=self._transport
        points=self._planner.plan_joint_goal(goal,start).points_rad
        if not native_route_clear(t,points):
            raise ValueError('Robot intersects another object along inspection path')
        transforms=self.matrices(points)
        centers=transforms[:,:3,3]+transforms[:,:3,:3]@np.array(t.apple_in_tcp.position_m)
        radius=t.cfg['simulation']['apples'][t.active]['radius_m']
        margin=self._cfg['free_inspection']['min_clearance_m']
        if (centers[:,2]<radius+margin).any():
            raise ValueError('Held apple approaches the floor')
        for index,other in enumerate(t.positions()):
            if index!=t.active and (np.linalg.norm(centers-other,axis=1)<radius+t.cfg['simulation']['apples'][index]['radius_m']+margin).any():
                raise ValueError('Held apple approaches another apple')
        for box in t.cfg['workcell']['boxes'].values():
            lower=np.array(box['origin_m'])-.002
            upper=np.array(box['origin_m'])+[*(np.array(box['grid'])*box['cell_pitch_m']),box['wall_height_m']]
            distance=np.maximum(np.maximum(lower-centers,centers-upper),0)
            if (np.linalg.norm(distance,axis=1)<radius+margin).any():
                raise ValueError('Held apple approaches a bin')
        return points

    def prepare_inspection(self,apple):
        t=self._transport
        settings=t.settings
        timeout=self._cfg['free_inspection']['timeout_s']
        target=Pose3D(tuple(settings['inspection_center_m']),
            _multiply(_axis_angle((0,1,0),settings['tilt_rad']),t.apple_in_tcp.orientation_xyzw))
        central=compose(target,inverse(t.apple_in_tcp))
        with stage_timer(self,'free_center_transfer'):
            self._high_transfer(central.position_m[:2],speed_scale=1.,timeout_s=timeout)
            self._move_poses((central,),speed_scale=1.,deadline=self._deadline(timeout))
        self.entry=t.joint_state()
        with stage_timer(self,'free_view_search'):
            joints,report=self.candidates()
            surface=fibonacci_sphere(params_from_config(self._cfg).surface_bins)
            probe=t.mj.MjData(t.model);probe.qpos[:]=t.data.qpos
            validator=self._planner._segment_validator
            choices=[]
            rejected=0
            for q in joints:
                if not validator.is_segment_safe(q,q):
                    rejected+=1
                    continue
                mask=self.visibility(q,probe,surface)
                if mask.any():
                    choices.append((q,mask))
            seen=np.zeros(len(surface),bool)
            current=self.entry
            self.selected=[]
            gains=[]
            path_rejections=0
            for _ in range(self._cfg['free_inspection']['max_views']):
                ranked=sorted(range(len(choices)),key=lambda i:(-np.count_nonzero(choices[i][1]&~seen),
                    float(np.max(abs(choices[i][0]-current)))))
                found=False
                for i in ranked:
                    q,mask=choices[i]
                    gain=float(np.mean(mask&~seen))
                    if gain<self._cfg['free_inspection']['min_gain']:
                        break
                    try:
                        self.safe_path(current,q)
                    except (ValueError,RuntimeError):
                        path_rejections+=1
                        continue
                    self.selected.append(q)
                    seen|=mask
                    gains.append(dict(gain=gain,predicted_coverage=float(seen.mean())))
                    current=q
                    choices.pop(i)
                    found=True
                    break
                if not found or seen.mean()>=self._cfg['free_inspection']['target_coverage']:
                    break
            if not self.selected:
                raise RuntimeError('No safe occlusion-aware inspection path found')
            orientations=self.matrices(self.selected)
            actual_angles=[float(math.atan2(m[0,2],m[2,2])-settings['tilt_rad']) for m in orientations]
            self._cfg['inspection']['views']['rotation_angles_rad']=actual_angles
            report.update(apple=t.active+1,views=len(self.selected),predicted_coverage=float(seen.mean()),
                endpoint_rejections=rejected,path_rejections=path_rejections,gains=gains,
                goals_joint_rad=[q.tolist() for q in self.selected],
                tool_positions_m=orientations[:,:3,3].tolist(),
                joint_ranges_rad=np.ptp(np.array([self.entry,*self.selected]),axis=0).tolist())
            self.search_reports.append(report)
            print(f"자유 자세 탐색: {report['geometric_candidates']}개 → {len(self.selected)}뷰, 예상 관측 {seen.mean():.3f}",flush=True)

    def move_checked(self,goal,deadline):
        t=self._transport
        with stage_timer(self,'free_joint_motion'):
            path=self.safe_path(t.joint_state(),goal)
            for q in path:
                if not t.move_joint_goal(q,speed_scale=1.,timeout_s=self._remaining(deadline)):
                    raise TimeoutError('Free inspection joint goal did not arrive')

    def inspect_views(self,profile,inspection_plan,*,effort,timeout_s,view_check=None):
        del inspection_plan,effort
        t=self._transport
        deadline=self._deadline(timeout_s)
        captures=[]
        width_watch=self._grip_width_watch()
        visited=[self.entry]
        t.rotating=True
        try:
            for index,q in enumerate(self.selected):
                self.move_checked(q,deadline)
                visited.append(t.joint_state())
                capture=t.capture_in_hand(profile.apple_id,index,timeout_s=self._remaining(deadline))
                if capture is None:
                    return InspectionPhaseResult(False,len(captures),tuple(captures))
                if width_watch is not None and width_watch():
                    captures.append(capture)
                    return InspectionPhaseResult(False,len(captures),tuple(captures),slip_view_index=index)
                if view_check is not None:
                    verdict=view_check(index,capture)
                    if verdict==ViewVerdict.UNVERIFIED:
                        capture=t.capture_in_hand(profile.apple_id,index,timeout_s=self._remaining(deadline))
                        if width_watch is not None and width_watch():
                            if capture is not None:
                                captures.append(capture)
                            return InspectionPhaseResult(False,len(captures),tuple(captures),slip_view_index=index)
                        verdict=view_check(index,capture) if capture is not None else ViewVerdict.SLIP
                    if verdict!=ViewVerdict.OK:
                        if capture is not None:
                            captures.append(capture)
                        return InspectionPhaseResult(False,len(captures),tuple(captures),slip_view_index=index)
                captures.append(capture)
            # Prefer a checked direct return; retain the checked reverse route
            # when that shortcut would collide with the floor or scene objects.
            try:
                self.safe_path(t.joint_state(),self.entry)
                return_goals=[self.entry]
            except (ValueError,RuntimeError):
                return_goals=list(reversed(visited[:-1]))
            for previous in return_goals:
                self.move_checked(previous,deadline)
            return InspectionPhaseResult(True,len(captures),tuple(captures))
        finally:
            t.rotating=False
