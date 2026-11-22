"""Render genuine model candidates as non-physical 3D wire grippers and axes."""
import numpy as np
from scipy.spatial.transform import Rotation


def draw_grasps(stage,candidates,selected=None,root='/World/GraspNetPredictions'):
    from pxr import UsdGeom,Gf,UsdShade,Sdf
    if stage.GetPrimAtPath(root):stage.RemovePrim(root)
    UsdGeom.Xform.Define(stage,root)
    for rank,candidate in enumerate(candidates):
        r=Rotation.from_quat(candidate['quat_xyzw']).as_matrix()
        tcp=np.asarray(candidate['xyz_m']);half=candidate['opening_m']/2
        tip=candidate['physical_tip_from_tcp_m']
        # Wire geometry is attached to the predicted world pose, not the robot.
        # No Collision/RigidBody/Joint API is applied to any overlay.
        local=np.array([[-half,0,tip-.055],[-half,0,tip],[half,0,tip-.055],[half,0,tip],
                        [-half,0,tip-.055],[half,0,tip-.055],[0,0,tip-.055],[0,0,tip-.09]])
        world=local@r.T+tcp
        color=(.1,1.,.1) if rank==selected else (1.-rank/max(1,len(candidates)),.35,rank/max(1,len(candidates)))
        curve=UsdGeom.BasisCurves.Define(stage,f'{root}/Candidate_{rank:02d}/Gripper')
        curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([2]*4)
        curve.CreatePointsAttr([Gf.Vec3f(*p) for p in world]);curve.CreateWidthsAttr([.004 if rank==selected else .0008]*8)
        curve.CreateDisplayColorAttr([color])
        if rank==selected:
            material=UsdShade.Material.Define(stage,f'{root}/SelectedMaterial')
            shader=UsdShade.Shader.Define(stage,f'{root}/SelectedMaterial/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.,.25,0.))
            shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.05,1.,.05))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)
        curve.GetPrim().SetCustomDataByKey('model_score',candidate['score'])
        curve.GetPrim().SetCustomDataByKey('source_prediction_index',candidate['index'])
        if rank==selected:
            for i,name in enumerate(['X_jaw','Y','Z_approach']):
                points=np.array([tcp,tcp+.045*r[:,i]])
                axis=UsdGeom.BasisCurves.Define(stage,f'{root}/SelectedAxes/{name}')
                axis.CreateTypeAttr('linear');axis.CreateCurveVertexCountsAttr([2]);axis.CreatePointsAttr([Gf.Vec3f(*p) for p in points]);axis.CreateWidthsAttr([.002]*2)
                axis.CreateDisplayColorAttr([tuple(np.eye(3)[i])])
            # Show the approach *before* contact, rather than hiding the whole
            # blue axis inside the object. Its endpoint is the predicted TCP.
            arrow=np.array([tcp-.10*r[:,2],tcp,
                            tcp-.012*r[:,2]+.007*r[:,0],tcp,
                            tcp-.012*r[:,2]-.007*r[:,0],tcp])
            axis=UsdGeom.BasisCurves.Define(stage,f'{root}/SelectedAxes/ApproachArrow')
            axis.CreateTypeAttr('linear');axis.CreateCurveVertexCountsAttr([2,2,2]);axis.CreatePointsAttr([Gf.Vec3f(*p) for p in arrow]);axis.CreateWidthsAttr([.0035]*6)
            axis.CreateDisplayColorAttr([(.05,.25,1.)])
    return root


def information_window(candidate,counts):
    """Keep the returned UI window alive in the caller while the GUI is open."""
    import omni.ui as ui
    window=ui.Window('GraspNet 6-DoF predictions',width=440,height=260)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(f'Real model output: {counts} candidates | Green = selected',word_wrap=True)
            if candidate is not None:
                ui.Label('World TCP xyz (m): '+str([round(v,4) for v in candidate['xyz_m']]),word_wrap=True)
                ui.Label('Quaternion XYZW: '+str([round(v,4) for v in candidate['quat_xyzw']]),word_wrap=True)
                ui.Label(f'Score {candidate["score"]:.4f} | opening {candidate["opening_m"]*1000:.1f} mm')
                ui.Label(f'Approach tilt from down: {candidate["tilt_from_world_down_deg"]:.1f} deg')
            ui.Label('RGB-D workspace crop; no oracle object pose. Red X = jaw, blue Z = approach.',word_wrap=True)
            ui.Label('Markers remain at the observed prediction pose after the object moves.',word_wrap=True)
    return window
