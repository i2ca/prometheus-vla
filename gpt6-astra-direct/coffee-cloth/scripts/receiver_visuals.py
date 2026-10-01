"""Volume-ledger overlay only. No forces, mass changes, extraction or CFD here."""
import numpy as np
import mujoco
from water_visuals import _add_cylinder,_add_capsule

def add_receiver_visuals(scene,model,data,volume_ml,grounds_g=0.,drain_ml_s=0.):
    b=model.body('copo').id;R=data.xmat[b].reshape(3,3)
    if volume_ml<=.05 or R[2,2]<=0:return None
    radius=.037;bottom=.006;top=.092  # same cavity used by KettleWater's volume ledger
    height=min(top-bottom,max(0,volume_ml)*1e-6/(np.pi*radius**2))
    color=np.array([.09,.035,.014,.96] if grounds_g>=1 else ([.55,.35,.15,.65] if grounds_g>0 else [.25,.50,.85,.55]),dtype=np.float32)
    cosine=float(R[2,2]);sine=float(np.sqrt(max(0,1-cosine*cosine)))
    # For a horizontal cut wholly between floor and rim, volume/area gives the
    # central local depth exactly; its world-space boundary is an ellipse.
    variation=radius*sine/cosine
    if variation>=min(height,top-bottom-height):return None
    z=float(data.xpos[b,2]+cosine*(bottom+height));axis=R[:2,2]
    axis=axis/max(np.linalg.norm(axis),1e-12) if np.linalg.norm(axis)>1e-12 else np.array([1.,0.])
    basis=np.array([[axis[0],-axis[1],0.],[axis[1],axis[0],0.],[0.,0.,1.]])
    center=data.xpos[b]+R[:,2]*(bottom+height)
    center[2]=z
    if scene.ngeom>=scene.maxgeom:return None
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      np.array([(radius-.0005)/cosine,radius-.0005,.00015]),
                      center,basis.ravel(),color);scene.ngeom+=1
    if drain_ml_s>.01:
        f=model.body('coador').id;start=data.xpos[f]+data.xmat[f].reshape(3,3)@np.array([0.,0.,.135]);end=start.copy();end[2]=z
        if start[2]>end[2]:_add_capsule(scene,start,end,.0005+.0006*np.sqrt(min(drain_ml_s/5,1)),color)
    return {'volume_ml':float(volume_ml),'height_m':float(height),'surface_world_z_m':z,'cup_tilt_deg':float(np.rad2deg(np.arccos(cosine))),'color':'illustrative coffee tint, no extraction model' if grounds_g>0 else 'water'}
