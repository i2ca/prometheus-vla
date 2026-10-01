"""Reduced quasi-static liquid model. Not CFD, thermal, or chemical extraction.

Reservoir is an assumed cylinder fitted to catalog capacity, not measured inside
geometry. Flow is a bounded Torricelli-like jet; gravity determines free surface,
jet interception, receiving capacity and spills. Every ml remains accounted for.
"""
from dataclasses import dataclass, field
import numpy as np


def cylinder_points(radius, bottom, top, nr=12, na=48, nz=48):
    r=radius*np.sqrt((np.arange(nr)+.5)/nr)
    a=2*np.pi*(np.arange(na)+.5)/na
    z=bottom+(top-bottom)*(np.arange(nz)+.5)/nz
    rr,aa,zz=np.meshgrid(r,a,z,indexing='ij')
    return np.stack([rr*np.cos(aa),rr*np.sin(aa),zz],axis=-1).reshape(-1,3)


def jet_intersection(origin, velocity, center, rotation):
    """First forward downward crossing of the receiving aperture plane."""
    n=rotation[:,2];gravity=np.array([0.,0.,-9.81])
    aa=.5*float(n@gravity);bb=float(n@velocity);cc=float(n@(origin-center))
    if n[2]<.5 or cc<=0:return None
    roots=np.roots([aa,bb,cc]) if abs(aa)>1e-12 else [-cc/bb]
    times=[float(t.real) for t in roots if abs(t.imag)<1e-9 and t.real>0 and bb+2*aa*t.real<0]
    if not times:return None
    t=min(times);p=origin+velocity*t+.5*gravity*t*t
    return {'time_s':t,'point':p,'radial_m':float(np.linalg.norm((rotation.T@(p-center))[:2]))}


@dataclass
class KettleWater:
    initial_ml: float=600.
    capacity_ml: float=1800.
    coffee_g: float=0.
    max_flow_ml_s: float=30.
    filter_capacity_ml: float=85.
    source_ml: float=field(init=False)
    filter_ml: float=0.
    cloth_retained_ml: float=0.
    coffee_retained_ml: float=0.
    receiver_ml: float=0.
    spilled_ml: float=0.
    discharged_ml: float=0.
    captured_ml: float=0.

    def __post_init__(self):
        if not np.isfinite(self.initial_ml) or not 0<=self.initial_ml<=self.capacity_ml:
            raise ValueError('Initial water must fit in kettle')
        if self.coffee_g<0 or self.max_flow_ml_s<=0:raise ValueError('Invalid coffee or flow setting')
        self.source_ml=float(self.initial_ml)
        self.bottom=.025;self.top=.210
        self.radius=np.sqrt(self.capacity_ml*1e-6/(np.pi*(self.top-self.bottom)))
        self.points=cylinder_points(self.radius,self.bottom,self.top)
        self.spout_local=np.array([-.091012658,.00010992,.209200575])
        self.cup_radius=.037;self.cup_bottom=.006;self.cup_top=.092
        self.cup_points=cylinder_points(self.cup_radius,self.cup_bottom,self.cup_top)
        self.cup_capacity_ml=np.pi*self.cup_radius**2*(self.cup_top-self.cup_bottom)*1e6

    def retained_capacity(self, rotation):
        axis=rotation[2];level=float(axis@self.spout_local)
        return float(self.capacity_ml*np.mean(self.points@axis<=level))

    def step(self, dt, kettle_pos, kettle_rot, filter_pos, filter_rot, cup_pos, cup_rot):
        if not np.isfinite(dt) or dt<=0:raise ValueError('dt must be positive and finite')
        kp,kr,fp,fr,cp,cr=map(np.asarray,[kettle_pos,kettle_rot,filter_pos,filter_rot,cup_pos,cup_rot])
        if not all(np.all(np.isfinite(x)) for x in [kp,kr,fp,fr,cp,cr]):raise ValueError('Non-finite pose')
        axis=kr[2];heights=self.points@axis;fraction=self.source_ml/self.capacity_ml
        surface=float(np.quantile(heights,np.clip(fraction,0,1)))
        lip=kp+kr@self.spout_local;head=max(0.,surface-float(axis@self.spout_local))
        speed=.65*np.sqrt(2*9.81*head)
        capacity=self.capacity_ml*np.mean(heights<=float(axis@self.spout_local))
        flow=min(self.max_flow_ml_s,np.pi*.006**2*speed*1e6,max(0.,self.source_ml-capacity)/.5)
        released=min(self.source_ml,flow*dt);self.source_ml-=released;self.discharged_ml+=released
        velocity=kr@np.array([-speed,0.,0.]);mouth=fp+fr@np.array([0.,0.,.205]);hit=jet_intersection(lip,velocity,mouth,fr)
        captured=hit is not None and hit['radial_m']+.006<=.035
        if captured:self.filter_ml+=released;self.captured_ml+=released
        else:self.spilled_ml+=released
        cloth=min(self.filter_ml,max(0.,5.-self.cloth_retained_ml));self.cloth_retained_ml+=cloth;self.filter_ml-=cloth
        coffee=min(self.filter_ml,max(0.,2*self.coffee_g-self.coffee_retained_ml));self.coffee_retained_ml+=coffee;self.filter_ml-=coffee
        overflow=max(0.,self.filter_ml-self.filter_capacity_ml);self.filter_ml-=overflow;self.spilled_ml+=overflow
        drain=min(self.filter_ml,.4*np.sqrt(self.filter_ml)*dt);self.filter_ml-=drain
        drain_pos=fp+fr@np.array([0.,0.,.135]);cup_mouth=cp+cr@np.array([0.,0.,self.cup_top]);drip=jet_intersection(drain_pos,np.zeros(3),cup_mouth,cr)
        receiver_aligned=drip is not None and drip['radial_m']+.003<self.cup_radius
        if receiver_aligned:self.receiver_ml+=drain
        else:self.spilled_ml+=drain
        cup_axis=cr[2];horizontal=np.linalg.norm(cup_axis[:2]);rim_level=self.cup_top*cup_axis[2]-self.cup_radius*horizontal
        cup_capacity=self.cup_capacity_ml*np.mean(self.cup_points@cup_axis<=rim_level)
        excess=max(0.,self.receiver_ml-cup_capacity);self.receiver_ml-=excess;self.spilled_ml+=excess
        total=self.source_ml+self.filter_ml+self.cloth_retained_ml+self.coffee_retained_ml+self.receiver_ml+self.spilled_ml
        if abs(total-self.initial_ml)>1e-7:raise RuntimeError('Water mass balance violated')
        return {'source_ml':self.source_ml,'filter_ml':self.filter_ml,'cloth_retained_ml':self.cloth_retained_ml,'coffee_retained_ml':self.coffee_retained_ml,'receiver_ml':self.receiver_ml,'spilled_ml':self.spilled_ml,'discharged_ml':self.discharged_ml,'captured_ml':self.captured_ml,'flow_ml_s':released/dt,'mass_balance_error_ml':total-self.initial_ml,'spout_m':lip.tolist(),'jet_velocity_m_s':velocity.tolist(),'jet_hits_filter':bool(captured),'jet_hit_m':None if hit is None else hit['point'].tolist()}
