"""Conservative reduced ground-coffee ledger, NOT a granular-particle simulation.

Geometry follows the measured 30x25mm spoon bowl, assumed 5mm inner depth.
Bulk density, transfer efficiency, angle of repose and flow coefficients remain
explicit unmeasured parameters. Mass cannot appear from a timer or command.
Pickup requires the bowl to sweep inside the pot's current powder bed. Release
requires physical tilt; the falling dose is assigned by ballistic intersection.
"""
from dataclasses import dataclass,asdict,field
import numpy as np
from kettle_liquid import jet_intersection,cylinder_points

@dataclass
class CoffeeGrounds:
    initial_g: float=100.0
    bulk_density_g_ml: float=.35
    pot_inner_radius_m: float=.052
    pot_floor_m: float=.010
    bowl_a_m: float=.014
    bowl_b_m: float=.0115
    bowl_depth_m: float=.005
    repose_deg: float=28.0
    pot_g: float=field(init=False)
    spoon_g: float=0.0
    filter_g: float=0.0
    spilled_g: float=0.0
    pickup_efficiency: float=.55
    last_bowl_world: object=None
    last_pot_local: object=None

    def __post_init__(self):
        if self.initial_g<0 or self.bulk_density_g_ml<=0:raise ValueError('invalid inventory/density')
        self.pot_g=float(self.initial_g)
        self.pot_rim_m=.064
        self._pot_points=cylinder_points(self.pot_inner_radius_m,self.pot_floor_m,self.pot_rim_m,nr=8,na=24,nz=24)
        if self.bed_height_m>self.pot_rim_m:raise ValueError('initial powder exceeds pot capacity')

    @property
    def capacity_ml(self):
        return float(np.pi*self.bowl_a_m*self.bowl_b_m*self.bowl_depth_m/2*1e6)

    @property
    def bed_height_m(self):
        return self.pot_floor_m+self.pot_g/self.bulk_density_g_ml/1e6/(np.pi*self.pot_inner_radius_m**2)

    def snapshot(self):
        result=asdict(self)
        for key in ['last_bowl_world','last_pot_local']:
            if result[key] is not None:result[key]=np.asarray(result[key]).tolist()
        return result

    def step(self,dt,spoon_pos,spoon_R,pot_pos,pot_R,filter_pos,filter_R):
        if dt<=0:raise ValueError('positive timestep required')
        spoon_pos=np.asarray(spoon_pos);spoon_R=np.asarray(spoon_R);pot_pos=np.asarray(pot_pos);pot_R=np.asarray(pot_R);filter_pos=np.asarray(filter_pos);filter_R=np.asarray(filter_R)
        bowl=spoon_pos+spoon_R@np.array([-.045,0,.003]);local=pot_R.T@(bowl-pot_pos)
        prev=bowl if self.last_bowl_world is None else np.asarray(self.last_bowl_world)
        previous_local=local if self.last_pot_local is None else np.asarray(self.last_pot_local)
        velocity=(bowl-prev)/dt;travel=float(np.linalg.norm(local-previous_local));self.last_bowl_world=bowl.copy();self.last_pot_local=local.copy()
        upright=float(np.clip(spoon_R[2,2],-1,1));tilt=float(np.rad2deg(np.arccos(upright)));capacity_g=self.capacity_ml*self.bulk_density_g_ml
        # Tilt reduces the possible retained volume; no heaped dose is assumed.
        effective_tilt=max(0.,tilt-self.repose_deg)
        # Friction supports powder up to the assumed repose angle. Beyond it,
        # the shallow bowl loses usable depth; a paraboloid's volume scales
        # with depth squared. This remains an uncalibrated bulk approximation.
        depth_fraction=max(0.,1-np.tan(np.deg2rad(min(effective_tilt,89.9)))*self.bowl_a_m/self.bowl_depth_m)
        safe_fraction=depth_fraction**2
        retained_capacity=capacity_g*safe_fraction
        picked=0.;released=0.;captured=0.;returned=0.;hit=None
        inside=(np.linalg.norm(local[:2])+self.bowl_a_m<self.pot_inner_radius_m and self.pot_floor_m+self.bowl_depth_m/2<local[2]<self.bed_height_m and upright>.5)
        if inside and travel>1e-7:
            submerged=min(self.bowl_depth_m,max(0,self.bed_height_m-local[2]))
            swept_ml=2*self.bowl_b_m*submerged*travel*1e6
            picked=min(self.pot_g,max(0,retained_capacity-self.spoon_g),swept_ml*self.bulk_density_g_ml*self.pickup_efficiency)
            self.pot_g-=picked;self.spoon_g+=picked
        excess=max(0,self.spoon_g-retained_capacity)
        if excess>0:
            released=min(excess,dt*max(.01,capacity_g)*3)
            self.spoon_g-=released
            # Intersection with the physical filter-mouth plane, then pot bed.
            mouth=filter_pos+filter_R@np.array([0,0,.205]);intersection=jet_intersection(bowl,velocity,mouth,filter_R)
            if intersection is not None and intersection['radial_m']<.030:
                captured=released;self.filter_g+=captured;hit=intersection['point'].tolist()
            if not captured:
                bed=pot_pos+pot_R@np.array([0,0,self.bed_height_m]);intersection=jet_intersection(bowl,velocity,bed,pot_R)
                inside_bed=np.linalg.norm(local[:2])<self.pot_inner_radius_m-.005 and self.pot_floor_m<local[2]<=self.bed_height_m and pot_R[2,2]>.8
                if inside_bed or (intersection is not None and intersection['radial_m']<self.pot_inner_radius_m-.005):
                    returned=released;self.pot_g+=returned
                else:self.spilled_g+=released
        # A falling/tipping pot cannot keep an invariant stock glued inside it.
        axis=pot_R[2];rim_level=self.pot_rim_m*axis[2]-self.pot_inner_radius_m*np.linalg.norm(axis[:2])
        volume_ml=np.pi*self.pot_inner_radius_m**2*(self.pot_rim_m-self.pot_floor_m)*1e6
        pot_capacity_g=volume_ml*self.bulk_density_g_ml*np.mean(self._pot_points@axis<=rim_level)
        pot_spill=min(max(0,self.pot_g-pot_capacity_g),self.initial_g*dt*2)
        self.pot_g-=pot_spill;self.spilled_g+=pot_spill
        if filter_R[2,2]<.5:
            filter_spill=min(self.filter_g,self.initial_g*dt)
            self.filter_g-=filter_spill;self.spilled_g+=filter_spill
        total=self.pot_g+self.spoon_g+self.filter_g+self.spilled_g
        if abs(total-self.initial_g)>1e-7 or min(self.pot_g,self.spoon_g,self.filter_g,self.spilled_g)<-1e-9:raise RuntimeError('ground coffee mass conservation failure')
        return {'pot_g':self.pot_g,'spoon_g':self.spoon_g,'filter_g':self.filter_g,'spilled_g':self.spilled_g,'picked_g':picked,'released_g':released,'captured_g':captured,'returned_g':returned,'bowl_world_m':bowl.tolist(),'tilt_deg':tilt,'jet_hit_m':hit,'inside_powder':bool(inside),'capacity_g':capacity_g,'bed_height_m':self.bed_height_m}
