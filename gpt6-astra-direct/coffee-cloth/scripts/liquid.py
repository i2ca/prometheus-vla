"""Conservative quasi-static water surrogate (NOT CFD), gpt-6-astra.

Volume units: ml. Geometry: m. No slosh, viscosity solution, jet momentum,
wetting chemistry or temperature. Vertical narrow jet from lowest cup rim.
Cylinder capacity is integrated by deterministic equal-volume quadrature.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class WaterTransfer:
    initial_ml: float = 200.0
    max_flow_ml_s: float = 20.0
    drain_ml_s: float = 8.0
    filter_capacity_ml: float = 140.0
    receiver_capacity_ml: float = 350.0
    relaxation_s: float = 0.5
    source_ml: float = field(init=False)
    filter_ml: float = 0.0
    receiver_ml: float = 0.0
    spilled_ml: float = 0.0
    captured_ml: float = 0.0

    def __post_init__(self):
        self.source_ml = self.initial_ml
        self.radius, self.bottom, self.top = .037, .006, .092
        radius = self.radius * np.sqrt((np.arange(12) + .5) / 12)
        theta = 2*np.pi*(np.arange(32)+.5)/32
        height = self.bottom+(self.top-self.bottom)*(np.arange(24)+.5)/24
        rr,tt,zz = np.meshgrid(radius,theta,height,indexing='ij')
        self.points = np.stack((rr*np.cos(tt),rr*np.sin(tt),zz),axis=-1).reshape(-1,3)
        self.capacity_ml = np.pi*self.radius**2*(self.top-self.bottom)*1e6
        if not 0 <= self.initial_ml <= self.capacity_ml:
            raise ValueError('Initial water must fit in source vessel')

    def geometry(self, position, rotation):
        gravity_axis = rotation[2]
        sideways = np.linalg.norm(gravity_axis[:2])
        low_rim = np.array([0.,0.,self.top])
        if sideways > 1e-9:
            low_rim[:2] = -self.radius*gravity_axis[:2]/sideways
        level = low_rim @ gravity_axis
        capacity = self.capacity_ml*np.mean(self.points @ gravity_axis <= level)
        return position + rotation @ low_rim, capacity

    def step(self, dt, position, rotation, filter_center, receiver_position):
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError('dt must be positive and finite')
        lip, capacity = self.geometry(np.asarray(position),np.asarray(rotation))
        discharged = min(self.source_ml, max(0., self.source_ml-capacity)/self.relaxation_s*dt,
                         self.max_flow_ml_s*dt)
        self.source_ml -= discharged
        # Diagnostic simplification: vertical jet, no momentum or finite footprint.
        capture = lip[2] > filter_center[2]+.002 and np.linalg.norm(lip[:2]-filter_center[:2]) < .036
        if capture:
            self.filter_ml += discharged
            self.captured_ml += discharged
        else:
            self.spilled_ml += discharged
        overflow = max(0.,self.filter_ml-self.filter_capacity_ml)
        self.filter_ml -= overflow
        self.spilled_ml += overflow
        drain = min(self.filter_ml,self.drain_ml_s*dt)
        self.filter_ml -= drain
        receiver_aligned = np.linalg.norm(np.asarray(receiver_position)[:2]-filter_center[:2]) < .026
        if receiver_aligned:
            into_cup = min(drain,max(0.,self.receiver_capacity_ml-self.receiver_ml))
            self.receiver_ml += into_cup
            self.spilled_ml += drain-into_cup
        else:
            self.spilled_ml += drain
        total=self.source_ml+self.filter_ml+self.receiver_ml+self.spilled_ml
        if abs(total-self.initial_ml)>1e-7:
            raise RuntimeError('Water conservation failed')
        return {'source_ml':self.source_ml,'filter_ml':self.filter_ml,'receiver_ml':self.receiver_ml,
                'spilled_ml':self.spilled_ml,'captured_ml':self.captured_ml,'lip':lip.tolist(),
                'flow_ml_s':discharged/dt,'capture':bool(capture),'mass_balance_error_ml':total-self.initial_ml}
