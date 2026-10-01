"""Lumped energy-balance kettle model; not CFD or a certified appliance model.

1200W at127V is from the local Electrolux EEK10 manual. Efficiency, heat loss,
body heat capacity, boiling point and rocker force/travel are unmeasured assumptions.
Power only starts after a physical rocker press while seated and filled. There is
no fixed 'wait3seconds => boiling' shortcut. Removal, tilt, low fill or boil stop power.
"""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass
class KettleHeater:
    ambient_C: float=25.
    temperature_C: float=25.
    boiling_C: float=100.
    rated_power_W: float=1200.
    efficiency: float=.85
    heat_loss_W_K: float=2.4
    body_heat_capacity_J_K: float=390.
    water_heat_capacity_J_kg_K: float=4180.
    on: bool=False
    input_energy_J: float=0.
    delivered_energy_J: float=0.
    loss_energy_J: float=0.
    press_seconds: float=0.
    press_armed: bool=True
    last_event: str='initial_cold'

    def step(self,dt,water_ml,seated,upright,lid_closed,rocker_angle_rad=0.,robot_contact_N=0.):
        if not np.isfinite(dt) or dt<=0 or water_ml<0:raise ValueError('invalid time/water')
        safe=seated and upright and lid_closed and 500<=water_ml<=1800
        pressed=rocker_angle_rad>.14 and robot_contact_N>.5
        self.press_seconds=self.press_seconds+dt if pressed else 0.
        if rocker_angle_rad<.04:self.press_armed=True
        if self.press_armed and self.press_seconds>=.04:
            self.press_armed=False
            if safe and self.temperature_C<self.boiling_C-1:
                self.on=not self.on;self.last_event='physical_press_on' if self.on else 'physical_press_off'
            else:self.last_event='press_rejected_operating_conditions'
        if self.on and not safe:self.on=False;self.last_event='power_interlock'
        power=self.rated_power_W if self.on else 0.;loss=self.heat_loss_W_K*(self.temperature_C-self.ambient_C);capacity=self.body_heat_capacity_J_K+water_ml/1000*self.water_heat_capacity_J_kg_K
        delta=(self.efficiency*power-loss)*dt/capacity
        # Cut power at the crossing time; account only for the energy actually used.
        energized_dt=dt
        if self.on and self.temperature_C+delta>=self.boiling_C:
            energized_dt=dt*(self.boiling_C-self.temperature_C)/max(delta,1e-12);self.temperature_C=self.boiling_C;self.on=False;self.last_event='automatic_boil_cutoff'
            self.input_energy_J+=power*energized_dt;self.delivered_energy_J+=self.efficiency*power*energized_dt;self.loss_energy_J+=loss*energized_dt
            remainder=dt-energized_dt;cool=self.heat_loss_W_K*(self.temperature_C-self.ambient_C)*remainder;self.temperature_C-=cool/capacity;self.loss_energy_J+=cool
        else:
            self.temperature_C+=delta;self.input_energy_J+=power*dt;self.delivered_energy_J+=self.efficiency*power*dt;self.loss_energy_J+=loss*dt
        return asdict(self)
