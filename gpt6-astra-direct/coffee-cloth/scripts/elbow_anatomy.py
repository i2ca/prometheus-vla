"""Geometric elbow flexion, independent of the robot encoder's zero convention.

Use shoulder-pitch, elbow and wrist-pitch joint anchors, projected into the elbow
hinge plane. Positive means flexion; negative means extension past the projected
straight-arm configuration. This is an ergonomic proxy for this robot, not a
clinical human model. The task band of 5..145 degrees leaves a small extension
margin. Manufacturer mechanical limits remain active as well.
"""
import numpy as np


class ElbowAnatomy:
    def __init__(self, model, minimum_deg=5., maximum_deg=145.):
        self.minimum_deg=minimum_deg;self.maximum_deg=maximum_deg
        self.joints=[tuple(model.joint(side+'_'+name+'_joint').id
                           for name in ['shoulder_pitch','elbow','wrist_pitch'])
                     for side in ['left','right']]

    def angles(self, data):
        values=[]
        for shoulder,elbow,wrist in self.joints:
            axis=data.xaxis[elbow]
            upper=data.xanchor[elbow]-data.xanchor[shoulder]
            fore=data.xanchor[wrist]-data.xanchor[elbow]
            upper=upper-axis*(axis@upper);fore=fore-axis*(axis@fore)
            values.append(-np.rad2deg(np.arctan2(axis@np.cross(upper,fore),upper@fore)))
        return np.asarray(values)

    def penalty(self, data):
        angles=self.angles(data)
        return np.deg2rad(np.r_[np.minimum(angles-(self.minimum_deg+1.),0.),
                               np.maximum(angles-(self.maximum_deg-1.),0.)])*50

    def valid(self, data, tolerance_deg=0.):
        values=self.angles(data)
        return bool(np.all(values>=self.minimum_deg-tolerance_deg)
                    and np.all(values<=self.maximum_deg+tolerance_deg))

    def bound_search(self, model, names, bounds):
        """Restore allowed flexion; use actual geometry for the final gate."""
        for i,name in enumerate(names):
            if name.endswith('_elbow_joint'):
                bounds[i]=model.jnt_range[model.joint(name).id]
                bounds[i,1]=min(bounds[i,1],np.deg2rad(75.))
        return bounds
