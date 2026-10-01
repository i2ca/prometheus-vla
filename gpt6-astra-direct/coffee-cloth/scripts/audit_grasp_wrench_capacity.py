"""Static contact-wrench feasibility audit, never an object force actuator."""
import argparse,json,sys,shutil
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import linprog
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim

def skew(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def solve(model,data,object_id,hand_prefix,positions,normals,body_ids,mus,torsions,load_scale=1.):
 n=len(positions);nv=4*n
 A=np.zeros((6,nv));T=np.zeros((model.nu,nv))
 dofs=model.jnt_dofadr[model.actuator_trnid[:,0]]
 for i,(p,norm,b) in enumerate(zip(positions,normals,body_ids)):
  A[:3,4*i:4*i+3]=np.eye(3);A[3:,4*i:4*i+3]=skew(p-data.xipos[object_id]);A[3:,4*i+3]=norm
  jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(model,data,jp,jr,p,b)
  T[:,4*i:4*i+3]=jp[:,dofs].T;T[:,4*i+3]=jr[:,dofs].T@norm
 rows=[];rhs=[];cost=np.zeros(nv)
 for i,(norm,mu,twist) in enumerate(zip(normals,mus,torsions)):
  cost[4*i:4*i+3]=norm
  tangent=np.cross(norm,[0,0,1] if abs(norm[2])<.9 else [0,1,0]);tangent/=np.linalg.norm(tangent);bitangent=np.cross(norm,tangent)
  for angle in np.arange(8)*np.pi/4:
   row=np.zeros(nv);row[4*i:4*i+3]=np.cos(angle)*tangent+np.sin(angle)*bitangent-mu*np.cos(np.pi/8)*norm;rows.append(row);rhs.append(0.)
  for sign in [-1,1]:
   row=np.zeros(nv);row[4*i:4*i+3]=-twist*norm;row[4*i+3]=sign;rows.append(row);rhs.append(0.)
  row=np.zeros(nv);row[4*i:4*i+3]=-norm;rows.append(row);rhs.append(-.1)
 active=[i for i in range(model.nu) if model.joint(int(model.actuator_trnid[i,0])).name.startswith(hand_prefix)]
 bias=data.qfrc_bias[dofs]
 for i in active:
  rows.extend([T[i],-T[i]]);rhs.extend([model.actuator_ctrlrange[i,1]-bias[i],-model.actuator_ctrlrange[i,0]+bias[i]])
 gravity=-model.opt.gravity*model.body_mass[object_id]*load_scale
 fit=linprog(cost,A_ub=np.array(rows),b_ub=np.array(rhs),A_eq=A,b_eq=np.r_[gravity,np.zeros(3)],bounds=[(None,None)]*nv,method='highs')
 return fit,T,active,bias

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--object',default='chaleira');ap.add_argument('--hand',default='left');a=ap.parse_args();a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
 r=json.loads((a.source/'report.json').read_text());s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(a.source/'continuation.npz')
 for name in ['body_mass','body_ipos','body_inertia','body_iquat','bvh_aabb']:getattr(m,name)[:]=ck[name]
 mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
 obj=m.body(a.object).id;groups={}
 for i,c in enumerate(d.contact):
  bodies=[int(m.geom_bodyid[g]) for g in [c.geom1,c.geom2]]
  if c.dist>=0 or obj not in bodies:continue
  b=bodies[1-bodies.index(obj)];name=m.body(b).name
  if not name.startswith(a.hand+'_hand'):continue
  f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);w=float(f[0]);n=c.frame[:3]*(1 if bodies[0]==b else -1)
  g=groups.setdefault(b,{'weight':0.,'p':np.zeros(3),'n':np.zeros(3),'mu':1e3,'twist':1e3});g['weight']+=w;g['p']+=w*c.pos;g['n']+=w*n;g['mu']=min(g['mu'],float(c.friction[0]));g['twist']=min(g['twist'],float(c.friction[2]))
 groups={b:g for b,g in groups.items() if g['weight']>.01};bs=list(groups);ps=[g['p']/g['weight'] for g in groups.values()];ns=[g['n']/np.linalg.norm(g['n']) for g in groups.values()]
 fit,T,active,bias=solve(m,d,obj,a.hand+'_hand',ps,ns,bs,[g['mu'] for g in groups.values()],[g['twist'] for g in groups.values()])
 report={'model':'gpt-6-astra','source':str(a.source),'pass':bool(fit.success),'message':fit.message,'object_mass_kg':float(m.body_mass[obj]),'scope':'static three aggregated soft-finger contacts with inscribed friction pyramids and actual motor torque caps; necessary diagnostic approximation, not dynamic proof'}
 if fit.success:
  report['contacts']=[{'body':m.body(b).name,'position':p.tolist(),'normal':n.tolist(),'force_N':fit.x[4*i:4*i+3].tolist(),'normal_force_N':float(n@fit.x[4*i:4*i+3]),'twist_Nm':float(fit.x[4*i+3])} for i,(b,p,n) in enumerate(zip(bs,ps,ns))];tau=T@fit.x+bias;report['motor_torques_Nm']={m.joint(int(m.actuator_trnid[i,0])).name:float(tau[i]) for i in active}
 (a.out/'report.json').write_text(json.dumps(report,indent=2));print(report)
