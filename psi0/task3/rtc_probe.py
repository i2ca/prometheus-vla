import os
import sys; sys.argv=sys.argv[:1]
from pathlib import Path
import numpy as np, pandas as pd, torch, av
from PIL import Image
from psi.deploy.serve_psi0_amo import Server
from psi.utils import pad_to_len
D=Path(os.environ["PSI0_DATA"])
srv=Server(policy="psi", run_dir=Path(os.environ["PSI0_CKPT"]), ckpt_step=40000, device="cuda:0"); M=srv.model; mm=srv.maxmin
df=pd.read_parquet(D/"data/chunk-000/episode_000000.parquet"); S=np.stack(df.states.values); A=np.stack(df.action.values)
with av.open(str(D/"videos/chunk-000/egocentric/episode_000000.mp4")) as c: F=[f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
lo,hi=np.array(mm.action_min,np.float32),np.array(mm.action_max,np.float32); norm=lambda a:((a-lo)/np.maximum(hi-lo,1e-8)*2-1).astype(np.float32)
I="g1/pick_bottle_and_turn_and_pour_into_cup"
def ob(t): o=pad_to_len(S[t,:32].astype(np.float32),36,dim=0)[0]; return torch.from_numpy(mm.normalize_state_func(o)[None,None].astype(np.float32)).cuda()
def img(t): return [srv.preprocess_image({"cam0":Image.fromarray(F[t])})["cam0"]]
for t in [150,300,450,600,750]:
    g=A[t:t+30]; r={}
    with torch.inference_mode():
        torch.manual_seed(0); p=M.predict_action(observations=img(t),states=ob(t),traj2ds=None,instructions=[I],num_inference_steps=8)[0].float().cpu().numpy()
        r["sem_prefixo"]=np.abs(mm.denormalize(p)[6:,:28]-g[6:,:28]).mean()
        for nome,prev in [("prefixo_gt",norm(g)),("prefixo_zero",np.zeros((30,36),np.float32)),("prefixo_gt_atrasado15",norm(A[t-15:t+15]))]:
            torch.manual_seed(0); p=M.predict_action_with_training_rtc_flow(observations=img(t),states=ob(t),traj2ds=None,instructions=[I],num_inference_steps=8,
                prev_actions=torch.from_numpy(prev[None]).cuda(),inference_delay=6,max_delay=8)[0].float().cpu().numpy()
            r[nome]=np.abs(mm.denormalize(p)[6:,:28]-g[6:,:28]).mean()
    r["segurar"]=np.abs(g[6:,:28]-g[0,:28]).mean()
    print(t, {k:round(float(v),3) for k,v in r.items()}, flush=True)
