"""Conditional 2D Flow Matching with Sinkhorn optimal-transport coupling.
Compare independent random pairing against an OT coupling. OT tries to pair
noise and target points with shorter Euclidean transport paths.
"""
import argparse, math, os, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

RED, BLUE, GRAY, PURPLE = "#EF4444", "#2563EB", "#94A3B8", "#7C3AED"

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def targets(n, condition, device):
    if condition == 0:
        a=torch.rand(n,device=device)*math.pi; side=torch.randint(0,2,(n,),device=device)
        x=torch.where(side[:,None]==0, torch.stack([torch.cos(a),torch.sin(a)],1), torch.stack([1-torch.cos(a),.5-torch.sin(a)],1))
        x[:,0]-=.5; x[:,1]-=.25; return 1.65*(x+.07*torch.randn_like(x))
    a=torch.rand(n,device=device)*2*math.pi; r=1.35+.08*torch.randn(n,device=device)
    return torch.stack([r*torch.cos(a),r*torch.sin(a)],1)

def sinkhorn_pairs(source, target, reg=.12, iterations=60):
    """Return a soft OT plan with uniform marginals, then sample pairs."""
    cost=torch.cdist(source,target).pow(2)
    log_k=-cost/reg
    n,m=cost.shape; log_a=torch.full((n,),-math.log(n),device=source.device); log_b=torch.full((m,),-math.log(m),device=source.device)
    u=torch.zeros_like(log_a); v=torch.zeros_like(log_b)
    for _ in range(iterations):
        u=log_a-torch.logsumexp(log_k+v[None,:],1)
        v=log_b-torch.logsumexp(log_k+u[:,None],0)
    plan=torch.exp(log_k+u[:,None]+v[None,:])
    # one target for each source; row-normalized sampling is differentiable-free
    idx=torch.multinomial(plan/(plan.sum(1,keepdim=True)+1e-12),1).squeeze(1)
    return target[idx], cost.detach(), plan.detach()

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.emb=nn.Embedding(2,16); self.net=nn.Sequential(nn.Linear(19,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,2))
    def forward(self,x,t,c): return self.net(torch.cat([x,t,self.emb(c)],1))

@torch.no_grad()
def ode(model,z,c,steps,device):
    frames=[z.cpu()]; dt=1/steps
    for i in range(steps): z=z+dt*model(z,torch.full((len(z),1),i/steps,device=device),torch.full((len(z),),c,device=device,dtype=torch.long)); frames.append(z.cpu())
    return frames

def axis(ax,title): ax.set(title=title,xlim=(-3,3),ylim=(-3,3),aspect='equal'); ax.grid(alpha=.18)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=2500); ap.add_argument('--batch-size',type=int,default=256); ap.add_argument('--particles',type=int,default=700); ap.add_argument('--ode-steps',type=int,default=80); ap.add_argument('--reg',type=float,default=.12); ap.add_argument('--output-dir',default='outputs_ot'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    seed_all(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); os.makedirs(args.output_dir,exist_ok=True)
    model=Net().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3); losses=[]; path_lengths=[]
    for step in range(args.steps):
        c=torch.randint(0,2,(args.batch_size,),device=device); source=torch.randn(args.batch_size,2,device=device)*1.15; raw=targets(args.batch_size, int(c[0]), device)
        # Use separate same-condition batches so each OT plan has one target distribution.
        data=torch.empty_like(source)
        for label in (0,1):
            mask=c==label
            if int(mask.sum()):
                s=source[mask]; d=targets(len(s),label,device); data[mask],cost,_=sinkhorn_pairs(s,d,args.reg,30); path_lengths.append(float(cost.mean().sqrt().cpu()))
        t=torch.rand(args.batch_size,1,device=device); xt=(1-t)*source+t*data; loss=((model(xt,t,c)-(data-source))**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        if step==0 or (step+1)%max(1,args.steps//10)==0: print('step %d/%d loss %.4f'%(step+1,args.steps,np.mean(losses[-100:])))
    torch.save(model.state_dict(),os.path.join(args.output_dir,'ot_checkpoint.pt'))
    initial=torch.randn(args.particles,2)*1.15; traj=[ode(model,initial,c,args.ode_steps,device) for c in (0,1)]; data=[targets(args.particles,c,device).cpu() for c in (0,1)]
    fig,ax=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for r,(name,col) in enumerate([('OT-coupled moons',RED),('OT-coupled ring',BLUE)]):
        ax[r,0].scatter(initial[:,0],initial[:,1],s=6,c=GRAY,alpha=.5); axis(ax[r,0],name+' source')
        ax[r,1].scatter(data[r][:,0],data[r][:,1],s=6,c=col,alpha=.5); axis(ax[r,1],name+' target')
        ax[r,2].scatter(traj[r][-1][:,0],traj[r][-1][:,1],s=6,c=col,alpha=.55); axis(ax[r,2],name+' generated')
        ids=np.linspace(0,args.particles-1,70).astype(int); forx=np.array([initial[ids].numpy(),traj[r][-1][ids].numpy()]); ax[r,3].plot(forx[:,:,0],forx[:,:,1],color=col,alpha=.25); ax[r,3].scatter(initial[ids,0],initial[ids,1],c=GRAY,s=12); ax[r,3].scatter(traj[r][-1][ids,0],traj[r][-1][ids,1],c=col,s=12); axis(ax[r,3],name+' paired paths')
    fig.suptitle('Sinkhorn Optimal Transport coupling for Conditional Flow Matching',fontsize=16,fontweight='bold'); fig.savefig(os.path.join(args.output_dir,'01_ot_overview.png'),dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(losses,color=PURPLE); ax.set(title='OT-CFM velocity regression loss',xlabel='step',ylabel='MSE'); ax.grid(alpha=.2); fig.savefig(os.path.join(args.output_dir,'02_ot_loss.png'),dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(path_lengths,color=PURPLE,alpha=.5); ax.set(title='OT batch transport distance (lower means straighter local pairing)',xlabel='OT mini-batch sample',ylabel='mean distance'); ax.grid(alpha=.2); fig.savefig(os.path.join(args.output_dir,'03_ot_transport_distance.png'),dpi=180); plt.close(fig)
    # Extra visual diagnostics: velocity fields at multiple times.
    times=[0,.2,.4,.6,.8,1.0]
    fig,axes=plt.subplots(2,6,figsize=(19,7),constrained_layout=True)
    for row,(name,col) in enumerate([('moons',RED),('ring',BLUE)]):
        for j,tv in enumerate(times):
            gx,gy=np.meshgrid(np.linspace(-2.7,2.7,17),np.linspace(-2.5,2.5,15)); grid=torch.tensor(np.c_[gx.ravel(),gy.ravel()],dtype=torch.float32,device=device); tt=torch.full((len(grid),1),tv,device=device); cc=torch.full((len(grid),),row,dtype=torch.long,device=device); vv=model(grid,tt,cc).detach().cpu().numpy(); sp=np.linalg.norm(vv,axis=1); idx=int(tv*(len(traj[row])-1)); axes[row,j].scatter(traj[row][idx][:,0],traj[row][idx][:,1],s=3,c=col,alpha=.12); axes[row,j].quiver(gx,gy,vv[:,0],vv[:,1],sp,cmap='viridis',scale=7); axis(axes[row,j],'%s t=%.1f'%(name,tv))
    fig.suptitle('OT-CFM: velocity fields evolve over time',fontsize=15,fontweight='bold'); fig.savefig(os.path.join(args.output_dir,'04_ot_velocity_fields_over_time.png'),dpi=180); plt.close(fig)
    # Final samples and trajectories at a glance.
    fig,axes=plt.subplots(2,4,figsize=(15,8),constrained_layout=True); ids=np.linspace(0,args.particles-1,35).astype(int)
    for row,(name,col) in enumerate([('moons',RED),('ring',BLUE)]):
        for j,idx in enumerate(np.linspace(0,len(traj[row])-1,4).astype(int)):
            axes[row,j].scatter(traj[row][idx][:,0],traj[row][idx][:,1],s=5,c=col,alpha=.55); axis(axes[row,j],'%s t=%.2f'%(name,idx/(len(traj[row])-1)))
    fig.suptitle('OT-CFM sampling trajectory snapshots',fontsize=15,fontweight='bold'); fig.savefig(os.path.join(args.output_dir,'05_ot_trajectory_snapshots.png'),dpi=180); plt.close(fig)
    # Pair-distance distribution gives an intuitive OT diagnostic.
    s=torch.randn(args.particles,2,device=device)*1.15
    for row in (0,1):
        d=targets(args.particles,row,device); _,cost,_=sinkhorn_pairs(s,d,args.reg,60); plt.figure(figsize=(7,4)); plt.hist(cost.sqrt().cpu().numpy(),bins=35,color=PURPLE,alpha=.8); plt.title('OT paired transport distances: '+('moons' if row==0 else 'ring')); plt.xlabel('Euclidean distance'); plt.ylabel('count'); plt.grid(alpha=.2); plt.savefig(os.path.join(args.output_dir,'06_ot_distance_hist_%d.png'%row),dpi=180); plt.close()
    print('device:',device); print('saved to',args.output_dir)
if __name__=='__main__': main()


