"""
Minimal ABM for H4: does social information reduce the variability of foraging gain,
and how does this depend on patch ephemerality?

World      : W x W torus. K resource patches, each disappears with probability 1/L per step
             (mean lifetime L steps) and reappears at a random location (constant patch number).
Agents     : N foragers that alternate between searching and feeding.
Personal   : a searching agent detects a patch within r_sense and heads to it.
             Otherwise it does a correlated random walk.
Social     : feeding agents are "callers". A searching agent with no current target hears the
             nearest caller within R_call with probability q per step (q = information
             availability). It stores the caller's location and walks to it. The patch may have
             vanished by the time it arrives (stale information).
Competition: agents on the same patch share intake: rate / n_on_patch**c  (c = 0: no competition).
Outcome    : intake per bout of `bout` steps (analogue of one foraging bout / day).
             Variability = within-agent variance and CV of bout intake; shortfall = share of
             bouts below a survival threshold.

Run:  python abm_social_info.py          (about 5 minutes)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def torus_vec(a,b,W):
    """Shortest vector from a to b on a W x W torus (broadcasts over leading dims)."""
    return(b-a+W/2)%W-W/2

def simulate(q,L,seed,W=100.0,N=20,K=8,r_patch=3.0,r_sense=5.0,R_call=40.0,
             speed=1.0,turn_sd=0.4,c=0.5,rate=1.0,T=4500,bout=200,burn=500,
             position_history=None,patch_history=None):
    # Fail early on invalid settings instead of silently producing misleading output.
    if not(0.0<=q<=1.0):
        raise ValueError("q must be between 0 and 1.")
    if L<1:
        raise ValueError("L must be >= 1.")
    if min(W,N,K,r_patch,r_sense,R_call,speed,rate,T,bout)<=0:
        raise ValueError("W, N, K, radii, speed, rate, T, and bout must be positive.")
    if burn<0 or burn>=T:
        raise ValueError("burn must be >= 0 and smaller than T.")
    if c<0 or turn_sd<0:
        raise ValueError("c and turn_sd must be non-negative.")
    if r_sense<r_patch:
        raise ValueError("r_sense should be >= r_patch for the intended detection rule.")

    rng=np.random.default_rng(seed)
    pc=rng.uniform(0,W,(K,2))# patch centres
    pos=rng.uniform(0,W,(N,2))# agent positions
    th=rng.uniform(0,2*np.pi,N)# agent headings
    target=np.full((N,2),np.nan)# stored location from a call
    idx=np.arange(N)
    cur=np.zeros(N)
    out=[]
    for t in range(T):
        # Burn-in is for equilibration only. Reset accumulated intake exactly when
        # measurement begins, so the first recorded bout cannot include burn-in.
        if t==burn:
            cur[:]=0.0

        # patch turnover (memoryless lifetimes, mean L)
        dead=rng.random(K)<1.0/L
        if dead.any():
            pc[dead]=rng.uniform(0,W,(dead.sum(),2))

        # nearest patch for each agent
        v=torus_vec(pos[:,None,:],pc[None,:,:],W)# (N, K, 2)
        dist=np.hypot(v[...,0],v[...,1])
        j=dist.argmin(1)
        dmin=dist[idx,j]
        on=dmin<r_patch # feeding agents

        # intake, shared among agents on the same patch
        occ=np.bincount(j[on],minlength=K)
        cur[on]+=rate/occ[j[on]]**c
        target[on]=np.nan
        s=~on # searching agents

        # social information: feeding agents call, searchers may hear the nearest caller
        if q>0 and on.any():
            callers=pos[on]
            vc=torus_vec(pos[:,None,:],callers[None,:,:],W)
            dc=np.hypot(vc[...,0],vc[...,1])
            k=dc.argmin(1)
            hear=(s&np.isnan(target[:,0])&(dc[idx,k]<R_call)
                    &(rng.random(N)<q))
            target[hear]=callers[k[hear]]

        # heading: personal sensing > stored social target > random walk
        heading=th+rng.normal(0,turn_sd,N)
        sees=s&(dmin<r_sense)
        vp=v[idx,j]
        heading[sees]=np.arctan2(vp[sees,1],vp[sees,0])
        target[sees]=np.nan
        has_t=s&~sees&~np.isnan(target[:,0])
        if has_t.any():
            ids=np.where(has_t)[0]
            vt=torus_vec(pos[ids],target[ids],W)
            heading[ids]=np.arctan2(vt[:,1],vt[:,0])
            target[ids[np.hypot(vt[:,0],vt[:,1])<2.0]]=np.nan # arrived
        th=np.where(s,heading,th)

        # move searchers; feeding agents stay put
        step=speed*np.stack([np.cos(th),np.sin(th)],axis=1)
        pos=np.where(s[:,None],(pos+step)%W,pos)

        if position_history is not None:
            position_history.append(pos.copy())
        if patch_history is not None:
            patch_history.append(pc.copy())

        # record bout intake
        if t>=burn and(t-burn+1)%bout==0:
            out.append(cur.copy())
            cur[:]=0
    return np.array(out)# (n_bouts, N)

def plot_position_gif(q=0.3,L=80,seed=0,W=100.0,N=20,K=8,T=4500,
                      show_agents=20,filename="abm_progress_robust.gif",fps=30):
    """Save an animation with one frame per timestep.

    Note: storing every position and patch location uses O(T * (N + K)) memory.
    For long runs, reduce T or add a frame_stride before recording history.
    """
    positions=[]
    patches=[]
    simulate(q=q,L=L,seed=seed,W=W,N=N,K=K,T=T,
             burn=0,position_history=positions,patch_history=patches)
    history=np.asarray(positions)
    patch_history=np.asarray(patches)
    n_agents=min(show_agents,N)

    fig,ax=plt.subplots(figsize=(5,5))
    ax.set_xlim(0,W)
    ax.set_ylim(0,W)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    agent_colors=plt.get_cmap("tab20")(np.arange(n_agents))
    patch_scatter=ax.scatter(patch_history[0,:,0],patch_history[0,:,1],
                               s=38,color="tab:green",label="patches")
    agent_scatter=ax.scatter(history[0,:n_agents,0],history[0,:n_agents,1],
                               s=22,c=agent_colors,label="agents")
    trails=[ax.plot([],[],color=agent_colors[i],lw=1,alpha=0.65)[0]
              for i in range(n_agents)]
    timestep_label=ax.text(0.02,0.98,"",transform=ax.transAxes,va="top")
    ax.legend(loc="upper right",fontsize=8)

    def update(frame):
        patch_scatter.set_offsets(patch_history[frame])
        agent_scatter.set_offsets(history[frame,:n_agents])
        trail_start=max(0,frame-19)
        for i,line in enumerate(trails):
            line.set_data(history[trail_start:frame+1,i,0],
                          history[trail_start:frame+1,i,1])
        timestep_label.set_text("Timestep: {}".format(frame+1))
        return[patch_scatter,agent_scatter,timestep_label]+trails

    animation=FuncAnimation(fig,update,frames=T,interval=1000/fps,blit=False)
    animation.save(filename,writer="pillow",fps=fps,dpi=50)
    plt.close(fig)
    return filename

def plot_position_gif_comparison(L_values=(20,80,300,1500),q=0.3,seed=0,
                                W=100.0,N=20,K=8,T=300,show_agents=20,
                                filename="abm_progress_cleaned_comparison.gif",fps=20):
    """Save synchronized position animations for multiple patch lifetimes."""
    histories=[]
    patch_histories=[]
    for L in L_values:
        positions=[]
        patches=[]
        simulate(q=q,L=L,seed=seed,W=W,N=N,K=K,T=T,burn=0,
                 position_history=positions,patch_history=patches)
        histories.append(np.asarray(positions))
        patch_histories.append(np.asarray(patches))

    n_agents=min(show_agents,N)
    colors=plt.get_cmap("tab20")(np.arange(n_agents))
    fig,axes=plt.subplots(2,2,figsize=(10,10),constrained_layout=True)
    panels=[]

    for ax,L,history,patch_history in zip(axes.flat,L_values,histories,patch_histories):
        ax.set_xlim(0,W)
        ax.set_ylim(0,W)
        ax.set_aspect("equal")
        ax.set_title("Patch lifetime L = {}".format(L),fontsize=13)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        patch_scatter=ax.scatter(patch_history[0,:,0],patch_history[0,:,1],
                                   s=40,color="tab:green",label="patches")
        agent_scatter=ax.scatter(history[0,:n_agents,0],history[0,:n_agents,1],
                                   s=24,c=colors,label="agents")
        trails=[ax.plot([],[],color=colors[i],lw=1.2,alpha=0.75)[0]
                  for i in range(n_agents)]
        ax.legend(loc="upper right",fontsize=8)
        panels.append((history,patch_history,patch_scatter,agent_scatter,trails))

    timestep_label=fig.suptitle("",fontsize=14)

    def update(frame):
        artists=[timestep_label]
        timestep_label.set_text("Simulation progress: timestep {}".format(frame+1))
        trail_start=max(0,frame-19)
        for history,patch_history,patch_scatter,agent_scatter,trails in panels:
            patch_scatter.set_offsets(patch_history[frame])
            agent_scatter.set_offsets(history[frame,:n_agents])
            for i,line in enumerate(trails):
                line.set_data(history[trail_start:frame+1,i,0],
                              history[trail_start:frame+1,i,1])
            artists.extend([patch_scatter,agent_scatter])
            artists.extend(trails)
        return artists

    animation=FuncAnimation(fig,update,frames=T,interval=1000/fps,blit=False)
    animation.save(filename,writer="pillow",fps=fps,dpi=100)
    plt.close(fig)
    return filename

def summarise(arr,thr):
    """Summarise bout intake; return NaN for undefined small/empty samples."""
    arr=np.asarray(arr,dtype=float)
    if arr.size==0 or arr.ndim!=2:
        raise ValueError("arr must be a non-empty 2D array: bouts x agents.")
    mean=float(np.mean(arr))
    # Estimate within-agent variability across bouts, then average across agents.
    if arr.shape[0]>=2:
        var=float(np.var(arr,axis=0,ddof=1).mean())
    else:
        var=np.nan
    cv=np.sqrt(var)/mean if mean>0 and np.isfinite(var)else np.nan
    short=float(np.mean(arr<thr))
    return mean,var,cv,short

def analytic_cv(q,L,lam0=0.008,kappa=0.05,tau=25.0,B=200.0):
    """Renewal-reward approximation (see the LaTeX file).
    finding rate  lam = lam0 + kappa * q * exp(-tau / L)      (social information gets stale)
    patch loss    delta = 1 / L
    CV^2 of bout intake = 2 * delta / (B * lam * (lam + delta))   (valid for B >> 1/lam + 1/delta)
    Parameters are illustrative, not fitted."""
    lam=lam0+kappa*q*np.exp(-tau/L)
    delta=1.0/L
    return np.sqrt(2*delta/(B*lam*(lam+delta)))

def grid_search_tipping_point(W=100.0,N=20,K=8,L=100,T=4500,burn=500,bout=200):
    """
    Coarsely compares social vs silent conditions across competition strengths.
    This is a grid scan, not an exact tipping-point estimator.
    """
    # Define a clean grid of parameters to test
    c_values=[0.0,0.2,0.25,0.4,0.6,1.0]
    q_values=[0.0,0.2]# Comparing silent vs. social
    seed=42 # Fixed seed for a fair baseline comparison

    results={}

    print("Running parameter sweep across competition intensities...")
    print(f"Fixed conditions: L={L}, N={N}, K={K}, T={T}\n")

    for c in c_values:
        results[c]={}
        for q in q_values:
            # Call your original simulation function
            out=simulate(q=q,L=L,seed=seed,W=W,N=N,K=K,c=c,T=T,burn=burn,bout=bout)

            # Use a dummy threshold of 0 to get the overall mean intake
            mean_intake,_,_,_=summarise(out,thr=0)
            results[c][q]=mean_intake

    # Print a structured breakdown of the results
    print(f"{'Scramble (c)':<15}{'Silent (q=0.0)':<18}{'Social (q=0.2)':<18}{'Net Effect':<15}")
    print("-"*65)

    for c in c_values:
        mean_silent=results[c][0.0]
        mean_social=results[c][0.2]
        net_effect=mean_social-mean_silent

        # Visually tag if sociality is a benefit or a trap
        status="Gain" if net_effect>0 else "Loss"
        if abs(net_effect)<1.0:
            status="Balanced"

        print(f"{c:<15.2f}{mean_silent:<18.1f}{mean_social:<18.1f}{net_effect:<15.1f} {status}")

if __name__=="__main__":
    q_values=[0.0,0.01,0.03,0.1,0.3,1.0]# information availability
    L_values=[20,80,300,1500]# mean patch lifetime (steps): ephemeral -> persistent
    c_values=[(0.0,"No competition (c = 0)\nEach agent gets the full patch intake"),
                (1.0,"Strong competition (c = 1)\nPatch intake is shared among agents")]
    n_rep=6
    bout,rate=200,1.0
    thr=0.25*bout*rate # "survival threshold" per bout

    rows=[]
    for c,_ in c_values:
        for L in L_values:
            for q in q_values:
                res=np.array([summarise(simulate(q,L,seed=s,c=c,bout=bout,rate=rate),thr)
                                for s in range(n_rep)])
                m=res.mean(0)
                se=res.std(0,ddof=1)/np.sqrt(n_rep)
                rows.append([c,L,q,*m,*se])
                print(f"c={c:.0f} L={L:5d} q={q:5.2f} mean={m[0]:6.1f} var={m[1]:8.1f} "
                      f"CV={m[2]:.3f} shortfall={m[3]:.3f}",flush=True)

    header=["c","L","q","mean","var","cv","shortfall",
              "mean_se","var_se","cv_se","shortfall_se"]
    with open("abm_results_robust.csv","w",newline="")as f:
        w=csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # Figure 1: rows compare competition settings; columns compare food outcomes.
    data=np.array(rows)
    metrics=[(3,"Food per bout"),(4,"Variation in food between bouts"),
               (5,"Food variability relative to the average"),
               (6,"Chance of a low-food bout")]
    y_labels=["Food per bout","Food variation","Relative variation","Probability of low intake"]
    # A log axis cannot display q=0. Plot q=0 at a labelled visual placeholder;
    # the tick label makes clear that it is exactly zero, not q=0.003.
    xs=np.array([0.003 if q==0 else q for q in q_values])
    fig,ax=plt.subplots(2,4,figsize=(16,7),sharex=True)
    for r,(c,clabel)in enumerate(c_values):
        for col,(k,title)in enumerate(metrics):
            a=ax[r,col]
            for L in L_values:
                d=data[(data[:,0]==c)&(data[:,1]==L)]
                a.errorbar(xs,d[:,k],yerr=d[:,k+4],marker="o",capsize=2,
                           label=f"mean patch life={L}")
            a.set_xscale("log")
            a.set_xticks(xs)
            a.set_xticklabels(["0" if q==0 else f"{q:g}" for q in q_values])
            a.set_ylabel(y_labels[col],fontsize=9)
            if r==0:
                a.set_title(title,fontsize=10)
            if r==1:
                a.set_xlabel("Chance of hearing a call each step (q)")
        fig.text(0.015,0.69-0.39*r,clabel,ha="left",va="center",fontsize=9)
    ax[0,0].legend(fontsize=8)
    fig.subplots_adjust(left=0.2,bottom=0.13,top=0.9,wspace=0.45,hspace=0.38)
    plt.savefig("abm_results_cleaned.png",dpi=200)

    # Figure 2: analytic prediction (illustrative parameters)
    qq=np.linspace(0,1,100)
    plt.figure(figsize=(5.5,3.8))
    for L in L_values:
        plt.plot(qq,analytic_cv(qq,L),label=f"mean patch life={L}")
    plt.xlabel("Chance of hearing a call each step (q)")
    plt.ylabel("Food variability relative to the average")
    plt.title("Predicted food variability",fontsize=10)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("analytic_cv.png",dpi=200)
    plt.close()

    gif_file=plot_position_gif_comparison()
    print("saved abm_results_robust.csv, abm_results_cleaned.png, analytic_cv.png, {}".format(gif_file))

    # Execute the parameter sweep profile
    grid_search_tipping_point(
        W=100.0,
        N=20,
        K=8,
        L=100,# Dynamic patch lifetime
        T=4500,# Simulation timeline length
        burn=500,# Burn-in timeline window
        bout=200 # Interval slice window
    )
