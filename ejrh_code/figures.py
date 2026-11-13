# Copyright (c) 2026 Longqi Zhang

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, LogNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from .exposure import PERIODS, SSPS

width = 170 / 25.4
products = ["SMCI", "GLEAM"]
colors = {"SMCI":"#175583", "GLEAM":"#68A9C9"}
dark, light = '#10467f', '#69aacf'
ZONE = zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
zone_colors = dict(zip(zones, ["#c3dce9","#f3ddb0","#c5d9c0","#d8cfe2","#e8c5c2"]))
COL = {'SMCI':'#104581','GLEAM':'#69a9cf'}
INK = "#25343a"
HATCH_INK = "#4a5156"
HATCHES = {3:"////", 4:"//", 5:""}
EXPOSURE = "five_GCM_median"

def save(fig, output):
    fig.savefig(Path(output), facecolor="white")
    plt.close(fig)

def light_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#E6EBED", linewidth=.48)

def err(ax,x,y,lo,hi,color,marker="o",filled=True,orient="x"):
    if orient=="x":
        ax.hlines(y,lo,hi,color=color,lw=.94,zorder=2)
    else:
        ax.vlines(x,lo,hi,color=color,lw=.86,zorder=2)
    ax.plot(x,y,marker=marker,ms=3.65,mec=color,mew=.9,mfc=color if filled else "white",
            ls="none",zorder=3)

def left_title(ax,label):
    ax.set_title(label, loc="left", pad=5)

def legend_products(ax, **kwargs):
    return ax.legend(handles=[Line2D([],[],color=colors[p],marker="o",ms=3.7,lw=1.15,label=p) for p in products],
                     frameon=False,fontsize=7.7,**kwargs)

def aesthetics(size=8):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':size,'axes.labelsize':size,
                         'axes.spines.right':False,'axes.spines.top':False,'axes.edgecolor':'#49545d',
                         'axes.linewidth':.7,'axes.axisbelow':True,'grid.color':'#e6eaed','grid.linewidth':.6,
                         'xtick.color':'#202830','ytick.color':'#202830','text.color':'#202830',
                         'pdf.fonttype':42})

def fig3(corr, terms, output):
    fig=plt.figure(figsize=(width,4.65))
    g=fig.add_gridspec(2,1,height_ratios=[1.78,1],left=.125,right=.975,bottom=.095,top=.93,hspace=.48)
    a=fig.add_subplot(g[0]); b=fig.add_subplot(g[1])
    left_title(a,"(a) Soil-state persistence, 2001–2022")
    for p in products:
        for tag,ls in [("seasonal","-"),("seasonal_linear_detrend","--")]:
            q=corr[(corr["product"]==p)&(corr.transform_id==tag)].sort_values("lag_months")
            a.fill_between(q.lag_months,q.ci_low,q.ci_high,color=colors[p],alpha=.11 if ls=="-" else .075,linewidth=0)
            a.plot(q.lag_months,q.estimate,color=colors[p],ls=ls,lw=1.3,marker="o" if ls=="-" else None,ms=2.5)
    a.axhline(0,color="#879398",ls=":",lw=.65)
    a.set(xlim=(-.15,12.15),xticks=range(13),ylim=(-.15,1.035),yticks=[0,.25,.50,.75,1],
          xlabel="Lag (calendar months)",ylabel="State correlation, r")
    light_grid(a)
    leg=legend_products(a,loc="upper right",ncol=2,borderaxespad=.15)
    a.add_artist(leg)
    a.legend(handles=[Line2D([],[],color="#465259",ls="-",label="Seasonal anomalies"),
                      Line2D([],[],color="#465259",ls="--",label="Linearly detrended")],
             loc="upper right",bbox_to_anchor=(1,.84),frameon=False,fontsize=7.6)
    left_title(b,"(b) Conditional lag-1 soil-state coefficients, 2001–2014")
    for j,c in enumerate(["WOODY","HERBACEOUS","SPARSE"]):
        for p,dy in zip(products,[.11,-.11]):
            q=terms[(terms["product"]==p)&(terms.stratum==c)].iloc[0]
            err(b,q.estimate,2-j+dy,q.ci_low,q.ci_high,colors[p])
    b.set(yticks=[2,1,0],yticklabels=["Woody","Herbaceous","Barren cover"],
          ylim=(-.38,2.38),xlim=(0,.9),xticks=np.arange(0,1,.2),xlabel="Lag-1 coefficient")
    light_grid(b,"x")
    b.text(.995,.02,"SMCI F3_Dfree  |  GLEAM F2_Dfree",transform=b.transAxes,ha="right",va="bottom",fontsize=7.5,color="#4B565E")
    save(fig, output)

def fig4(a07, a08, output):
    ids=['C_SMCI_SPEI3_U6','V_SMCI_Y_U6','V_SMCI_FVC_U6','C_GLEAM_SPEI3_U6','V_GLEAM_Y_U6','V_GLEAM_FVC_U6']
    titles=['SPEI3 → SMCI','SMCI → kNDVI','SMCI → FVC','SPEI3 → GLEAM SMrz','GLEAM SMrz → kNDVI','GLEAM SMrz → FVC']
    u=a07.set_index('spec_id').loc[ids].reset_index()
    l=a08.query("stratum_or_region=='REGION' and spec_id in @ids").copy()
    l['panel']=l.spec_id.map(dict(zip(ids,'abcdef')))
    l['interval_drawn']=l.panel.isin(['c','f'])
    aesthetics(8.5)
    fig,axs=plt.subplots(2,3,figsize=(6.69,4.33),sharex=True,sharey=True)
    fig.subplots_adjust(left=.105,right=.982,bottom=.115,top=.905,wspace=.17,hspace=.33)
    min_y=min(l.coefficient.min(),l.loc[l.interval_drawn,'ci_low'].min(),-.11)
    max_y=max(l.coefficient.max(),l.loc[l.interval_drawn,'ci_high'].max())
    ylo=np.floor(min_y*10)/10-.035; yhi=np.ceil(max_y*10)/10+.16
    for i,(ax,key,title) in enumerate(zip(axs.flat,ids,titles)):
        g=l[l.spec_id==key].sort_values('lag_months')
        assert g.lag_months.tolist()==list(range(7))
        summ=u[u.spec_id==key].iloc[0]
        assert abs(g.coefficient.sum()-summ.cumulative_net)<1e-12
        assert abs(np.average(g.lag_months,weights=g.coefficient.abs())-summ.centroid_lag)<1e-12
        col=dark if i<3 else light
        ax.vlines(g.lag_months,0,g.coefficient,color=col,alpha=.63,lw=.85)
        if i in [2,5]:
            for q in g.itertuples():
                ax.vlines(q.lag_months,q.ci_low,q.ci_high,color='#5f6c76',lw=.75)
                ax.hlines([q.ci_low,q.ci_high],q.lag_months-.13,q.lag_months+.13,color='#5f6c76',lw=.75)
        ax.plot(g.lag_months,g.coefficient,'o',color=col,ms=4.7,mew=.55,mec=dark if i<3 else '#3b83b1')
        ax.axhline(0,color='#a0aab3',ls='--',lw=.65); ax.grid(axis='y')
        ax.set(xlim=(-.45,6.45),ylim=(ylo,yhi),xticks=range(7))
        ax.set_title(title,fontweight='bold',fontsize=9.1,pad=4)
        ax.text(-.035,1.065,f'({"abcdef"[i]})',transform=ax.transAxes,fontweight='bold')
        ax.text(.968,.94,f'Cumulative = {summ.cumulative_net:.3f}\nCentroid = {summ.centroid_lag:.2f} mo',ha='right',va='top',transform=ax.transAxes,color='#4c5963',fontsize=7.7)
    fig.supxlabel('Lag (calendar months)',fontsize=9,x=.54,y=.025)
    fig.supylabel('Association coefficient',fontsize=9,x=.025)
    save(fig, output)

def fig7(z, zonepair, output):
    fig=plt.figure(figsize=(170/25.4,5.5))
    g=fig.add_gridspec(2,2,left=.092,right=.973,bottom=.105,top=.94,wspace=.25,hspace=.49,height_ratios=[1,1.05])
    a=fig.add_subplot(g[0,0]); b=fig.add_subplot(g[0,1])
    a.set_title("(a) Retained units by zone",loc="left",pad=5)
    za=z[(z["product"]=="SMCI")&(z.lag==3)].pivot(index="zone",columns="stratum",values="unit_n").loc[zones]
    for j,(zone,row) in enumerate(za.iterrows()):
        i=4-j
        for k,dy,fill in [("ALL",.155,True),("HERBACEOUS",-.155,False)]:
            n=int(row[k]); color=zone_colors[zone]
            a.barh(i+dy,n,height=.252,facecolor=color if fill else "white",edgecolor="#68787F",linewidth=.68)
            a.text(n+.65,i+dy,str(n),va="center",fontsize=7.3,color="#25333A")
    a.set(xlim=(0,45),xticks=[0,10,20,30,40],yticks=[4,3,2,1,0],yticklabels=zones,
          ylim=(-.43,4.45),xlabel="Analysis units")
    light_grid(a)
    b.set_title("(b) Z5 − Z3 contrast",loc="left",pad=5)
    for j,(p,lag) in enumerate([(p,k) for p in products for k in [3,6]]):
        for st,dy,solid in [("ALL",.115,True),("HERBACEOUS",-.115,False)]:
            q=zonepair[(zonepair["product"]==p)&(zonepair.lag==lag)&(zonepair.stratum==st)].iloc[0]
            err(b,q.contrast_estimate,3-j+dy,q.contrast_simultaneous_low,q.contrast_simultaneous_high,colors[p],filled=solid)
    b.axvline(0,color="#879398",lw=.65,ls=":")
    b.set(yticks=[3,2,1,0],yticklabels=["SMCI · 3","SMCI · 6","GLEAM · 3","GLEAM · 6"],
          ylim=(-.46,3.48),xlabel="Difference in r (simultaneous 95% interval)")
    light_grid(b)
    for j,p in enumerate(products):
        sub=g[1,j].subgridspec(1,2,wspace=.18)
        for k,lag in enumerate([3,6]):
            ax=fig.add_subplot(sub[0,k])
            for i,zone in enumerate(zones):
                for st,dy,filled in [("ALL",.115,True),("HERBACEOUS",-.115,False)]:
                    q=z[(z["product"]==p)&(z.lag==lag)&(z.zone==zone)&(z.stratum==st)].iloc[0]
                    err(ax,q.estimate,4-i+dy,q.ci_low,q.ci_high,colors[p],filled=filled)
            ax.axvline(0,color="#879398",lw=.65,ls=":")
            ax.set(xlim=(-.39,.85),xticks=[-.25,0,.25,.5,.75],yticks=[4,3,2,1,0],
                   yticklabels=zones if not k else [],ylim=(-.42,4.45))
            ax.tick_params(axis="x",labelsize=7.1,pad=2)
            if k: ax.tick_params(axis="y",length=0)
            light_grid(ax); ax.set_title(f"Lag {lag}",fontsize=8,pad=4)
            if not k:
                ax.text(0,1.24,f"({chr(99+j)}) {p} zonal estimates",transform=ax.transAxes,ha="left",
                        va="top",weight="bold",fontsize=9)
    fig.text(.53,.050,"State correlation, r (pointwise 95% interval)",ha="center",fontsize=8)
    fig.legend(handles=[Line2D([],[],color="#3D525E",marker="o",ls="none",ms=4,label="All retained units"),
                        Line2D([],[],color="#3D525E",marker="o",ls="none",mfc="white",ms=4,label="Common herbaceous subset")],
               loc="lower center",bbox_to_anchor=(.53,.001),ncol=2,frameon=False,fontsize=7.6)
    save(fig, output)

def fig_s4(units, env, output):
    clim=env[env.term.eq("D_per_100_mm") & env["product"].isin(products)].copy()
    specs=[
     ("climate_only","Climate only"),("climate_coarse_cover","+ Coarse cover"),
     ("climate_common_herb","Common herbaceous"),("climate_cover_zone","+ Cover + zone"),
     ("climate_cover_zone_PAWC33","+ Cover + zone + PAWC33"),
     ("climate_cover_fractions","+ Cover fractions"),
     *[(f"LOZO_without_Z{i}",f"Omit Z{i}") for i in range(1,6)]]
    fig=plt.figure(figsize=(width,6.0))
    g=fig.add_gridspec(2,2,left=.30,right=.976,top=.92,bottom=.105,hspace=.48,wspace=.16,height_ratios=[.88,1.18])
    scatter_values=units[[f"{p}_z{k}" for p in products for k in [3,6]]].stack()
    yl=(scatter_values.min()-.08,scatter_values.max()+.08)
    xrange=clim[["ci_low","ci_high","estimate"]].stack()
    xlo,xhi=xrange.min()-.02,xrange.max()+.02
    for j,p in enumerate(products):
        a=fig.add_subplot(g[0,j]); b=fig.add_subplot(g[1,j])
        left_title(a,f"({chr(97+j)}) {p} unit observations")
        left_title(b,f"({chr(99+j)}) {p} climate slopes")
        for k,marker,filled,dy in [(3,"o",True,.15),(6,"s",False,-.15)]:
            a.scatter(units.D_mm_season,units[f"{p}_z{k}"],s=9,marker=marker,
                      facecolors=colors[p] if filled else "none",edgecolors=colors[p],lw=.48,alpha=.61)
            for i,(spec,label) in enumerate(specs):
                q=clim[(clim["product"]==p)&(clim.lag==k)&(clim.specification==spec)].iloc[0]
                y=len(specs)-i-1+dy
                if q.n_resamples<950:
                    b.plot(q.estimate,y,marker="x",ms=4,color="#A85D4D",ls="none")
                else:
                    err(b,q.estimate,y,q.ci_low,q.ci_high,colors[p],marker=marker,filled=filled)
        a.set(ylim=yl,xticks=[-750,-500,-250])
        a.set_xlabel("May–September P − PET (mm/season)")
        if j==0: a.set_ylabel("Unit Fisher z")
        else: a.tick_params(labelleft=False)
        light_grid(a)
        b.set(xlim=(xlo,xhi),ylim=(-.52,len(specs)-.46),yticks=np.arange(len(specs)),
              yticklabels=[x[1] for x in specs][::-1] if j==0 else [])
        b.axvline(0,color="#8C989E",lw=.65,ls=":")
        if j: b.tick_params(axis="y",length=0)
        light_grid(b,"x")
    fig.text(.64,.057,"Slope (Fisher z per 100 mm/season), pointwise 95% interval",ha="center",fontsize=7.7)
    fig.legend(handles=[Line2D([],[],color="#28658B",marker="o",ls="none",ms=4,label="Lag 3"),
                       Line2D([],[],color="#28658B",marker="s",mfc="white",ls="none",ms=4,label="Lag 6"),
                       Line2D([],[],color="#A85D4D",marker="x",ls="none",ms=4,label="Insufficient valid refits: point only")],
               loc="lower center",bbox_to_anchor=(.64,.002),frameon=False,ncol=3,fontsize=7.5)
    save(fig, output)

def fig_s5(full, output):
    f5=full[(full.stratum_or_region=="REGION")&(full.mask_version=="original2020")&
            full.period_id.isin(["2001-2022","2001-2014"])&full["product"].isin(products)&
            full.transform_id.isin(["seasonal","seasonal_linear_detrend"])&
            full.aggregation_weight_id.eq("fixed_valid_area_FisherZ")].copy()
    fig,axs=plt.subplots(1,2,figsize=(width,2.55),sharey=True)
    fig.subplots_adjust(left=.087,right=.982,top=.84,bottom=.26,wspace=.14)
    for ax,period in zip(axs,["2001-2022","2001-2014"]):
        left_title(ax,period.replace("-","–"))
        for p in products:
            for tag,ls in [("seasonal","-"),("seasonal_linear_detrend","--")]:
                q=f5[(f5["product"]==p)&(f5.period_id==period)&(f5.transform_id==tag)].sort_values("lag_months")
                ax.fill_between(q.lag_months,q.ci_low,q.ci_high,color=colors[p],alpha=.1,linewidth=0)
                ax.plot(q.lag_months,q.estimate,color=colors[p],ls=ls,lw=1.15)
        ax.set(xlabel="Lag (calendar months)",xticks=[0,3,6,9,12],ylim=(-.24,1.04))
        ax.axhline(0,ls=":",color="#879398",lw=.65); light_grid(ax)
    axs[0].set_ylabel("State correlation, r")
    fig.legend(handles=[Line2D([],[],color=colors[p],label=p) for p in products]+
                       [Line2D([],[],color="#555555",ls="-",label="Seasonal"),
                        Line2D([],[],color="#555555",ls="--",label="Detrended")],
               loc="lower center",ncol=4,frameon=False,fontsize=7.6)
    save(fig, output)

def fig_s3(data, output):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
        'font.size':8,'axes.titlesize':9,'axes.titleweight':'bold','axes.labelsize':8.3,'axes.edgecolor':'#596670',
        'axes.labelcolor':'#202b34','xtick.color':'#34414a','ytick.color':'#34414a','text.color':'#202b34',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,
        'axes.axisbelow':True,'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axs=plt.subplots(2,3,figsize=(170/25.4,121/25.4))
    fig.subplots_adjust(left=.103,right=.983,bottom=.174,top=.938,wspace=.26,hspace=.50)
    axes=axs.ravel()
    for i,z in enumerate(ZONE):
        a=axes[i]
        d=data[(data.zone==z)&(data.kind=='curve')]
        u=d[(d.stratum=='ALL')&(d['product']=='SMCI')].iloc[0]
        h=d[(d.stratum=='HERBACEOUS')&(d['product']=='SMCI')].iloc[0]
        a.set_title(f'({chr(97+i)})  {z}',loc='left',pad=5)
        a.text(.97,.94,f'{int(u.unit_n)} / {int(h.unit_n)} units',transform=a.transAxes,ha='right',va='top',fontsize=7.1,color='#4a5660')
        a.set_ylim(-.35,1.07)
        a.set_xlim(-.35,12.3)
        a.set_xticks([0,3,6,9,12])
        a.set_yticks([-.25,0,.25,.5,.75,1])
        a.set_yticklabels(['−0.25','0','0.25','0.50','0.75','1.00'] if i%3==0 else ['']*6)
        a.grid(axis='y',color='#dde4e8',lw=.46)
        a.axhline(0,ls=(0,(3,3)),color='#7e8c97',lw=.75)
        for p,c in COL.items():
            q=d[(d['product']==p)&(d.stratum=='ALL')].sort_values('lag')
            a.fill_between(q.lag,q.ci_low,q.ci_high,color=c,alpha=.15,lw=0,zorder=1)
            a.plot(q.lag,q.estimate,color=c,lw=1.38,zorder=4)
            k=q[q.lag.isin([3,6])]
            a.scatter(k.lag,k.estimate,color=c,s=9,zorder=5)
            q=d[(d['product']==p)&(d.stratum=='HERBACEOUS')].sort_values('lag')
            if z!='Z4':
                a.plot(q.lag,q.estimate,color=c,lw=1,ls=(0,(3,2)),zorder=3)
        a.tick_params(length=2.5,width=.65,pad=2)
    axes[0].set_ylabel('State correlation, r')
    axes[3].set_ylabel('State correlation, r')
    axes[3].set_xlabel('Lag (months)')
    axes[4].set_xlabel('Lag (months)')
    a=axes[5]
    a.set_title('(f)  SMCI minus GLEAM',loc='left',pad=5)
    q=data[data.kind=='product_difference']
    a.set_ylim(min(-.46,float(q.ci_low.min())-.06),max(.84,float(q.ci_high.max())+.12))
    a.set_xlim(-.5,4.5)
    a.grid(axis='y',color='#dde4e8',lw=.46)
    a.yaxis.set_major_locator(MultipleLocator(.25))
    a.axhline(0,ls=(0,(3,3)),color='#7e8c97',lw=.75)
    for lag,offset,marker,color,fill in [(3,-.14,'o','#465864','#465864'),(6,.14,'s','#465864','white')]:
        v=data[(data.kind=='product_difference')&(data.lag==lag)].set_index('zone').loc[ZONE]
        a.vlines(np.arange(5)+offset,v.ci_low,v.ci_high,color=color,lw=.8)
        a.scatter(np.arange(5)+offset,v.estimate,s=14,marker=marker,edgecolor=color,facecolor=fill,lw=.85,zorder=3,label=f'{lag} months')
    a.set_xticks(range(5),ZONE)
    a.tick_params(length=2.5,width=.65,pad=2)
    a.set_xlabel('Reporting zone')
    a.legend(loc='upper right',fontsize=6.7,frameon=False,handletextpad=.2,borderpad=.2,labelspacing=.25)
    items=[Line2D([0],[0],color=c,lw=2,label='GLEAM SMrz' if p=='GLEAM' else p) for p,c in COL.items()]
    items+=[Line2D([0],[0],color='#53616d',lw=1.4,label='All units'),
        Line2D([0],[0],color='#53616d',lw=1.2,ls=(0,(3,2)),label='Common herbaceous')]
    fig.legend(handles=items,loc='lower center',bbox_to_anchor=(.51,.022),ncol=4,frameon=False,columnspacing=1.35,handlelength=2.2,fontsize=7.6)
    save(fig, output)

def heading(ax, title, period=None):
    ax.text(.024, .957, title, transform=ax.transAxes, fontsize=8.4, ha="left", va="top",
            color=INK, fontweight="bold", zorder=9)
    if period:
        ax.text(.025, .825, period, transform=ax.transAxes, fontsize=7.4,
                ha="left", va="top", color=INK, zorder=9)

def bar(fig, rect, norm, cmap, ticks, label, extend="neither"):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(rect),
                      orientation="horizontal", ticks=ticks, extend=extend)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.solids.set_linewidth(.14)
    cb.outline.set_linewidth(.46)
    cb.ax.tick_params(length=0, pad=2, labelsize=7.2, labelcolor=INK)
    cb.set_label(label, fontsize=7.5, labelpad=2, color=INK)
    return cb

def fig5c(geometry, contrast, value_column, output):
    """Fig. 5c block-error contrast; symmetric P95 display clipping."""
    joined = geometry.merge(contrast, left_on="spatial_block_id", right_on="block_id",
                            validate="one_to_one")
    if len(joined) != len(geometry) or len(joined) != len(contrast):
        raise ValueError("Every supplied evaluation block needs one geometry and value")
    delta = 1e4 * joined[value_column]
    limit = float(np.percentile(abs(delta), 95))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Need a finite positive P95 display scale")
    norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    fig.subplots_adjust(bottom=.28, top=.97)
    joined.assign(display=delta).plot(column="display", ax=ax, cmap="RdBu", norm=norm,
                                     linewidth=0, edgecolor="none")
    heading(ax, "(c) Block error contrast")
    bar(fig, [.17,.135,.66,.025], norm, "RdBu", [-limit,0,limit],
        "Δ MSE reduction (0.0001 kNDVI²)", extend="both")
    fig.text(.5,.025,f"Colour saturated beyond ±P95 ({(abs(delta)>limit).sum()} blocks)",ha="center",fontsize=7)
    save(fig, output)


def fig6e(curve, output):
    """Render the confirmation-month recovery inset from the paired risk/curve table."""
    q = curve.sort_values("month_since_origin")
    if q.month_since_origin.duplicated().any() or not q.estimate.dropna().is_monotonic_increasing:
        raise ValueError("Expected one primary cumulative recovery curve")
    aesthetics(8.7)
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    fig.subplots_adjust(left=.19,right=.96,bottom=.38,top=.85)
    x = q.month_since_origin
    ax.step(x,q.estimate,where='post',color='#2c78ad',lw=1.5)
    ax.fill_between(x,q.ci_low,q.ci_high,step='post',color='#2c78ad',alpha=.16,lw=0)
    ticks = list(range(0, int(x.max()) + 1, 2))
    ax.set(xlim=(0, x.max() + .2), ylim=(0,1.04), xticks=ticks,
           yticks=[0,.25,.5,.75,1], xlabel='Months since suppression onset', ylabel='Cumulative recovery')
    ax.grid(axis='y')
    fig.text(.035,.95,'(e)   Confirmed recovery',va='top',fontsize=9.1,fontweight='bold')
    fig.text(.035,.132,'At risk',fontsize=7.8,fontweight='bold')
    for m, n in q.loc[q.month_since_origin.isin(ticks), ["month_since_origin", "n_risk"]].itertuples(index=False):
        ax.text(m,-.51,str(int(n)),transform=ax.get_xaxis_transform(),ha='center',fontsize=7.6)
    save(fig, output)


def fig8(values, zone_geometry, output, limit=.774):
    """Six full-domain exposure maps from authorized zone polygons and estimates."""
    if zone_geometry.crs is None:
        raise ValueError("Zone polygons must declare their CRS")
    geo = zone_geometry.to_crs(4326)
    key = "assigned_zone_id"
    if not geo[key].is_unique or set(geo[key]) != set(ZONE):
        raise ValueError("Require exactly Z1–Z5 with one geometry each")
    if not (values.n_gcm.eq(5) & values.variant.eq("CORRECTED_WARMUP") &
            values.period_statistic.eq("median") & values[EXPOSURE].abs().le(limit)).all():
        raise ValueError("Use five-GCM corrected period medians within the display range")
    norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    fig, axes = plt.subplots(2, 3, figsize=(170/25.4,104/25.4))
    fig.subplots_adjust(left=.064,right=.991,top=.879,bottom=.198,wspace=.029,hspace=.070)
    fig.text(.064,.966,"Five-zone full-domain climatic exposure",va="top",fontsize=10,fontweight="bold")
    for i, (lo, hi) in enumerate(PERIODS):
        for j, (ssp, label) in enumerate(zip(SSPS, ("SSP1-2.6","SSP2-4.5","SSP5-8.5"))):
            ax = axes[i,j]
            panel = values.loc[values.ssp.eq(ssp) & values.period_start.eq(lo) & values.period_end.eq(hi)]
            joint = geo.merge(panel, left_on=key, right_on="support_id", validate="one_to_one")
            if len(joint) != len(geo) or not joint.direction_agreement.isin(HATCHES).all():
                raise ValueError("Incomplete panel or invalid five-GCM agreement")
            joint.plot(ax=ax, column=EXPOSURE, norm=norm, cmap="RdBu_r", edgecolor="#60717a", linewidth=.41)
            for n, hatch in HATCHES.items():
                group = joint[joint.direction_agreement.eq(n)]
                if hatch and not group.empty:
                    group.plot(ax=ax,facecolor="none",hatch=hatch,edgecolor=HATCH_INK,linewidth=0,alpha=.55)
            for row in joint.itertuples():
                p = row.geometry.representative_point()
                ax.text(p.x,p.y,row.support_id,ha="center",va="center",fontsize=7.25,fontweight="bold",
                        path_effects=[effects.withStroke(linewidth=1.55,foreground="white")])
            ax.tick_params(labelbottom=i==1,labelleft=j==0,labelsize=6)
            heading(ax,f"({chr(97+i*3+j)}) {label}",f"{lo}–{hi}")
    bar(fig,[.086,.123,.541,.015],norm,"RdBu_r",[-limit,-limit/2,0,limit/2,limit],
        "Median ΔCWBI3 (baseline SD)")
    handles = [Patch(facecolor="#f2f4f3",edgecolor=HATCH_INK,hatch=h,label=f"{n}/5")
               for n,h in HATCHES.items()]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.818,.155),ncol=3,
               title="Direction agreement",title_fontsize=7.5,fontsize=7.15,frameon=False)
    save(fig, output)


def fig5(a5, p5, windows, geometry, contrast, value_column, output):
    tasks=["TEMPORAL_2015_2022","SPATIAL_NEW_POOLED"]
    rms=p5.assign(se=(p5.y_raw-p5.yhat_raw)**2).groupby("model_id").se.mean().pow(.5)
    mapped = geometry.merge(contrast, left_on="spatial_block_id", right_on="block_id",
                            validate="one_to_one")
    if len(mapped) != len(geometry) or len(mapped) != len(contrast):
        raise ValueError("Each block must have one geometry and one contrast value")
    fig=plt.figure(figsize=(width,5.65))
    g=fig.add_gridspec(2,2,left=.125,right=.965,bottom=.11,top=.91,wspace=.37,hspace=.52)
    a=fig.add_subplot(g[0,0]); b=fig.add_subplot(g[0,1]); c=fig.add_subplot(g[1,0]); d=fig.add_subplot(g[1,1])
    left_title(a,"(a) Soil information increments")
    contrasts=["CURRENT","PAST_GIVEN_CURRENT","TOTAL"]
    for i,task in enumerate(tasks):
        for j,term in enumerate(contrasts):
            for prod,dy in zip(products,[.095,-.095]):
                q=a5[(a5.task_id==task)&(a5.contrast_id==term)&(a5["product"]==prod)].iloc[0]
                err(a,q.relative_RMSE_reduction_pct,5-j-i*3.6+dy,q.CI_lower,q.CI_upper,colors[prod])
    a.axvline(0,color="#849098",ls="--",lw=.6)
    a.axhline(2.2,color="#C7D2D6",lw=.55)
    a.set(yticks=[5,4,3,1.4,.4,-.6],yticklabels=["Current","Past","Total"]*2,
          ylim=(-.96,6.3),xlabel="Relative RMSE reduction (%)")
    a.text(.025,5.86,"Time holdout, 2015–2022",transform=a.get_yaxis_transform(),fontsize=7.1,weight="bold")
    a.text(.025,1.94,"New locations, 2001–2022",transform=a.get_yaxis_transform(),fontsize=7.1,weight="bold")
    light_grid(a,"x")
    fig.legend(handles=[Line2D([],[],color=colors[p],marker="o",ms=3.7,lw=1.15,label=p) for p in products],
               loc="upper center",bbox_to_anchor=(.29,.99),ncol=2,frameon=False,fontsize=7.7)
    left_title(b,"(b) Time holdout, M2 predictions")
    b.set_axis_off()
    lo=min(0,p5.yhat_raw.min(),p5.y_raw.min()); hi=max(p5.yhat_raw.max(),p5.y_raw.max())*1.015
    b.text(.47,-.18,"Observed raw kNDVI",ha="center",transform=b.transAxes)
    b.text(-.17,.45,"Predicted raw kNDVI",va="center",rotation=90,transform=b.transAxes)
    for j,prod in enumerate(products):
        ax=b.inset_axes([j*.478,.18,.402,.71])
        q=p5[p5.model_id.eq(f"M2_{prod}")]
        h=ax.hexbin(q.y_raw,q.yhat_raw,gridsize=24,extent=(lo,hi,lo,hi),cmap="Blues",norm=LogNorm(vmin=1,vmax=1000),mincnt=1,linewidths=0)
        ax.plot([lo,hi],[lo,hi],color="#B75346",ls=":",lw=.8)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xticks=[0,.25,.5,.75,1],yticks=[0,.25,.5,.75,1])
        ax.tick_params(labelsize=6.35,pad=1.5)
        if j: ax.tick_params(labelleft=False)
        ax.set_title(prod,color=colors[prod],pad=3,fontsize=8)
        ax.text(.5,-.30,"M1/M2 RMSE\n"+f"{rms[f'M1_{prod}']:.3f} / {rms[f'M2_{prod}']:.3f}",
                ha="center",transform=ax.transAxes,fontsize=6.8)
    b.text(.47,-.29,f"M0 RMSE {rms['M0']:.3f}  |  n = {p5.row_id.nunique():,} per product",ha="center",transform=b.transAxes,fontsize=7)
    cbar=fig.colorbar(h,cax=b.inset_axes([.906,.18,.032,.71]),ticks=[1,10,100,1000],format="%d")
    cbar.ax.minorticks_off()
    cbar.ax.set_title("n",fontsize=6.6,pad=3)
    cbar.ax.tick_params(labelsize=5.5,length=1.6,pad=1.5)
    _draw_fig5c(fig, c, mapped, value_column)
    left_title(d,"(d) Rolling-origin total increment")
    for prod,shift in zip(products,[-.11,.11]):
        q=windows[windows["product"]==prod].sort_values("task_id")
        roll=q[q.task_id.ne("ROLLING_POOLED")]
        d.plot(np.arange(len(roll))+shift,roll.relative_RMSE_reduction_pct,color=colors[prod],lw=.75,alpha=.7)
        for idx,(_,row) in enumerate(roll.iterrows()):
            err(d,idx+shift,row.relative_RMSE_reduction_pct,row.CI_lower,row.CI_upper,colors[prod],orient="y")
        row=q[q.task_id.eq("ROLLING_POOLED")].iloc[0]
        err(d,6.5+shift,row.relative_RMSE_reduction_pct,row.CI_lower,row.CI_upper,colors[prod],marker="D",orient="y")
    d.axhline(0,color="#879398",lw=.65,ls=":")
    d.axvline(5.75,color="#CBD4D8",lw=.6)
    d.set(xlim=(-.45,7),xticks=[0,1,2,3,4,5,6.5],
          xticklabels=["11–12","13–14","15–16","17–18","19–20","21–22","Pooled"],ylabel="RMSE reduction (%)")
    d.tick_params(axis="x",labelsize=6.6)
    d.set_xlabel("Evaluation years (20xx)")
    light_grid(d)
    save(fig, output)


def _draw_fig5c(fig, ax, mapped, value_column):
    delta = 1e4 * mapped[value_column]
    p95 = float(np.percentile(abs(delta), 95))
    if not np.isfinite(p95) or p95 <= 0:
        raise ValueError("Need a finite positive P95 display scale")
    norm = TwoSlopeNorm(vcenter=0, vmin=-p95, vmax=p95)
    mapped.assign(display=delta).plot(column="display", ax=ax, cmap="RdBu", norm=norm,
                                     linewidth=0, edgecolor="none")
    left_title(ax, "(c) Block error contrast")
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap="RdBu"), ax=ax, orientation="horizontal",
                      fraction=.065,pad=.22,extend="both",ticks=[-p95,0,p95])
    cb.set_label("Δ MSE reduction (0.0001 kNDVI²)",fontsize=6.8)
    cb.ax.tick_params(labelsize=6)
    ax.text(.5,-.48,f"Colour saturated beyond ±P95 ({(abs(delta)>p95).sum()} blocks)",
            ha="center",transform=ax.transAxes,fontsize=6.2)


PALE_OCEAN = "#EDF4F7"
OUTSIDE_LAND = "#F6F5EF"

def draw_context(ax, world, counties, extent):
    ax.set_facecolor(PALE_OCEAN)
    ax.add_patch(Rectangle((extent[0], extent[2]), extent[1]-extent[0], extent[3]-extent[2], facecolor=PALE_OCEAN, edgecolor="none", zorder=-4))
    if not world.empty:
        world.plot(ax=ax, facecolor=OUTSIDE_LAND, edgecolor="#BFC9CD", linewidth=0.18, zorder=-3)
        world.boundary.plot(ax=ax, color="#BFC9CD", linewidth=0.18, zorder=-2)
    if not counties.empty:
        counties.boundary.plot(ax=ax, color="#D8DCDD", linewidth=0.10, zorder=-1)

def draw_panel(ax, mapped, outer, world, counties, inside_counties, value, cmap, norm, show_lon, show_lat, extent, lon_ticks, lat_ticks):
    draw_context(ax, world, counties, extent)
    mapped.plot(ax=ax, column=value, cmap=cmap, norm=norm, edgecolor="none", linewidth=0, missing_kwds={"color":"#ECEDEB", "edgecolor":"none"}, rasterized=True, zorder=2)
    if not inside_counties.empty:
        inside_counties.boundary.plot(ax=ax, color="#AEB5B8", linewidth=0.22, zorder=3)
    outer.boundary.plot(ax=ax, color="#414B50", linewidth=0.92, zorder=4)
    ax.set_xlim(*extent[:2]); ax.set_ylim(*extent[2:]); ax.set_aspect("equal", adjustable="box")
    if show_lon:
        ax.set_xticks(lon_ticks); ax.set_xticklabels([f"{v}°E" for v in lon_ticks])
    else:
        ax.set_xticks([])
    if show_lat:
        ax.set_yticks(lat_ticks); ax.set_yticklabels([f"{v}°N" for v in lat_ticks])
    else:
        ax.set_yticks([])
    ax.minorticks_off()
    ax.tick_params(axis="both", which="both", bottom=show_lon, labelbottom=show_lon, left=show_lat, labelleft=show_lat, top=False, labeltop=False, right=False, labelright=False, direction="out", length=0, width=0, pad=2.2, labelsize=7.7)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.50); spine.set_color("#596268")

def fig2(mapped, outer, world, counties, inside_counties, specifications, extent, output):
    """Draw six historical hydroclimate maps from supplied layers and scale specifications."""
    if len(specifications) != 6:
        raise ValueError("Provide six map specifications in figure order")
    layers = [x.to_crs(4326) for x in (mapped, outer, world, counties, inside_counties)]
    fig, axes = plt.subplots(2, 3, figsize=(170/25.4, 4.3))
    fig.subplots_adjust(left=.085, right=.985, bottom=.105, top=.93, wspace=.19, hspace=.32)
    lon_ticks = np.arange(np.ceil(extent[0]/5)*5, extent[1], 5, dtype=int)
    lat_ticks = np.arange(np.ceil(extent[2]/3)*3, extent[3], 3, dtype=int)
    for index, (ax, spec) in enumerate(zip(axes.flat, specifications)):
        value, title, units, cmap, norm = spec
        draw_panel(ax, *layers, value, cmap, norm, index >= 3, index % 3 == 0,
                   extent, lon_ticks, lat_ticks)
        left_title(ax, f"({chr(97+index)}) {title}")
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax,
                          orientation="horizontal", fraction=.07, pad=.17)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.set_label(units, fontsize=7)
    save(fig, output)
