"""Reproducible stakeholder charts and model-versus-baseline evaluation."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)


def create_report_plots(report, audit, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Criterion: explicitly contrast predictive model with a simple baseline.
    names = list(report["metrics"])
    vals = [report["metrics"][n]["roc_auc"] or 0 for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), vals, color=["#9BA5B1", "#426C99", "#179A87"])
    ax.set_xticks(range(len(names)), [n.replace("_", " ").title() for n in names]);ax.set_ylim(0, 1.07)
    ax.set_ylabel("Holdout ROC-AUC");ax.set_title("Model comparison: predictive lift over baseline", fontsize=14, fontweight="bold")
    _style(ax)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+.025, f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    suffix = "SYNTHETIC DEMO DATA: NOT REAL AAVAIL METRICS" if report["data_mode"]=="demo" else "Holdout evaluation on the provided AAVAIL dataset"
    fig.text(.08,.01,suffix,fontsize=9,color="#667085")
    fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(folder/"model_vs_baseline.png",dpi=160);plt.close(fig)
    markets = report["market_summary"]
    fig,ax=plt.subplots(figsize=(8,5))
    labels=[m["country"].replace("_"," ").title() for m in markets]
    rates=[100*m["churn_rate"] for m in markets]
    bars=ax.bar(labels,rates,color=["#426C99","#DF7757"][:len(labels)])
    ax.set_ylabel("Observed churn rate (%)");ax.set_title("Customer churn by market",fontweight="bold",fontsize=14)
    ax.set_ylim(0,max(rates+[10])*1.25);_style(ax)
    for b,r,m in zip(bars,rates,markets):ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{r:.1f}%  (n={m['n_customers']})",ha="center",fontsize=10)
    fig.text(.08,.01,suffix,fontsize=9,color="#667085")
    fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(folder/"market_churn.png",dpi=160);plt.close(fig)
    if audit:
        keys=[k for k,v in audit["missing_after_validation_before_imputation"].items() if v>0]
        if not keys:keys=list(audit["missing_after_validation_before_imputation"])
        before=[audit["missing_after_validation_before_imputation"][k] for k in keys]
        after=[audit["missing_after_imputation"][k] for k in keys]
        fig,ax=plt.subplots(figsize=(8,5));x=range(len(keys))
        ax.bar([i-.18 for i in x],before,width=.36,label="Before imputation",color="#DA784D")
        ax.bar([i+.18 for i in x],after,width=.36,label="After imputation",color="#1C9A87")
        ax.set_xticks(list(x),[k.replace("_"," ") for k in keys]);ax.set_ylabel("Number of missing entries")
        ax.set_title("Data quality: missing values before and after ingestion",fontweight="bold",fontsize=13)
        ax.legend();_style(ax);fig.text(.08,.01,suffix,fontsize=9,color="#667085")
        fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(folder/"missing_values.png",dpi=160);plt.close(fig)


def create_driver_plots(frame, output_dir, *, data_mode):
    """EDA context: plan, engagement, and age associations by market.

    These are descriptive associations, not identified causal drivers.
    """
    import numpy as np
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    markets=[c for c in ["united_states","singapore"] if c in set(frame["country"])]
    if not markets:return
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    groups=["aavail_basic","aavail_premium","aavail_unlimited"]
    bins=["<10", "10-19", "20+"]
    def engagement(n):
        return "<10" if n<10 else "10-19" if n<20 else "20+"
    age_bands=["<25", "25-39", "40+"]
    def age_band(n):
        return "<25" if n<25 else "25-39" if n<40 else "40+"
    parts=[("subscriber_type",groups,"Subscription plan"),
           ("stream_band",bins,"Streams consumed"),
           ("age_band",age_bands,"Customer age (years)")]
    f=frame.copy();f["stream_band"]=f["num_streams"].map(engagement);f["age_band"]=f["age"].map(age_band)
    palette=["#406D9E","#D97950"]
    for ax,(field,categories,title) in zip(axes,parts):
        x=np.arange(len(categories));w=.36
        for i,country in enumerate(markets):
            gg=f[f["country"]==country].groupby(field)["churn"].agg(["mean","size"])
            means=[100*float(gg.loc[v,"mean"]) if v in gg.index else 0 for v in categories]
            counts=[int(gg.loc[v,"size"]) if v in gg.index else 0 for v in categories]
            bars=ax.bar(x+(i-.5)*w,means,width=w,color=palette[i],label=country.replace("_"," ").title())
            for bar,val,count in zip(bars,means,counts):
                ax.text(bar.get_x()+bar.get_width()/2,val+1, f"{val:.0f}%\nn={count}", ha="center",fontsize=8)
        ax.set_xticks(x,[v.replace("aavail_", "").title() for v in categories]);ax.set_title(title,fontweight="bold")
        ax.set_ylabel("Observed churn rate (%)");ax.set_ylim(0,100);_style(ax)
    axes[0].legend(loc="upper right")
    fig.suptitle("Which customer segments are associated with churn?",fontsize=17,fontweight="bold",y=1.02)
    suffix="SYNTHETIC DEMO DATA - DO NOT PRESENT AS REAL CUSTOMER FINDINGS" if data_mode=="demo" else "Real source dataset; descriptive associations are not causal estimates"
    fig.text(.04,.01,suffix,fontsize=10,color="#667085")
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(output_dir/"eda_churn_factors.png",dpi=160,bbox_inches="tight");plt.close(fig)
