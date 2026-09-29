# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_wrangling import read_feature_set
from IPython.display import display
sns.set_theme(style="white")

# %%
featured_filtered, candidate_features = read_feature_set()

target = "Balancing_energy_total"

# %%
for feature_set_name in ["q50", "q95", "ex_post"]:

    features = candidate_features[feature_set_name]

    data = (
        featured_filtered[
            features + [target]
        ]
        .select_dtypes("number")
    )

    # =========================================================================
    # FULL CORRELATION MATRIX
    # =========================================================================

    corr = data.corr()

    plt.figure(
        figsize=(
            max(10, len(corr) * 0.45),
            max(8, len(corr) * 0.45),
        )
    )

    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
    )

    plt.title(
        f"{feature_set_name.upper()} feature correlations"
    )

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # CORRELATION WITH RESPONSE
    # =========================================================================

    corr_target = (
        corr[[target]]
        .sort_values(
            target,
            key=np.abs,
            ascending=False,
        )
    )

    plt.figure(
        figsize=(
            5,
            max(6, len(corr_target) * 0.35),
        )
    )

    sns.heatmap(
        corr_target,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
    )

    plt.title(
        f"{feature_set_name.upper()} vs {target}"
    )

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print(feature_set_name.upper())
    print("=" * 80)

    display(
        corr_target
        .drop(index=target)
        .sort_values(
            target,
            key=np.abs,
            ascending=False,
        )
        .head(25)
    )

# %%
# ============================================================================
# DISTRIBUTION OF TARGET
# ============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4),
)

featured_filtered[target].hist(
    bins=100,
    ax=axes[0],
)

axes[0].set_title(target)

featured_filtered[target].clip(
    upper=featured_filtered[target].quantile(0.99)
).hist(
    bins=100,
    ax=axes[1],
)

axes[1].set_title("Target clipped at p99")

plt.tight_layout()
plt.show()

# %%
# ============================================================================
# COMBINED CORRELATION RANKING
# ============================================================================

all_features = sorted(
    set(
        candidate_features["q50"]
        + candidate_features["q95"]
        + candidate_features["ex_post"]
    )
)

corrs = (
    featured_filtered[
        all_features + [target]
    ]
    .select_dtypes("number")
    .corr()[target]
    .drop(target)
    .sort_values(
        key=np.abs,
        ascending=False,
    )
)

display(corrs.head(50))

# %%
# ============================================================================
# STRONGEST ABSOLUTE CORRELATIONS
# ============================================================================

fig, ax = plt.subplots(
    figsize=(8, 12)
)

corrs.abs().head(25).sort_values().plot.barh(ax=ax)

ax.set_title(
    f"Strongest absolute correlations with {target}"
)

plt.tight_layout()
plt.show()

# %%
# ============================================================================
# SCATTER PLOTS OF STRONGEST DRIVERS
# ============================================================================

top_features = (
    corrs.abs()
    .head(9)
    .index
)

fig, axes = plt.subplots(
    3,
    3,
    figsize=(14, 12),
)

for feature, ax in zip(
    top_features,
    axes.flat,
):

    sample = (
        featured_filtered[
            [feature, target]
        ]
        .dropna()
    )

    if len(sample) > 10_000:
        sample = sample.sample(
            10_000,
            random_state=42,
        )

    ax.scatter(
        sample[feature],
        sample[target],
        alpha=0.2,
        s=5,
    )

    ax.set_title(feature)

plt.tight_layout()
plt.show()

# %%
# ============================================================================
# COMPARE EXTREME EVENTS vs NORMAL EVENTS
# ============================================================================

threshold = (
    featured_filtered[target]
    .quantile(0.95)
)

featured_filtered["is_extreme"] = (
    featured_filtered[target]
    >= threshold
)

# %%
# Focus on q95 candidates first

summary = (
    featured_filtered
    .groupby("is_extreme")[
        candidate_features["q95"]
    ]
    .mean()
    .T
)

summary.columns = [
    "normal",
    "extreme",
]

summary["change_pct"] = (
    100
    * (
        summary["extreme"]
        - summary["normal"]
    )
    / summary["normal"].abs()
)

summary = summary.sort_values(
    "change_pct",
    key=np.abs,
    ascending=False,
)

display(summary.head(30))

# %%
# ============================================================================
# FEATURE DISTRIBUTIONS DURING EXTREME EVENTS
# ============================================================================

top_extreme_features = (
    summary["change_pct"]
    .abs()
    .head(6)
    .index
)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8),
)

for feature, ax in zip(
    top_extreme_features,
    axes.flat,
):

    sns.boxplot(
        data=featured_filtered[
            [feature, "is_extreme"]
        ],
        x="is_extreme",
        y=feature,
        ax=ax,
    )

    ax.set_title(feature)

plt.tight_layout()
plt.show()

# %%
# ============================================================================
# OPTIONAL: FEATURE DISTRIBUTIONS
# ============================================================================

for feature in top_extreme_features:

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    sns.kdeplot(
        data=featured_filtered,
        x=feature,
        hue="is_extreme",
        common_norm=False,
        ax=ax,
    )

    ax.set_title(feature)

    plt.tight_layout()
    plt.show()

# %%
# ============================================================================
# RANDOM FOREST SCREENING
# ============================================================================

from sklearn.ensemble import RandomForestRegressor

features = candidate_features["q95"]

data = (
    featured_filtered[
        features + [target]
    ]
    .dropna()
)

X = data[features]
y = data[target]

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X, y)

importance = (
    pd.Series(
        rf.feature_importances_,
        index=features,
    )
    .sort_values(
        ascending=False,
    )
)

display(importance.head(30))

# %%
fig, ax = plt.subplots(
    figsize=(8, 10)
)

importance.head(25).sort_values().plot.barh(
    ax=ax
)

ax.set_title(
    "Random Forest feature importance screening"
)

plt.tight_layout()
plt.show()