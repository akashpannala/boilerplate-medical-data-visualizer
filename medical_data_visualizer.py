import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Add 'overweight' column
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# 3. Normalize 'cholesterol' and 'gluc'
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Draw Categorical Plot
def draw_cat_plot():
    # 5. Melt the DataFrame
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and count
    df_grouped = df_cat.groupby(['cardio', 'variable', 'value']).size().unstack(fill_value=0)

    # 7. Plot using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for i, cardio_value in enumerate([0, 1]):
        df_plot = df_grouped.loc[cardio_value]
        df_plot.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Cardio = {cardio_value}')
        axes[i].set_ylabel('Total')
        axes[i].set_xlabel('Variable')
        axes[i].legend(title='Value')

    fig.tight_layout()

    # 8. Save and return
    fig.savefig('catplot.png')
    return fig

# 9. Draw Heat Map
def draw_heat_map():
    # 10. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 11. Calculate correlation matrix
    corr = df_heat.corr()

    # 12. Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. Plot heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-0.1, vmax=0.3)
    fig.colorbar(cax)

    # 14. Annotate values
    for (i, j), val in np.ndenumerate(corr.values):
        if not mask[i, j]:
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', color='black')

    # 15. Set axis labels
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    fig.tight_layout()

    # 16. Save and return
    fig.savefig('heatmap.png')
    return fig