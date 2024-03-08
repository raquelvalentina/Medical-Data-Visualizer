import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = np.where((df['weight']/(df['height']*0.01)**2) > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = np.where(df['cholesterol'] >1, 1, 0)
df['gluc'] = np.where(df['gluc'] ==1, 0, 1)


# Draw Categorical Plot
def draw_cat_plot():

    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['gluc', 'smoke', 'alco', 'active', 'overweight', 'cholesterol'], var_name='variable', value_name='Grupo')
    cardio_0 = df_cat[df_cat['cardio'] == 0]
    cardio_1 = df_cat[df_cat['cardio'] == 1]

    a_0 = cardio_0.groupby(['Grupo','variable']).size().reset_index(name='0')
    a_1 = cardio_1.groupby(['Grupo','variable']).size().reset_index(name='1')

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.merge(a_0, a_1, on=['Grupo','variable'], suffixes=('_0', '_1'))
    
    figure_negativeP = df_cat[df_cat['Grupo'] == 0].drop(columns=['Grupo'])

    # Filtra los datos para el grupo '1'
    figure_negativeN = df_cat[df_cat['Grupo'] == 1].drop(columns=['Grupo'])

    # Crear la figura y los ejes
    fig, axis = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)

    # Trazar el gráfico de barras para cardio = 0
    figure_negativeP.plot(kind='bar', ax=axis[0], title='cardio = 0', xlabel='variable', ylabel='total', rot=0, legend=None, width=0.8)

    # Trazar el gráfico de barras para cardio = 1
    figure_negativeN.plot(kind='bar', ax=axis[1], title='cardio = 1', xlabel='variable', rot=0, yerr=0, legend=None, width=0.8)

    # Configurar las etiquetas del eje y en el segundo gráfico para que no se muestren
    axis[1].set_yticklabels([])

    # Agregar una leyenda a la figura
    fig.legend(['0', '1'], bbox_to_anchor=(1.05, 0.5), title='Value', frameon=False)

    # Mover la leyenda a la derecha
    sns.move_legend(fig, "upper right")
    axis[0].set_xticklabels(figure_negativeP['variable'])
    axis[1].set_xticklabels(figure_negativeP['variable'])

    # Ocultar los bordes superiores y derechos de los subplots
    for ax in axis:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Ajustar el espaciado entre subplots
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
    df_heat = df[(df['ap_lo']<=df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025))& (df['height'] <= df['height'].quantile(0.975))&(df['weight'] >= df['weight'].quantile(0.025))&(df['weight'] <= df['weight'].quantile(0.975))]
    # Create height correlation matrix using the dataset. Plot the correlation matrix using seaborn's heatmap(). Mask the upper triangle. The chart should look like examples/Figure_2.png.
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax =plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, fmt=".1f", linewidths=0.5, cbar_kws={'shrink': 0.4}, annot=True, center=0, vmax=.24,vmin=-.08, ax=ax)    # Show plot
    plt.show()

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
