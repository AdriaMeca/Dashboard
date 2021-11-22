from holoviews import opts, dim
from sodapy import Socrata

import geopandas as gpd
import holoviews as hv
import matplotlib.animation as ani
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

hv.extension('matplotlib')
hv.output(fig='svg', size=300)


def get_me_pie(i, df, column, year, labels, fig, ax, title):
    """
    Function that builds the pie chart.
    """
    ax.clear()

    series = df.loc[df['any'] == str(year), column].value_counts().sort_values()

    fs = 12
    if title == 'Victim-aggressor relationship':
        fs = 11
        options = ['Germà/germans', 'Altres familiars', 'Pare', 'Fill/fills']
        temp = {l: 0 for l in labels}
        for k in series.index:
            if k in options:
                temp['Family'] += series[k]
            elif k == 'Exparella':
                temp['Partner'] += series[k]
            elif k == 'Parella':
                temp['Former partner'] += series[k]
        series = pd.Series(temp).sort_values()

    values = series.values / series.values.sum()
    labels = [f'{l} ({v:.1%})' for v, l in zip(values, labels)]

    n = len(labels)
    if n == 2:
        colors = [cmap(int(256/i)) for i in range(2, 0, -1)]
    else:
        colors = [cmap(int(256*i/(n-1))) for i in range(n)]

    explode = [0.002 * i for _ in range(n)]

    patches, texts = ax.pie(x=series.values, colors=colors, explode=explode)
    ax.legend(patches, labels, loc='lower right', framealpha=0.7)

    circle = plt.Circle(xy=(0, 0), radius=0.7, fc='white')
    ax.annotate(title, xy=(0, 0), ha='center', fontsize=fs)

    fig = plt.gcf()
    fig.gca().add_artist(circle)
    fig.tight_layout()

@st.cache(show_spinner=False)
def load_dataframes():
    """
    Function that loads the Dataframes.
    """
    client = Socrata("analisi.transparenciacatalunya.cat", None)
    result = client.get("q2sg-894k", limit=150000)

    af = pd.DataFrame.from_records(result)

    #Special modifications needed for Clara's Dataframe.
    cf = pd.DataFrame.from_records(result)
    cf.loc[
        cf['edat'] == 'Menors de 18 anys', 'edat'
    ] = 'Menor de 18 anys'
    cf.loc[
        cf['edat'] == 'Entre 18 i 30 anys', 'edat'
    ] = 'Entre 18 i 31 anys'
    cf.loc[
        cf['relacioagressor'] == 'Fill / fills', 'relacioagressor'
    ] = 'Fill/fills'
    cf.loc[
        cf['relacioagressor'] == 'Germà / germans', 'relacioagressor'
    ] = 'Germà/germans'

    return af, cf

@st.cache(show_spinner=False)
def load_map(df, number_years):
    """
    Function that builds the map of Catalonia.
    """
    catalonia_map = gpd.read_file("./data/catalonia_map.geojson")
    population = pd.read_excel("./data/population.xls")

    regions = catalonia_map['nom_comar'].values
    number_regions = len(regions)

    number_calls = np.zeros((number_years, number_regions))
    number_citizens = np.zeros((number_years, number_regions))
    for i, year in enumerate(range(first_year, chosen_year+1)):
        regions_column = population.columns[0]

        temp1 = df[df['any'] == str(year)]['comarca'].value_counts()
        temp2 = dict(population[[regions_column, year]].values)
        for j, region in enumerate(regions):
            try:
                number_calls[i, j] = temp1[region]
                number_citizens[i, j] = temp2[region]
            except KeyError:
                number_calls[i, j] = 0.0
                number_citizens[i, j] = 1.0
        catalonia_map[year] = 10**6 * number_calls[i, :] / number_citizens[i, :]
    return catalonia_map

#Constants and general settings.
cmap_name = 'PuRd'
cmap = cm.get_cmap(cmap_name)

first_year = 2013
final_year = 2020
list_years = list(range(first_year, final_year+1))

#We load the Dataframes.
af, cf = load_dataframes()

#
#1.
#

st.title('Title')
st.markdown('This is a test.')

#DrMeca: this is one of my interactive plots.
months = 12
conversion = {
    'Gener': 1,
    'Febrer': 2,
    'Març': 3,
    'Abril': 4,
    'Maig': 5,
    'Juny': 6,
    'Juliol': 7,
    'Agost': 8,
    'Setembre': 9,
    'Octubre': 10,
    'Novembre': 11,
    'Desembre': 12
}
english_names = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December'
]

chosen_year = st.select_slider('Select a year', list_years, key='map')
number_years = (chosen_year-first_year) + 1

month_fig, ax = plt.subplots()

ax.grid(linewidth=0.1)

square = np.zeros(months)
cumulative = np.zeros(months)
for year in range(first_year, chosen_year+1):
    series = af[af['any'] == str(year)]['mes'].value_counts()
    series.index = [conversion[month] for month in series.index]
    series = series.sort_index()
    square += series.values**2
    cumulative += series.values

    ax.plot(series.index, series.values, linewidth=0.5, color='tab:gray')
square /= number_years
cumulative /= number_years

error = np.sqrt((square-cumulative**2)/number_years)

ax.axhline(sum(cumulative)/len(cumulative), linestyle='--', color='tab:orange', zorder=0)
ax.errorbar(range(1, months+1),
            cumulative,
            error,
            elinewidth=2,
            ecolor='tab:gray',
            capsize=5,
            marker='s',
            markersize=8,
            linewidth=3,
            color='tab:blue')

ax.set_xticks(range(1, months+1))
ax.set_xticklabels(english_names, rotation=45)

ax.set_xlabel('Month')
ax.set_ylabel('Number of phone calls')

#DrLior: the famous map of Catalonia.
catalonia_map = load_map(af, number_years)

ax = catalonia_map.plot(column=chosen_year, cmap=cmap_name, vmin=0, vmax=3000)
ax.set_axis_off()

map_fig = ax.get_figure()
cax = map_fig.add_axes([0.8, 0.1, 0.05, 0.8])
sm = cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=0, vmax=3000))
cbar = map_fig.colorbar(sm, cax=cax, spacing='uniform')
cbar.set_label("$10^{6}\\cdot$ (Number of phone calls$\\,/\\,$Number of citizens)")

col1, col2 = st.columns(2)
with col1:
    st.pyplot(month_fig)
with col2:
    st.pyplot(map_fig)

#
#2.
#

#DraClara: the awesome pie chart.
pie_fig, ax = plt.subplots()

titles = {
    'Age of the victim': 'edat',
    'Civil state of the victim': 'estatcivil',
    'Sex of the victim': 'sexe',
    'Victim-aggressor relationship': 'relacioagressor'
}

labels = {
    'Age of the victim': [
        'Unknown',
        '${}<18$ years old',
        '${}>60$ years old',
        '$\in[51,60]$ years old',
        '$\in[18,31]$ years old',
        '$\in[41,50]$ years old',
        '$\in[31,40]$ years old'
    ],
    'Civil state of the victim': [
        'Widow',
        'Single',
        'Separated',
        'De facto couple',
        'Unknown',
        'Divorced',
        'Married'
    ],
    'Sex of the victim': [
        'Men',
        'Women'
    ],
    'Victim-aggressor relationship': [
        'Family',
        'Partner',
        'Former partner'
    ]
}

col1, _, col2 = st.columns([2, 1, 2])
with col1:
    title = st.selectbox('Select a topic', titles.keys())
with col2:
    year = st.select_slider('Select a year', list_years, key='pie')

_, col3, _ = st.columns([1, 3, 1])
with col3:
    column = titles[title]

    get_me_pie(0, cf, column, year, labels[title], pie_fig, ax, title)
    pie = st.pyplot(pie_fig)

    pie_ani = st.sidebar.checkbox('Pie animation on')
    if pie_ani:
        n = 15
        for i in range(1, n+1):
            get_me_pie(i, cf, column, year, labels[title], pie_fig, ax, title)
            pie.pyplot(pie_fig)
        for j in range(n-1, -1, -1):
            get_me_pie(j, cf, column, year, labels[title], pie_fig, ax, title)
            pie.pyplot(pie_fig)

#
#3.
#

#DrGuillem: madness in the form of a chart.
_, col, _ = st.columns([1, 4, 1])
with col:
    old_options = [c for c in af if c.startswith('v_')]
    new_options = ['Physical', 'Psychological', 'Sexual', 'Economical']
    types_of_violence = st.multiselect('Select a violence', options=new_options)

    conversion = {n: o for o, n in zip(old_options, new_options)}

    categories = []
    for v in types_of_violence:
        categories.append(v)
        categories.append(f'No {v}')

    n = len(types_of_violence)
    if n >= 2:
        gf = pd.DataFrame(columns=('s', 't', 'v'))
        for i in range(n):
            for j in range(i+1, n):
                n1, n2 = types_of_violence[i], types_of_violence[j]
                o1, o2 = conversion[n1], conversion[n2]

                temp = af[af['any'] == '2020'][[o1, o2]]
                temp[o1] = temp[o1].replace({'': n1, 'Sí': n1, 'No': f'No {n1}'})
                temp[o2] = temp[o2].replace({'': n2, 'Sí': n2, 'No': f'No {n2}'})
                temp = temp.rename(columns=dict([(n1, 's'), (n2, 't')])).value_counts()

                for k, v in zip(temp.index, temp.values):
                    gf = gf.append({'s': k[0], 't': k[1], 'v': float(v)}, ignore_index=True)

        categories_df = hv.Dataset(pd.DataFrame(categories, columns=['Option']))

        chord = hv.Chord((gf, categories_df))
        chord.opts(
            opts.Chord(
                cmap=cmap_name,
                edge_cmap=cmap_name,
                edge_color=dim('t').str(),
                node_color='Option',
                labels='Option'
            )
        )
        chord_fig = hv.render(chord, backend='matplotlib')
        st.pyplot(chord_fig)
