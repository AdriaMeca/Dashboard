from holoviews import opts, dim
from sodapy import Socrata
from time import sleep

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

def get_me_line(i, y_data, period, fig, ax):
    """
    Function that builds the line chart.
    """
    global C1, C2

    ax.clear()

    ylim = {
        1: (668.35, 1892.65),
        3: (2319.3, 5414.7),
        4: (3185.55, 6187.45),
        6: (5116.6, 8073.4)
    }

    x_temp, y_temp = [], []
    for j in range(0, i, period):
        x_temp.append(j+period)
        y_temp.append(sum(y_data[j:j+period]))

    ax.plot(x_temp, y_temp, marker='o', color=C1)

    ax.set_xlim(0, period+96)
    ax.set_xticks(range(0, 97, 12))
    ax.set_xlabel('Time (months)')

    ymin, ymax = ylim[period]
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Number of phone calls')

    if i == 0:
        ax.text(6, ymax, f'2013', ha='center', va='bottom', color='white')

    counter, correction = 0, 0
    for j in range(0, i, period):
        if j % 12 == 0:
            if j > 0:
                ax.axvline(j, linestyle='--', lw=0.3, color=C1, zorder=0)
            if counter == 7:
                correction += period / 2
            ax.text(j+correction+6, ymax, f'{counter+2013}', ha='center', va='bottom')
            counter += 1

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
    global MONTH_CONVERSION

    client = Socrata("analisi.transparenciacatalunya.cat", None)
    result = client.get("q2sg-894k", limit=150000)

    cf = pd.DataFrame.from_records(result)
    df = pd.DataFrame.from_records(result)

    #Modifications needed for Clara's Dataframe.
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

    #Special treatment of my data.
    x_temp, y_temp = [], []
    for idx, year in enumerate(range(2013, 2021)):
        series = df[df['any'] == str(year)]['mes'].value_counts()
        series.index = [MONTH_CONVERSION[month] for month in series.index]
        series = series.sort_index()

        x_temp += list(series.index + 12*idx)
        y_temp += list(series.values)

    return [x_temp, y_temp], df, cf

@st.cache(show_spinner=False)
def load_map(df, first_year, chosen_year):
    """
    Function that builds the map of Catalonia.
    """
    catalonia_map = gpd.read_file("./data/catalonia_map.geojson")
    population = pd.read_excel("./data/population.xls")

    regions = catalonia_map['nom_comar'].values
    number_regions = len(regions)

    number_years = (chosen_year-first_year) + 1
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

COLORS = [cmap(int(256*i/3)) for i in range(4)]
C1, C2 = COLORS[2:]

first_year = 2013
final_year = 2020
list_years = list(range(first_year, final_year+1))

MONTH_CONVERSION = {
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

#We load the Dataframes.
phone_data, af, cf = load_dataframes()

#Sidebar options.
st.sidebar.header("Animation's control panel")
line_ani = st.sidebar.checkbox('Line animation on')
pie_ani = st.sidebar.checkbox('Pie animation on')

#Title.
st.title('Title')
st.markdown('This is a test.')
#
#1. DrMeca: this is one of my interactive plots.
#
months = 12
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

x_data, y_data = phone_data

square = np.zeros(months)
cumulative = np.zeros(months)
for i in range(number_years):
    phone_calls = np.array(y_data[12*i:12*(i+1)])

    square += phone_calls**2
    cumulative += phone_calls

    ax.plot(range(1, months+1), phone_calls, lw=0.3, color=C1)

square /= number_years
cumulative /= number_years

error = np.sqrt((square-cumulative**2)/number_years)

ax.axhline(sum(cumulative)/len(cumulative), linestyle='--', color=C2, zorder=0)

ax.fill_between(
    range(1, months+1),
    cumulative-error,
    cumulative+error,
    color=C1,
    alpha=0.2,
    zorder=-1
)

ax.errorbar(
    range(1, months+1),
    cumulative,
    error,
    elinewidth=2,
    capsize=5,
    marker='s',
    markersize=8,
    linewidth=3,
    color=C1
)

ax.set_xticks(range(1, months+1))
ax.set_xticklabels(english_names, rotation=45)

ax.set_xlabel('Month')
ax.set_ylabel('Number of phone calls')
#
#1.5 DrLior: the famous map of Catalonia.
#
catalonia_map = load_map(af, first_year, chosen_year)

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
#2. DrMeca: unparalleled animation.
#
ani_fig, ax = plt.subplots()

col1, col2 = st.columns([1, 3])
with col1:
    period = st.selectbox('Period', options=[1, 3, 4, 6])
with col2:
    get_me_line(0, y_data, period, ani_fig, ax)
    line = st.pyplot(ani_fig)

    if line_ani:
        sleep(0.5)
        for i in range(0, len(y_data)+1):
            get_me_line(i, y_data, period, ani_fig, ax)
            line.pyplot(ani_fig)
    else:
        get_me_line(len(y_data), y_data, period, ani_fig, ax)
        line.pyplot(ani_fig)
#
#3. DraClara: the awesome pie chart.
#
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

    if pie_ani:
        n = 15
        for i in range(1, n+1):
            get_me_pie(i, cf, column, year, labels[title], pie_fig, ax, title)
            pie.pyplot(pie_fig)
        for j in range(n-1, -1, -1):
            get_me_pie(j, cf, column, year, labels[title], pie_fig, ax, title)
            pie.pyplot(pie_fig)
#
#4. DrGuillem: madness in the form of a chart.
#
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
