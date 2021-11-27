#Libraries.
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


#Constants and general settings.
hv.extension('matplotlib')
hv.output(fig='svg', size=300)

CMAP_NAME = 'PuRd'
CMAP = cm.get_cmap(CMAP_NAME)

COLORS = [CMAP(int(256*i/3)) for i in range(4)]
C1, C2 = COLORS[2:]

FIRST_YEAR = 2013
FINAL_YEAR = 2020
LIST_YEARS = list(range(FIRST_YEAR, FINAL_YEAR+1))

NUMBER_MONTHS = 12
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

DICT = {
    'Abril': 'April',
    'Age of the victim': 'edat',
    'Agost': 'August',
    'Altres': 'Other',
    'Altres familiars': 'Other relatives',
    'Casada': 'Married',
    'Civil status of the victim': 'estatcivil',
    'Dona': 'Women',
    'Desembre': 'December',
    'Divorciada': 'Divorced',
    'Exparella': 'Former partner',
    'Febrer': 'February',
    'Fill/fills': 'Son',
    'Gener': 'January',
    'Germà/germans': 'Brother',
    'Home': 'Men',
    'Juliol': 'July',
    'Juny': 'June',
    'Economical': 'v_economica',
    'Entre 18 i 31 anys': "$\\in[18,31)$ years old",
    'Entre 31 i 40 anys': "$\\in[31,40]$ years old",
    'Entre 41 i 50 anys': "$\\in[41,51]$ years old",
    'Entre 51 i 60 anys': "$\\in[51,60]$ years old",
    'Maig': 'May',
    'Març': 'March',
    'Menor de 18 anys': "${}<18$ years old",
    'Més de 60 anys': "${}>60$ years old",
    'No consta': 'Unknown',
    'Novembre': 'November',
    'Octubre': 'October',
    'Pare': 'Father',
    'Parella': 'Partner',
    'Parella de fet': 'De facto couple',
    'Psychological': 'v_psicologica',
    'Physical': 'v_fisica',
    'Separada': 'Separated',
    'Setembre': 'September',
    'Sex of the victim': 'sexe',
    'Sexual': 'v_sexual',
    'Soltera': 'Single',
    'Victim-aggressor relationship': 'relacioagressor',
    'Vídua': 'Widow'
}


#Loading functions.
@st.cache(show_spinner=False)
def load_dataframes():
    """
    Function that loads the Dataframes.
    """
    global LIST_YEARS, MONTH_CONVERSION

    client = Socrata("analisi.transparenciacatalunya.cat", None)
    result = client.get("q2sg-894k", limit=150000)

    cf = pd.DataFrame.from_records(result)
    df = pd.DataFrame.from_records(result)

    #Modifications needed for Clara's data.
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

    #Special treatment of Guillem's data.
    old_options = [c for c in df if c.startswith('v_')]
    cnv = dict([(DICT[w], w) for w in DICT if DICT[w] in old_options])
    evolution = dict()
    for year in LIST_YEARS:
        if year not in evolution:
            evolution[year] = dict()
        for violence in old_options:
            number = df[df['any'] == str(year)][violence].value_counts()['Sí']
            if violence not in evolution[year]:
                evolution[year][cnv[violence]] = number

    return [x_temp, y_temp], evolution, df, cf

@st.cache(show_spinner=False)
def load_map(df, chosen_year):
    """
    Function that builds the map of Catalonia.
    """
    global FIRST_YEAR

    catalonia_map = gpd.read_file("./data/catalonia_map.geojson")
    population = pd.read_excel("./data/population.xls")

    regions = catalonia_map['nom_comar'].values
    number_regions = len(regions)

    number_years = (chosen_year-FIRST_YEAR) + 1
    number_calls = np.zeros((number_years, number_regions))
    number_citizens = np.zeros((number_years, number_regions))
    for i, year in enumerate(range(FIRST_YEAR, chosen_year+1)):
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


#Logical functions.
def add_vspace(num):
    """
    Function that adds extra vertical space to align objects in Streamlit.
    """
    for _ in range(num):
        st.text('')

def get_me_line(i, y_data, period, fig, ax):
    """
    Function that builds the animated line chart.
    """
    global C1, C2

    ax.clear()

    #I have to set the limits of the y-axis manually.
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
        ax.text(6, ymax, '2013', ha='center', va='bottom', color='white')

    counter, correction = 0, 0
    for j in range(0, i, period):
        if j % 12 == 0:
            if j > 0:
                ax.axvline(j, linestyle='--', lw=0.3, color=C1, zorder=0)
            if counter == 7:
                correction += period / 2
            ax.text(j+correction+6, ymax, f'{counter+2013}', ha='center', va='bottom')
            counter += 1

def get_me_pie(i, df, topic, year, fig, ax):
    """
    Function that builds the animated pie chart.
    """
    global C1, C2, CMAP, DICT

    ax.clear()

    series = df.loc[df['any'] == str(year), DICT[topic]].value_counts().sort_values()
    values = series.values / series.values.sum()
    labels = [f'{DICT[l]} ({v:.1%})' for v, l in zip(values, series.index)]

    n = len(labels)
    colors = [C1, C2] if n == 2 else [CMAP(int(256*i/(n-1))) for i in range(n)]
    explode = [i/1000 for _ in range(n)]

    patches, texts = ax.pie(
        x=series.values,
        colors=colors,
        explode=explode,
        startangle=i/3
    )

    ax.legend(patches, labels, loc='lower right', framealpha=0.7)

    circle = plt.Circle(xy=(0, 0), radius=0.7, fc='white')

    fontsize = 11 if topic.startswith('V') else 12
    ax.annotate(topic, xy=(0, 0), ha='center', fontsize=fontsize)

    fig = plt.gcf()
    fig.gca().add_artist(circle)
    fig.tight_layout()


#Control panel that chooses the page we see in the dashboard.
def main():
    #We load the Dataframes.
    phone_data, evolution, af, cf = load_dataframes()

    #Control panel.
    st.sidebar.header('Table of contents')
    control = st.sidebar.radio(
        label='Select a section',
        options=[
            'Title',
            'Results map',
            'Results line',
            'Results pie',
            'Results chord',
            'Conclusions'
        ]
    )

    if control == 'Title':
        title()
    elif control == 'Results map':
        results_map(phone_data, af)
    elif control == 'Results line':
        results_line(phone_data[1])
    elif control == 'Results pie':
        results_pie(cf)
    elif control == 'Results chord':
        results_chord(af, evolution)
    elif control == 'Conclusions':
        pass


#Pages of the dashboard.
def title():
    st.title('Title')
    st.markdown('This is a test.')

def results_map(data, df):
    global CMAP_NAME, FIRST_YEAR, LIST_YEARS, MONTH_CONVERSION, NUMBER_MONTHS

    #DrMeca: interactive craziness.
    chosen_year = st.select_slider('Select a year', LIST_YEARS, key='map')
    number_years = (chosen_year-FIRST_YEAR) + 1

    month_fig, ax = plt.subplots()

    ax.grid(linewidth=0.1)

    x_data, y_data = data

    square = np.zeros(NUMBER_MONTHS)
    cumulative = np.zeros(NUMBER_MONTHS)
    for i in range(number_years):
        phone_calls = np.array(y_data[12*i:12*(i+1)])

        square += phone_calls**2
        cumulative += phone_calls

        ax.plot(range(1, NUMBER_MONTHS+1), phone_calls, lw=0.3, color=C1)

    square /= number_years
    cumulative /= number_years

    error = np.sqrt((square-cumulative**2)/number_years)

    ax.axhline(sum(cumulative)/len(cumulative), linestyle='--', color=C2, zorder=0)

    ax.fill_between(
        range(1, NUMBER_MONTHS+1),
        cumulative-error,
        cumulative+error,
        color=C1,
        alpha=0.2,
        zorder=-1
    )

    ax.errorbar(
        range(1, NUMBER_MONTHS+1),
        cumulative,
        error,
        elinewidth=2,
        capsize=5,
        marker='s',
        markersize=8,
        linewidth=3,
        color=C1
    )

    ax.set_xticks(range(1, NUMBER_MONTHS+1))
    ax.set_xticklabels([DICT[m] for m in MONTH_CONVERSION.keys()], rotation=45)

    ax.set_xlabel('Month')
    ax.set_ylabel('Number of phone calls')

    #DrLior: the famous map of Catalonia.
    catalonia_map = load_map(df, chosen_year)

    ax = catalonia_map.plot(column=chosen_year, cmap=CMAP_NAME, vmin=0, vmax=3000)
    ax.set_axis_off()

    map_fig = ax.get_figure()
    cax = map_fig.add_axes([0.8, 0.1, 0.05, 0.8])
    sm = cm.ScalarMappable(cmap=CMAP_NAME, norm=plt.Normalize(vmin=0, vmax=3000))
    cbar = map_fig.colorbar(sm, cax=cax, spacing='uniform')
    cbar.set_label("$10^{6}\\cdot$ (Number of phone calls$\\,/\\,$Number of citizens)")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(month_fig)
    with col2:
        st.pyplot(map_fig)

def results_line(y_data):
    #Sidebar options.
    st.sidebar.header("Animation's control panel")
    line_ani = st.sidebar.checkbox('Line animation on')

    #DrMeca: one of my renowned animations.
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

def results_pie(df):
    global LIST_YEARS

    #Sidebar options.
    st.sidebar.header("Animation's control panel")
    pie_ani = st.sidebar.checkbox('Pie animation on')

    #DraClara: the awesome pie chart.
    pie_fig, ax = plt.subplots()

    col1, _, col2 = st.columns([2, 1, 2])
    with col1:
        topic = st.selectbox(
            label='Select a topic',
            options=[
                'Age of the victim',
                'Civil status of the victim',
                'Sex of the victim',
                'Victim-aggressor relationship'
            ]
        )
    with col2:
        year = st.select_slider('Select a year', LIST_YEARS, key='pie')

    _, col3, _ = st.columns([1, 3, 1])
    with col3:
        get_me_pie(0, df, topic, year, pie_fig, ax)
        pie = st.pyplot(pie_fig)

        if pie_ani:
            n = 15
            for i in range(1, n+1):
                get_me_pie(i, df, topic, year, pie_fig, ax)
                pie.pyplot(pie_fig)
            for j in range(n-1, -1, -1):
                get_me_pie(j, df, topic, year, pie_fig, ax)
                pie.pyplot(pie_fig)

def results_chord(df, evolution):
    global CMAP, DICT, LIST_YEARS

    #DrGuillem: madness as a plot.
    new_options = ['Sexual', 'Economical', 'Physical', 'Psychological']
    types_of_violence = st.multiselect(
        label='Select a violence',
        options=new_options,
        default=new_options[:2]
    )
    n = len(types_of_violence)

    colors = dict()
    for v, c in zip(new_options, [CMAP(int(256*i/3)) for i in range(4)]):
        colors[v] = c
    colors['Sexual'] = CMAP(50)

    col1, col2 = st.columns(2)
    with col1:
        if n >= 2:
            categories = []
            for v in types_of_violence:
                categories.append(v)
                categories.append(f'No {v}')

            gf = pd.DataFrame(columns=('s', 't', 'v'))
            for i in range(n):
                for j in range(i+1, n):
                    n1, n2 = types_of_violence[i], types_of_violence[j]
                    o1, o2 = DICT[n1], DICT[n2]

                    temp = df[df['any'] == '2020'][[o1, o2]]
                    temp[o1] = temp[o1].replace({'': n1, 'Sí': n1, 'No': f'No {n1}'})
                    temp[o2] = temp[o2].replace({'': n2, 'Sí': n2, 'No': f'No {n2}'})
                    temp = temp.rename(columns=dict([(n1, 's'), (n2, 't')])).value_counts()

                    for k, v in zip(temp.index, temp.values):
                        gf = gf.append({'s': k[0], 't': k[1], 'v': float(v)}, ignore_index=True)

            categories_df = hv.Dataset(pd.DataFrame(categories, columns=['Option']))

            chord = hv.Chord((gf, categories_df))
            chord.opts(
                opts.Chord(
                    cmap=CMAP_NAME,
                    edge_cmap=CMAP_NAME,
                    edge_color=dim('t').str(),
                    node_color='Option',
                    labels='Option'
                )
            )
            chord_fig = hv.render(chord, backend='matplotlib')
            st.pyplot(chord_fig)
    with col2:
        if n >= 2:
            bar_fig, ax = plt.subplots()

            width = 0.2
            shift = (-width) * (len(types_of_violence)-1)/2
            for idx, violence in enumerate(types_of_violence):
                ax.bar(
                    [year+shift for year in LIST_YEARS],
                    [evolution[year][violence] for year in LIST_YEARS],
                    width=width,
                    label=violence,
                    color=colors[violence]
                )

                shift += width

            plt.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.1),
                ncol=len(types_of_violence)
            )

            plt.xlabel('Year')
            plt.ylabel('Number of cases')

            add_vspace(5)
            st.pyplot(bar_fig)

def conclusion():
    pass


#We run the main program.
if __name__ == '__main__':
    main()