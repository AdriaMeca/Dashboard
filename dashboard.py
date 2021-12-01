#Libraries.
from holoviews import opts, dim
from os.path import isfile
from sodapy import Socrata
from time import sleep

import geopandas as gpd
import holoviews as hv
import matplotlib.animation as ani
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import streamlit as st


#Constants and general settings.
st.set_page_config(layout='wide')

hv.extension('matplotlib')
hv.output(fig='svg', size=300)

CMAP_NAME = 'PuRd'
CMAP = cm.get_cmap(CMAP_NAME)

COLORS = [CMAP(int(256*i/3)) for i in range(4)]
C1, C2 = COLORS[2:]

FILENAME = './control_panel.txt'

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
    'Gender of the victim': 'sexe',
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

    result1 = client.get("q2sg-894k", limit=150000)
    result2 = client.get("6rcq-y46b", limit=300)

    cf = pd.DataFrame.from_records(result1)
    df = pd.DataFrame.from_records(result1)
    rf = pd.DataFrame.from_records(result2)

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

    calls = df['any'].value_counts().drop('2021').sort_index()
    calls /= calls.sum()

    cases = rf['any'].value_counts().drop(['2011', '2012', '2021']).sort_index()
    cases /= cases.sum()

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

    return [x_temp, y_temp], [calls.values, cases.values], evolution, df, cf

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
def add_hspace(char, num):
    """
    Function that adds extra horizontal space to align objects in Streamlit.
    """
    return num * char

def add_vspace(num):
    """
    Function that adds extra vertical space to align objects in Streamlit.
    """
    for _ in range(num):
        st.text('')

def get_me_corr(i, data, fig, axs):
    global C1, C2, LIST_YEARS

    axs[0].clear()
    axs[1].clear()

    calls, cases = data

    fig.suptitle('Is there any correlation?', y=0.95)

    axs[0].plot(
        LIST_YEARS[:i],
        calls[:i],
        marker='^',
        label='Normalized number of phone calls',
        color=C1
    )
    axs[0].plot(
        LIST_YEARS[:i],
        cases[:i],
        marker='v',
        label='Normalized number of cases',
        color=C2
    )
    axs[0].set_xlim(2013-0.5, 2020+0.5)
    axs[0].set_ylim(0.11, 0.16)
    axs[0].set_yticks(axs[0].get_yticks())
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Fraction of cases/phone calls')
    axs[0].legend(loc='upper center')

    axs0x = axs[0].secondary_xaxis('top')
    axs0y = axs[0].secondary_yaxis('right')
    axs0x.set_xticklabels('')
    axs0y.set_yticklabels('')

    axs[1].scatter(
        calls[:i],
        cases[:i],
        marker=u'$\u2640$',
        s=300,
        alpha=0.5,
        color=C1
    )
    axs[1].set_xlim(0.110-0.005, 0.160)
    axs[1].set_ylim(0.110, 0.155)
    axs[1].set_yticks(axs[1].get_yticks())
    axs[1].set_xlabel('Normalized number of phone calls')
    axs[1].set_ylabel('Normalized number of cases')

    axs1x = axs[1].secondary_xaxis('top')
    axs1y = axs[1].secondary_yaxis('right')
    axs1x.set_xticklabels('')
    axs1y.set_yticklabels('')

    for a in [*axs, axs0x, axs0y, axs1x, axs1y]:
        a.tick_params(axis='both', direction='in')

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

    ax2 = ax.secondary_xaxis('top')
    ax2.set_xlim(0, period+96)
    ax2.set_xticks(range(0, 97, 12))
    ax2.set_xticklabels('')

    ay2 = ax.secondary_yaxis('right')
    ay2.set_yticklabels('')

    for a in [ax, ax2, ay2]:
        a.tick_params(axis='both', direction='in')

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
    values = series.values / series.sum()
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
    phone_data, calls_cases, evolution, af, cf = load_dataframes()

    #Control panel.
    st.sidebar.header('Table of contents')
    control = st.sidebar.radio(
        label='Select a section',
        options=[
            'Title',
            'Methodology',
            'Results',
            add_hspace('-', 4) + ' Spatial and temporal dependece',
            add_hspace('-', 4) + ' Aggregated spatial evolution',
            add_hspace('-', 4) + ' Analysis of the phone calls',
            add_hspace('-', 4) + ' Violence interconnections',
            add_hspace('-', 4) + ' Is there any correlation?',
            'Conclusions',
            'Critique',
            'Contributions'
        ]
    )

    if not isfile(FILENAME):
        with open(FILENAME, 'w') as outputfile:
            outputfile.write(f'{control}\n')

    with open(FILENAME, 'r') as inputfile:
        for line in inputfile:
            last_page = line.strip()

    if last_page != control:
        with open(FILENAME, 'w') as outputfile:
            outputfile.write(f'{control}\n')
        waiting_room()

    if control == 'Title':
        title()
    elif control == 'Methodology':
        methodology()
    elif control == 'Results':
        results()
    elif control == add_hspace('-', 4) + ' Spatial and temporal dependece':
        results_map(phone_data, af)
    elif control == add_hspace('-', 4) + ' Aggregated spatial evolution':
        results_line(phone_data[1])
    elif control == add_hspace('-', 4) + ' Analysis of the phone calls':
        results_pie(cf)
    elif control == add_hspace('-', 4) + ' Violence interconnections':
        results_chord(af, evolution)
    elif control == add_hspace('-', 4) + ' Is there any correlation?':
        correlation(calls_cases)
    elif control == 'Conclusions':
        conclusions()
    elif control == 'Critique':
        critique()
    elif control == 'Contributions':
        contributions()


#Pages of the dashboard.
def title():
    #Title.
    m = """
        <div style='font-size: 45px; font-weight: bold; text-align: center'>
            MSc Physics of Complex Systems and Biophysics<br>
            <p style='font-size: 30px'>
                Analysis and Visualization of Big Data
            </p>
        </div>
    """
    t = """
        <h1 style='font-size: 60px; text-align: center'>
            Gender violence, a dark shadow of the pandemic
        </h1>
    """
    #Authors.
    n = """
        <div style='display: grid; text-align: center'>
            <div class='grid-item' style='grid-row-start: 1; font-size: 30px'>
                <b>Adrià Meca Montserrat</b><br>
                <p>
                    Git Master<br>
                    Python wizard
                </p>
            </div>
            <div class='grid-item' style='grid-row-start: 1; font-size: 30px'>
                <b>Clara Colet Díaz</b><br>
                <p>
                    Team mediator<br>
                    Pie chart leading expert
                </p>
            </div>
            <div class='grid-item' style='grid-row-start: 1; font-size: 30px'>
                <b>Guillem Güell Paule</b><br>
                <p>
                    Brainstorming manager<br>
                    Complex graphs CEO
                </p>
            </div>
            <div class='grid-item' style='grid-row-start: 1; font-size: 30px'>
                <b>Lior Tetro</b><br>
                <p>
                    Head writer<br>
                    Philosopher
                </p>
            </div>
        </div>
    """
    #Abstract.
    a = """
        <div style='display: grid; text-align: center'>
            <div class='grid-item' style='grid-row-start: 1; font-size: 25px'></div>
            <div class='grid-item' style='grid-row-start: 1; font-size: 25px; text-align: left; width: 1200px'>
                <b>Abstract:</b> Among all the things that SARS-CoV-2 has affected,
                gender violence is probably the most important. Information has been
                exposed regarding the relation between the pandemic spike and a big
                increase in gender-based complaints. However, the lack of scientific
                rigor envelops the whole topic. Here, we shed light on how and why gender
                violence has increased due to Covid-19, and we conclude that prior to the
                pandemic the number of complaints was homogeneous in time and space, and
                that the profile of the victim doesn't change significantly over the years.
                We also study the connections between different types of violence and we
                show if there is any correlation between complaints and real cases. These
                results can have an immediate application in resource distribution of
                victim attention and in the preparation of awareness campaigns.
            </div>
        </div>
    """

    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.image(image="./images/ub-logo.png")
    st.markdown(m, unsafe_allow_html=True)
    st.markdown(t, unsafe_allow_html=True)
    st.markdown(n, unsafe_allow_html=True)
    add_vspace(1)
    st.markdown(a, unsafe_allow_html=True)

def methodology():
    t = """
        <h1 style='font-size: 60px; text-align: center'>
            Methodology
        </h1>
    """
    code = """
        #Libraries.
        from holoviews import opts, dim
        from os.path import isfile
        from sodapy import Socrata
        from time import sleep

        import geopandas as gpd
        import holoviews as hv
        import matplotlib.animation as ani
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import pandas as pd
        import streamlit as st

        client = Socrata("analisi.transparenciacatalunya.cat", None)

        #Database of phone calls (complaints) from victims of gender violence.
        result1 = client.get("q2sg-894k", limit=150000)
        #Database of real cases of gender violence.
        result2 = client.get("6rcq-y46b", limit=300)
    """
    st.markdown(t, unsafe_allow_html=True)
    st.code(code, language='python')

    st.markdown("**Data Management Plan:** [Github repository](https://github.com/AdriaMeca/Dashboard.git).")

def results():
    t = """
        <h1 style='font-size: 200px; text-align: center'>
            Results
        </h1>
    """
    add_vspace(10)
    st.markdown(t, unsafe_allow_html=True)

def results_map(data, df):
    global CMAP_NAME, FIRST_YEAR, LIST_YEARS, MONTH_CONVERSION, NUMBER_MONTHS

    t = """
        <p style='font-size: 45px; text-align: center'>
            <b>Results:</b> Spatial and Temporal dependence
        </p>
    """
    st.markdown(t, unsafe_allow_html=True)

    #DrMeca: interactive craziness.
    st.sidebar.header('Plot widgets')
    chosen_year = st.sidebar.select_slider('Select a year', LIST_YEARS, key='map')
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

    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(range(1, NUMBER_MONTHS+1))
    ax2.set_xticklabels([DICT[m] for m in MONTH_CONVERSION.keys()], rotation=45)
    ax2.set_xticklabels('')

    ay2 = ax.secondary_yaxis('right')
    ay2.set_yticklabels('')

    for a in [ax, ax2, ay2]:
        a.tick_params(axis='both', direction='in')

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
        add_vspace(4)
        st.pyplot(month_fig)
    with col2:
        st.pyplot(map_fig)

def results_line(y_data):
    t = """
        <p style='font-size: 45px; text-align: center'>
            <b>Results:</b> Aggregated spatial evolution
        </p>
    """
    st.markdown(t, unsafe_allow_html=True)

    #Sidebar options.
    st.sidebar.header("Plot widgets")
    line_ani = st.sidebar.checkbox('Animation on')
    period = st.sidebar.selectbox('Period', options=[1, 3, 4, 6])

    #DrMeca: one of my renowned animations.
    ani_fig, ax = plt.subplots()

    _, col, _ = st.columns([1, 3, 1])
    with col:
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

    t = """
        <p style='font-size: 45px; text-align: center'>
            <b>Results:</b> Analysis of the phone calls
        </p>
    """
    st.markdown(t, unsafe_allow_html=True)

    #Sidebar options.
    st.sidebar.header("Plot widgets")
    pie_ani = st.sidebar.checkbox('Animation on')
    topic = st.sidebar.selectbox(
        label='Select a topic',
        options=[
            'Age of the victim',
            'Civil status of the victim',
            'Gender of the victim',
            'Victim-aggressor relationship'
        ]
    )
    year = st.sidebar.select_slider('Select a year', LIST_YEARS, key='pie')

    #DraClara: the awesome pie chart.
    pie_fig, ax = plt.subplots()

    _, col, _ = st.columns([1, 2, 1])
    with col:
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

        if topic == 'Age of the victim':
            t1 = """
                <p style='font-size: 20px'>
                    \u25b6 There is no significant change in the ages of the victims in 2020;
                </p>
            """
            t2 = """
                <p style='font-size: 20px'>
                    \u25b6 There is an important increase in the number of "Unknowns".
                </p>
            """
            st.markdown(t1, unsafe_allow_html=True)
            st.markdown(t2, unsafe_allow_html=True)
        elif topic == 'Civil status of the victim':
            t1 = """
                <p style='font-size: 20px'>
                    \u25b6 There is an increase in domestic cases in 2020 (married and
                    <i>de facto couple</i>), which is compatible with the lockdown.
                </p>
            """
            st.markdown(t1, unsafe_allow_html=True)
        elif topic == 'Gender of the victim':
            t1 = """
                <p style='font-size: 20px'>
                    \u25b6 There is no significant change in the gender of the victims in 2020.
                </p>
            """
            st.markdown(t1, unsafe_allow_html=True)
        else:
            t1 = """
                <p style='font-size: 20px'>
                    \u25b6 There is an increase in domestic cases in 2020 (partner and
                    family relations), which is compatible with the lockdown.
                </p>
            """
            st.markdown(t1, unsafe_allow_html=True)

def results_chord(df, evolution):
    global CMAP, DICT, LIST_YEARS

    t = """
        <p style='font-size: 45px; text-align: center'>
            <b>Results:</b> Violence interconnections
        </p>
    """
    st.markdown(t, unsafe_allow_html=True)

    #DrGuillem: madness as a plot.
    new_options = ['Sexual', 'Economical', 'Physical', 'Psychological']
    st.sidebar.header("Plot widgets")
    types_of_violence = st.sidebar.multiselect(
        label='Select a violence',
        options=new_options,
        default=new_options[:2]
    )
    n = len(types_of_violence)

    #Guillem's requested extra plot.
    extra = st.sidebar.button("Show extra plot")
    if extra:
        os.system("python ./guillem.py")

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
                    labels='Option',
                    title='Connections 2020'
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

            ax2 = ax.secondary_xaxis('top')
            ay2 = ax.secondary_yaxis('right')
            ax2.set_xticklabels('')
            ay2.set_yticklabels('')

            for a in [ax, ax2, ay2]:
                a.tick_params(axis='both', direction='in')

            add_vspace(5)
            st.pyplot(bar_fig)

def correlation(data):
    global C1, C2, LIST_YEARS

    t = """
        <p style='font-size: 45px; text-align: center'>
            <b>Results:</b> Correlation between phone calls and real cases
        </p>
    """
    st.markdown(t, unsafe_allow_html=True)

    #Sidebar options.
    st.sidebar.header("Plot widgets")
    corr_ani = st.sidebar.button('Show results')

    corr_fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    get_me_corr(0, data, corr_fig, axs)
    corr = st.pyplot(corr_fig)
    if corr_ani:
        for i in range(0, len(LIST_YEARS)+1):
            get_me_corr(i, data, corr_fig, axs)
            corr.pyplot(corr_fig)

def conclusions():
    t = """
        <h1 style='font-size: 60px; text-align: center'>
            Conclusions
        </h1>
    """
    a = """
        <div style='display: grid; text-align: center'>
            <div class='grid-item' style='grid-row-start: 1;'></div>
            <div class='grid-item' style='grid-row-start: 1; text-align: left; width: 600px'>
                <p style='font-size: 20px'>
                    \u25b6 Before the pandemic the number of phone calls (gender-based complaints)
                    was homogeneous, but there is an increase due to the confinement;
                </p>
                <p style='font-size: 20px'>
                    \u25b6 The profile of the victim doesn't change significantly over the years,
                    but there is a growth of domestic cases in 2020;
                </p>
                <p style='font-size: 20px'>
                    \u25b6 The most prevalent violence is psychological;
                </p>
                <p style='font-size: 20px'>
                    \u25b6 There is no evident correlation between phone calls and real cases
                    (in Catalonia at least).
                </p>
            </div>
        </div>
    """
    st.markdown(t, unsafe_allow_html=True)
    st.markdown(a, unsafe_allow_html=True)

def critique():
    t = """
        <h1 style='font-size: 60px; text-align: center'>
            Critique
        </h1>
    """
    st.markdown(t, unsafe_allow_html=True)
    add_vspace(8)

    #Sidebar attack.
    st.sidebar.header("Moves")
    attack = st.sidebar.button("Adrià's constructive criticism")

    if attack:
        i = 0
        while True:
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'darkblue', 'violet']
            c = f"""
                <h1 style='font-size: 200px; text-align: center; color: {colors[i]}'>
                    3 credits
                </h1>
            """

            temp = st.empty()
            with temp.container():
                st.markdown(c, unsafe_allow_html=True)
                sleep(0.5)
            temp.empty()

            i += 1
            if i > 6: i = 0

def contributions():
    t = """
        <h1 style='font-size: 60px; text-align: center'>
            Contributions
        </h1>
    """
    st.markdown(t, unsafe_allow_html=True)

    #Sidebar options.
    st.sidebar.header("Thank you!")
    thanks = st.sidebar.button("Press me")

    if isfile(FILENAME):
        os.remove(FILENAME)

    contrib = {
        'Conceptualization': [],
        'Methodology': [],
        'Software': [],
        'Validation': [],
        'Formal analysis': [],
        'Investigation': [],
        'Resources': [],
        'Data curation': [],
        'Writing - Original draft': [],
        'Writing - Review & editing': [],
        'Visualization': [],
        'Supervision': []
    }

    adria = list(len(contrib) * 'x')
    clara = list(len(contrib) * 'x')
    guillem = list(len(contrib) * 'x')
    lior = list(len(contrib) * 'x')

    adria[0], adria[8] = '', ''
    clara[3], clara[11] = '', ''
    guillem[1], guillem[11] = '', ''
    lior[0], lior[9] = '', ''

    for author in [adria, clara, guillem, lior]:
        for j, k in enumerate(contrib.keys()):
            contrib[k].append(author[j])

    #Table of contributions.
    table = pd.DataFrame(contrib, index=['Adrià', 'Clara', 'Guillem', 'Lior'])
    style = table.T.style

    #Style of the table.
    cell_hover = {
        'selector': 'td:hover',
        'props': [('background-color', 'pink')]
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #b7005c; color: white;'
    }
    style.set_table_styles([cell_hover, headers])
    style.set_table_styles([
        {'selector': 'th.col_heading', 'props': 'text-align: center;'},
        {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5em;'},
        {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'}
    ], overwrite=False)

    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.table(style.applymap(lambda _: 'background-color: #f6f2f4'))

    if thanks:
        while True:
            st.balloons()
            sleep(5)

def waiting_room():
    for _ in range(100):
        st.container()
    sleep(2)

#We run the main program.
if __name__ == '__main__':
    main()