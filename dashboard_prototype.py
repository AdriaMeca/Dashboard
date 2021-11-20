from sodapy import Socrata

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

@st.cache(show_spinner=False)
def load_calls_dataframe():
    client = Socrata("analisi.transparenciacatalunya.cat", None)
    result = client.get("q2sg-894k", limit=150000)
    return pd.DataFrame.from_records(result)

@st.cache(show_spinner=False)
def load_map():
    """
    Logical structure that builds the map of Catalonia.
    """
    catalonia_map = gpd.read_file("./data/catalonia_map.geojson")
    population = pd.read_excel("./data/population.xls")

    regions = catalonia_map['nom_comar'].values
    number_regions = len(regions)

    number_calls = np.zeros((number_years, number_regions))
    number_citizens = np.zeros((number_years, number_regions))
    for i, year in enumerate(range(first_year, final_year+1)):
        regions_column = population.columns[0]

        temp1 = phone_calls[phone_calls['any'] == str(year)]['comarca'].value_counts()
        temp2 = dict(population[[regions_column, year]].values)
        for j, region in enumerate(regions):
            try:
                number_calls[i, j] = temp1[region]
                number_citizens[i, j] = temp2[region]
            except KeyError:
                number_calls[i, j] = 0.0
                number_citizens[i, j] = 0.0
        catalonia_map[year] = 10**6 * number_calls[i, :] / number_citizens[i, :]
    return catalonia_map

#Constants and general settings.
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

first_year = 2013


phone_calls = load_calls_dataframe()

#
#1.
#

st.title('Title')
st.markdown('This is a test.')

#DrMeca: this is one of my interactive plots.
final_year = st.slider('Chosen year', min_value=2013, max_value=2020, step=1)
number_years = (final_year-first_year) + 1

monthly_figure, ax = plt.subplots()

ax.grid(linewidth=0.1)

square = np.zeros(months)
cumulative = np.zeros(months)
for year in range(first_year, final_year+1):
    series = phone_calls[phone_calls['any'] == str(year)]['mes'].value_counts()
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
catalonia_map = load_map()

ax = catalonia_map.plot(column=final_year, cmap='PuRd', vmin=0, vmax=3000)
ax.set_axis_off()

map_figure = ax.get_figure()
cax = map_figure.add_axes([0.8, 0.1, 0.05, 0.8])
sm = plt.cm.ScalarMappable(cmap='PuRd', norm=plt.Normalize(vmin=0, vmax=3000))
cbar = map_figure.colorbar(sm, cax=cax, spacing='uniform')
cbar.set_label("$10^{6}\\cdot$ (Number of phone calls$\\,/\\,$Number of citizens)")

col1, col2 = st.columns(2)
with col1:
    st.pyplot(monthly_figure)
with col2:
    st.pyplot(map_figure)
