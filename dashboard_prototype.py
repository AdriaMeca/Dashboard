from sodapy import Socrata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


#This is super important: it allows to load the DataFrame once instead of every
#time we change the widgets.
@st.cache(show_spinner=False)
def load_dataframe():
    client = Socrata("analisi.transparenciacatalunya.cat", None)
    result = client.get("q2sg-894k", limit=150000)
    return pd.DataFrame.from_records(result)


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

#We load the DataFrame from cache to avoid long waiting times.
data = load_dataframe()

#final_year is a Streamlit widget, a slider.
first_year = 2013
final_year = st.slider('Final year', min_value=2013, max_value=2021, step=1)
years = (final_year-first_year) + 1

#For this to work, we need to create a figure and axes. Then, instead of using
#plt (e.g. plt.plot(...)) we utilize ax (e.g. ax.plot(...)).
fig, ax = plt.subplots()

ax.grid(linewidth=0.1)

square = np.zeros(months)
cumulative = np.zeros(months)
for year in range(first_year, final_year+1):
    series = data[data['any'] == str(year)]['mes'].value_counts()
    #I convert the name of each month to its associated value to sort them.
    series.index = [conversion[month] for month in series.index]
    series = series.sort_index()
    try:
        square += series.values**2
        cumulative += series.values
    #I do this because this year's data covers only 8 months,
    #so I have to extend the array manually.
    except ValueError:
        square += list(series.values**2) + [0, 0]
        cumulative += list(series.values) + [0, 0]

    ax.plot(series.index, series.values, linewidth=0.5, color='tab:gray')
square /= years
cumulative /= years

error = np.sqrt((square-cumulative**2)/years)

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

#Finally, we feed our figure to Streamlit.
st.pyplot(fig)
