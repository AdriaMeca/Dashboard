#Guillem's request.
from sodapy import Socrata

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


dct = {
    "v_fisica": "physical",
    "v_psicologica": "psychological",
    "v_sexual": "sexual",
    "v_economica": "economical"
}

client = Socrata("analisi.transparenciacatalunya.cat", None)
results = client.get("q2sg-894k", limit=150000)

results_df = pd.DataFrame.from_records(results)
results2020_df = results_df[results_df["any"] == "2020"]

old_options = [v for v in results_df if v.startswith('v_')]

group2020 = results2020_df.groupby(old_options).size().reset_index(name="Freq")
group2020["%"] = 100 * group2020.Freq / group2020.Freq.sum()

agrupades2020 = {"Quantity of violence types": [str(i) for i in range(5)]}
for i in range(5):
    for _, row in group2020.iterrows():
        if list(row.values[:4]).count("Sí") == i:
            key = '-'.join([dct[idx] for idx in row.index if row[idx] == "Sí"])
            if not key:
                key = "none"
            elif key.count('-') == 3:
                key = "all"
            agrupades2020[key] = [0 for _ in range(i)] + [row.values[4]]
    if i < 4:
        for e in list(agrupades2020.keys())[1:]:
            agrupades2020[e].append(0)

fig = px.bar(
    agrupades2020,
    x="Quantity of violence types",
    y=list(agrupades2020.keys())
)

fig.update_layout(yaxis_title="# people")
fig.update_layout(xaxis_title="",legend_title="Violence")
fig.update_layout(title="Grouped violence types (2020)")
fig.update_xaxes(visible=False, showticklabels=False)
fig.show()