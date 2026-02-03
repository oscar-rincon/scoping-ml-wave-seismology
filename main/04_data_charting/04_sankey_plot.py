import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "json"

labels = [
    "All studies (44)",
    "Forward problems",
    "Inverse problems",
    "Neural operators",
    "Convolutional",
    "PINNs",
    "Gaussian processes",
    "Recurrent",
    "Generative adversarial",
    "Finite differences",
    "Spectral elements",    
]

# Sankey structure
source = [
    0, 0,
    1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2
]

target = [
    1, 2,
    3, 4, 5, 6, 7, 8, 9, 10,
    3, 4, 5, 6, 7, 8, 9, 10,
]

value = [
    10, 34,                 # All → Forward / Inverse

    # Forward → methods (sum = 10)
    2, 2, 2, 1, 1, 1, 1, 0,

    # Inverse → methods (sum = 34)
    7, 6, 8, 3, 3, 2, 3, 2
]

# Semantic colors
COL_ALL = "#b4bcc4"        # neutral gray
COL_FORWARD = "#1f77b4"    # blue 1f77b4
COL_INVERSE = "#79c7ff"    # orange 79c7ff

COL_ML = "#4c78a8"        # blue
COL_PROB = "#000000"     # purple
COL_CLASSICAL = "#8A8A8A" # gray

node_colors = [
    COL_ALL,
    COL_FORWARD,
    COL_INVERSE,
    COL_ML,
    COL_ML,
    COL_ML,
    COL_PROB,
    COL_ML,
    COL_ML,
    COL_CLASSICAL,
    COL_CLASSICAL
]

link_colors = (
    ["rgba(150,150,150,0.45)"] * 2 +
    ["rgba(31,119,180,0.45)"] * 8 +
    ["rgba(121,199,255,0.45)"] * 8
)

fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        label=labels,
        pad=6,
        thickness=12,
        color=node_colors,
        line=dict(width=0)
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors
    )
))

 
# Layout 
fig.update_layout(
    width=640,
    height=200,
    font=dict(size=11, color="black"),

    # Remove all backgrounds
    paper_bgcolor="white",
    plot_bgcolor="white",

    # Remove axes completely (important for Sankey)
    xaxis=dict(
        visible=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False
    ),
    yaxis=dict(
        visible=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False
    ),

    # Margins (space for legend)
    margin=dict(l=5, r=110, t=5, b=5),

    #Legend on the right, clean
    legend=dict(
        x=0.99,
        y=0.5,
        xanchor="left",
        yanchor="middle",
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=9)
    )
)

fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=10, color=COL_ML, symbol="square"),
    legendgroup="ml",
    showlegend=True,
    name="Machine learning"
))

fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=10, color=COL_CLASSICAL, symbol="square"),
    legendgroup="classical",
    showlegend=True,
    name="Standard numerical"
))

fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=10, color=COL_PROB, symbol="square"),
    legendgroup="prob",
    showlegend=True,
    name="Probabilistic"
))

# Save
fig.write_image(
    "figs/sankey_methods.svg",
    scale=1
)

fig.write_image(
    "figs/sankey_methods.pdf",
    scale=1
)