import matplotlib.pyplot as plt 
from matplotlib.patches import ConnectionPatch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.graph_objects as go
import circlify
import networkx as nx
import numpy as np 
import pandas as pd
from itertools import combinations
from collections import Counter
from typing import List, Iterable

def sankeyBool(df: pd.DataFrame, grouping_col: str, display_col: str,
                false_val: str, true_val: str, display_col_vals: Iterable,
                colors_bool: List[str],
                colors_display_col_vals: List[str]) -> go.Figure:
    
    """
    Create a plotly Sankey diagram showing flows from a like-boolean 
    grouping column (ex. 'Yes', 'No') to categories in a display column.
    The function groups `df` by `grouping_col` and counts occurrences of each
    value of `display_col` per boolean value, then renders these counts as a
    Sankey diagram where the left nodes are the boolean labels and the right
    nodes are unique values from `display_col`.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least `grouping_col` and `display_col`.
    grouping_col : str
        Column name in `df` that contains like-boolean values 
        (any values that can be interpreted as boolean). Values are mapped to
        `false_val` and `true_val` for display.
    display_col : str
        Column name in `df` containing the categorical target values to flow to.
    false_val : str
        Label to use for the "False" source node. (Example: "Day off")
    true_val : str
        Label to use for the "True" source node. (Example: "Working day")
    display_col_vals : Iterable
        Iterable (e.g., list, tuple, numpy.array) of expected unique values for
        `display_col`.
        Used to map category to color. Order should correspond to
        `colors_display_col_vals`.
    colors_bool : list[str]
        Two-element list of colors for the boolean source nodes:
        `[color_for_false, color_for_true]`. Colors can be CSS color names,
        hex strings, or rgba strings (e.g., "rgba(10,10,10,1)").
    colors_display_col_vals : list[str]
        List of colors for the display column categories. Length must match the
        length of `display_col_vals`. Colors should be provided in the same
        order as `display_col_vals`.
    
    Returns
    -------
    plotly.graph_objects.Figure
        A plotly Figure containing a Sankey diagram. You can display it with
        `fig.show()` or embed it in dashboards.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import plotly.graph_objects as go
    >>> df = pd.DataFrame({
    ...     "is_paid": ["Paid", "Paid", "Not paid", "Paid", "Not paid"],
    ...     "plan": ["A", "A", "B", "B", "C", "A"]
    ... })
    >>> fig = sankeyBool(
    ...     df,
    ...     grouping_col="is_paid",
    ...     display_col="plan",
    ...     false_val="Not paid",
    ...     true_val="Paid",
    ...     display_col_vals=["A", "B", "C"],
    ...     colors_bool=["rgba(200,50,50,0.95)", "rgba(50,200,50,0.95)"],
    ...     colors_display_col_vals=[
    ...         "rgba(100,100,255,0.95)",
    ...         "rgba(255,200,100,0.95)",
    ...         "rgba(200,100,200,0.95)"
    ...     ]
    ... )
    >>> fig.show()
    """
    
    grade_wkn = df.groupby(grouping_col)[display_col].value_counts()
    
    df_s = grade_wkn.rename('value').reset_index()
    df_s['source_label'] = df_s[grouping_col].map({False: false_val,
                                                    True: true_val})
    
    nodes = list(
        df_s['source_label'].unique()) + list(df_s[display_col].unique())
    node_index = {label: i for i, label in enumerate(nodes)}
    
    sources = df_s['source_label'].map(node_index).tolist()
    targets = df_s[display_col].map(node_index).tolist()
    values = df_s['value'].tolist()
    
    node_color_map = {
        false_val: colors_bool[0],
        true_val: colors_bool[1]
    }
    target_color_map = {
        display_col_vals[i]:colors_display_col_vals[i] for i in range(
                                                len(display_col_vals))}
    node_colors = []
    
    for n in nodes:
        if n in node_color_map:
            node_colors.append(node_color_map[n])
        else:
            node_colors.append(target_color_map.get(n, 'rgba(150,150,150,0.9)'))
    
    link_colors = []
    
    for tgt in df_s[display_col]:
        base = target_color_map.get(tgt, 'rgba(150,150,150,0.9)')
        link_colors.append(base.replace('0.95', '0.6').replace('0.9', '0.6'))
    
    total = np.sum(values)
    hovertemplate = "%{label}<br>Flow: %{value}<br>Share: %{customdata:.2f}%<extra></extra>"
    
    link_percents = [v / total * 100 for v in values]
    
    sankey = go.Sankey(
        node=dict(
            pad=15,
            thickness=25,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=link_percents,
            hovertemplate="Source: %{source.label}<br>"
                        "Target: %{target.label}<br>"
                        "Value: %{value}<br>"
                        "Share of total: %{customdata:.2f}%<extra></extra>"
        )
    )
    
    fig = go.Figure(sankey)
    fig.update_layout(
        title_text=f"{display_col} flow by {false_val} vs {true_val}",
        font_size=12,
        height=500,
        margin=dict(l=20, r=20, t=60, b=40)
    )
    
    return fig


def pieWithBar(top_n: int, df: pd.DataFrame, column: str, bar_col_val: str,
                bar_observe: str, colors_bar: List[str]) -> plt:
    
    """
    Draw a  Matplotlib figure: a donut pie chart of the top-N
    categories in `column`, and a stacked vertical bar showing the distribution
    of `bar_observe` for a specific category value `bar_col_val`. Connection
    lines visually link the pie segment for `bar_col_val` to the stacked bar.
    
    Parameters
    ----------
    top_n : int
        Number of top categories (by count) from `column` to show explicitly in
        the pie chart. Remaining categories are aggregated into an "Other" 
        slice.
    df : pandas.DataFrame
        Input dataframe with at least the columns `column` and `bar_observe`.
    column : str
        Column name in `df` whose top categories will be plotted as a pie.
    bar_col_val : str
        Specific category value from `column` for which the stacked bar is
        drawn (filtered with `df[df[column] == bar_col_val]`).
    bar_observe : str
        Column name in `df` whose value counts (for `bar_col_val`) are shown
        as the stacked vertical bar (each unique value becomes a stacked
        segment).
    colors_bar : List[str]
        List of color strings (hex / named / rgba) used for each stacked bar
        segment. Length should be >= number of unique values in `df[df[column]
        == bar_col_val][bar_observe]`.
    
    Returns
    -------
    matplotlib.pyplot
        The matplotlib `plt` module used to build the figure (so one can
        call `plt.show()` or further tweak the axes).
    
    Example
    -------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> df = pd.DataFrame({
    ...     "family": ["A","A","B","C","A","D","B","C","E","F","A","B"],
    ...     "event":  ["x","y","x","y","x","x","y","x","y","x","y","x"]
    ... })
    >>> plt_mod = pieWithBar(
    ...     top_n=3,
    ...     df=df,
    ...     column="family",
    ...     bar_col_val="A",
    ...     bar_observe="event",
    ...     colors_bar=["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    ... )
    >>> plt_mod.show()
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0)
    target_labels = df[column].value_counts()
    top_families = target_labels.nlargest(top_n)
    other_sum = target_labels.iloc[top_n:].sum()
    final_labels = top_families.copy()
    final_labels['Other'] = other_sum
    
    wedges, texts, autotexts = ax1.pie(
        final_labels,
        autopct='%1.1f%%',
        startangle=360,
        labels=final_labels.index,
        pctdistance=0.8,
        wedgeprops=dict(width=0.4)
    )
    
    ax1.set_aspect('equal')
    ax1.set_title(f'Top {top_n} {column}')
    
    events = df[df[column] == bar_col_val][bar_observe].value_counts()
    
    events_frac = events / events.sum()
    bottom = 0.0
    width = .25
    
    for j, (label, frac) in enumerate(events_frac.items()):
        bc = ax2.bar(0, frac, width, bottom=bottom, label=label,
                    alpha=0.6 - 0.1 * j, color=colors_bar[j])
        ax2.bar_label(bc, labels=[f"{frac*100:.1f}%"], label_type='center')
        bottom += frac
    
    ax2.set_title(f'{bar_observe} distribution for {bar_col_val}')
    ax2.legend()
    ax2.axis('off')
    ax2.set_xlim(-2 * width, 2 * width)
    ax2.set_ylim(0, 1)
    
    theta1, theta2 = wedges[0].theta1, wedges[0].theta2
    center, r = wedges[0].center, wedges[0].r
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    bar_top = bottom
    bar_bottom = 0.0
    
    con_top = ConnectionPatch(
        xyA=(-width / 2, bar_top), coordsA=ax2.transData,
        xyB=(x, y), coordsB=ax1.transData,
        arrowstyle='-', linewidth=2.0, color='C0', zorder=5
    )
    ax2.add_artist(con_top)
    
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    
    con_bot = ConnectionPatch(
        xyA=(-width / 2, bar_bottom), coordsA=ax2.transData,
        xyB=(x, y), coordsB=ax1.transData,
        arrowstyle='-', linewidth=2.0, color='C0', zorder=5
    )
    ax2.add_artist(con_bot)
    
    return plt


def optimized_circle_packing(df: pd.DataFrame, geo_names: List[str],
                            col_values: Iterable, col_name: str, 
                            fsz: tuple[int, int],
                            top_n: int = 10) -> tuple[Figure, Axes]:
    
    """
    Create a grid of circle-packing plots for categorical proportions across
    multiple geographic groupings.
    This function builds a matplotlib figure with `len(geo_names)` rows and
    `len(col_values)` columns. For each (geo_name, col_value) pair it computes
    the proportion of values in `col_name` per geographic unit and draws a
    circle-packing plot using `plot_incident_circles(ax, ...)`.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe that contains at least the geographic columns listed in
        `geo_names` and the categorical column `col_name`.
    geo_names : List[str]
        List of column names (geographic groupers) to use for rows. Each entry
        will be grouped with `.groupby(name)`.
    col_values : Iterable
        Iterable of values from `col_name` that will be plotted across the 
        columns of the grid. Each element is passed as `col` to
        `plot_incident_circles`.
    col_name : str
        The column to analyze proportions of (used inside the groupby +
        value_counts).
    fsz : tuple[int, int]
        Figure size passed to `plt.subplots(figsize=fsz)`.
    top_n : int, optional (default=10)
        How many top categories to keep (passed down to
        `plot_incident_circles`).
    
    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes (or its subclass)
        The created `fig` and the `axes` array returned by `plt.subplots`.
    
    Example
    -------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> df = pd.DataFrame({
    ...     "region": ["R1","R1","R2","R2","R3","R3","R1","R2","R3"],
    ...     "city":   ["C1","C2","C1","C2","C1","C2","C1","C2","C1"],
    ...     "incident_type": ["A","B","A","A","B","C","C","A","B"]
    ... })
    >>> fig, axes = optimized_circle_packing(
    ...     df=df,
    ...     geo_names=["region", "city"],
    ...     col_values=["A", "B", "C"],
    ...     col_name="incident_type",
    ...     fsz=(12, 8),
    ...     top_n=5
    ... )
    >>> fig.savefig("circle_grid.png")
    """
    
    fig, axes = plt.subplots(len(geo_names), len(col_values), figsize=fsz)
    
    for place, name in enumerate(geo_names):
        geo_data = df.groupby(name,
                            observed=False)[col_name].value_counts(
                                True).unstack(fill_value=0).reset_index()
        for place2, col in enumerate(col_values):
            plot_incident_circles(axes[place, place2], geo_data,
                name, col,
                f'{col} proportions by {name}', top_n)
    
    plt.tight_layout()
    return fig, axes


def plot_incident_circles(ax: Axes, data: pd.DataFrame, geo_level: str,
                        incident_type: str, title: str,
                        top_n: int = 10) -> None:
    
    """
    Draw a circle-packing visualization of the top-N geographic units for a
    given incident type on a provided Matplotlib Axes.
    The function filters rows where `incident_type` has positive values,
    selects the `top_n` largest values, and uses the `circlify` library to
    render packed circles whose sizes are proportional to the incident values.
    Each circle is labeled with the corresponding geographic name.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object on which the circle-packing plot will be drawn.
    data : pandas.DataFrame
        DataFrame containing at least geographic labels and numeric incident
        values.
    geo_level : str
        Column name in `data` representing the geographic unit (e.g. city,
        region). Values from this column are used as circle labels.
    incident_type : str
        Column name in `data` containing numeric values used to determine
        circle sizes.
    title : str
        Title text displayed above the plot.
    top_n : int, optional (default=10)
        Number of largest geographic units (by `incident_type`) to visualize.
    
    Returns
    -------
    None
    """
    
    filtered_data = data[data[incident_type] > 0].nlargest(top_n, incident_type)
    
    if len(filtered_data) == 0:
        ax.text(0.5, 0.5, f"No {incident_type} data", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        return
    
    values = filtered_data[incident_type].tolist()
    labels = filtered_data[geo_level].tolist()
    
    circles = circlify.circlify(values, show_enclosure=False, 
                                target_enclosure=circlify.Circle(x=0, y=0, r=1))
    circles = circles[::-1]
    
    for circle, label in zip(circles, labels):
        x, y, r = circle
        ax.add_patch(plt.Circle((x, y), r, alpha=0.7, linewidth=2))
        ax.annotate(f"{label}", (x, y), va='center', ha='center', 
                    fontweight='bold', fontsize=9)
    
    ax.set_title(f'{title}\nTop-{len(filtered_data)} largest')
    ax.axis('off')
    ax.set_aspect('equal')
    
    lim = max(abs(circle.x) + circle.r for circle in circles) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def graphFromPairs(top_pairs: int, pair_Counter_class: Counter, k_pos: int,
                    iter_pos: int, plot_width: int,
                    plot_height: int) -> go.Figure:
    
    """
    Build an interactive network graph from the most frequent item pairs
    and visualize it using plotly.
    The function takes the `top_pairs` most common elements from a Counter
    containing iterable pairs. For each pair, all 2-combinations are converted
    into graph edges. A networkx graph is constructed, positioned using a
    spring layout, and rendered as a plotly figure.
    
    Parameters
    ----------
    top_pairs : int
        Number of most common pairs to take from `pair_Counter_class`.
    pair_Counter_class : collections.Counter
        Counter where keys are iterables of items (e.g., tuples of tags) and
        values are occurrence counts.
    k_pos : int
        Spring layout optimal distance between nodes passed to
        `networkx.spring_layout`.
    iter_pos : int
        Number of iterations for the spring layout algorithm.
    plot_width : int
        Width of the output plotly figure in pixels.
    plot_height : int
        Height of the output plotly figure in pixels.
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure displaying the network graph, where nodes
        represent unique items and edges represent co-occurrence relations
        derived from the input pairs.
    
    Notes
    -----
    - Nodes with no connections (isolates) are removed before visualization.
    - Node positions are determined using a force-directed (spring) layout,
        so layout may differ slightly between runs.
    
    Example
    -------
    >>> import plotly.graph_objects as go
    >>> from collections import Counter
    >>> pair_counter = Counter({
    ...     ("T1059", "T1078"): 15,
    ...     ("T1059", "T1082"): 10,
    ...     ("T1078", "T1027"): 8,
    ...     ("T1059", "T1027"): 6,
    ...     ("T1082", "T1027"): 4,
    ...     ("T1105", "T1059"): 3
    ... })
    >>> fig = graphFromPairs(
    ...     top_pairs=5,
    ...     pair_Counter_class=pair_counter,
    ...     k_pos=1,
    ...     iter_pos=200,
    ...     plot_width=900,
    ...     plot_height=900
    ... )
    >>> fig.show()
    """
    
    edges = []
    graph_pairs = pd.DataFrame(pair_Counter_class.most_common(top_pairs),
        columns=['pair', 'count'])
    
    for tags in graph_pairs['pair']:
        if len(tags) > 1:
            edges.extend(list(combinations(tags, 2)))
    
    edges_df = pd.DataFrame(edges, columns=['src','dst'])
    G = nx.from_pandas_edgelist(edges_df, 'src', 'dst')
    G.remove_nodes_from(list(nx.isolates(G)))
    pos = nx.spring_layout(G, k=k_pos, iterations=iter_pos) 
    edge_x = []
    edge_y = []
    
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1),
        mode='lines',
        hoverinfo='none'
    )
    node_x = []
    node_y = []
    texts = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        texts.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=texts,
        textposition="top center",
        marker=dict(size=10),
        hoverinfo='text'
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False,
        width=plot_width, height=plot_height,
        title=f"MITRE Techniques Graph from top {top_pairs} pairs")
    
    return fig