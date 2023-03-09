# from lnn import Propositions, And, Fact, Model, Loss, Direction, Implies, Or, Not
from lnn.symbolic.logic.connective_neuron import _ConnectiveNeuron
from .viz import get_pos

from dash import Dash, html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto

node_state_color_map = {
    "EXACT_UNKNOWN": "#0f6afe",
    "CONTRADICTION": "#da1e28",
    "APPROX_TRUE": "#42be65",
    "APPROX_FALSE": "#8e6a00",
    "APPROX_UNKNOWN": "#78a9ff",
    "FALSE": "#d2a106",
    "TRUE": "#198038",
    "UNKNOWN": "#697077",
    "APPROX_CONTRADICTION": "#ff8389",
}

default_stylesheet = [
    {
        "selector": "node",
        "style": {
            "width": "5px",
            "height": "5px",
            "shape": "hexagon",
            "label": "data(label)",
            "font-size": "5%",
            "color": "black",
            "text-halign": "center",
            "text-opacity": 1,
            "text-valign": "center",
            "padding": "3% 3% 3% 3%",
            "border-width": "0.5",
            "background-color": "data(color)",
            "border-color": "data(basecolor)",
        },
    },
    {"selector": ".connective", "style": {}},
    {
        "selector": ".propositional",
        "style": {
            "width": "2px",
            "height": "2px",
            "color": "black",
            "shape": "ellipse",
            "text-halign": "left",
            "text-margin-x": "-1px",
        },
    },
    {
        "selector": ".propositional.parent",
        "style": {"border-opacity": "0", "background-opacity": "0"},
    },
    {
        "selector": ".connective.parent",
        "style": {
            "border-opacity": "0",
            "background-opacity": "0",
            "padding": "-10",
            "z-compound-depth": "orphan",
            "z-index": "10",
            "text-halign": "right",
            "text-margin-x": "1px",
            "color": "black",
            "text-background-color": "#eeeeee",
            "text-background-opacity": "1",
            "text-background-padding": "1px",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": "data(weight)",
            "z-compound-depth": "orphan",
            "z-index": "1",
            "line-color": "data(color)",
            "source-arrow-shape": "triangle",
            "arrow-scale": "0.5",
            "source-arrow-color": "data(color)",
            "curve-style": "unbundled-bezier",
            "control-point-weights": "0.15",
            "control-point-distances": "data(cpdistance)",
        },
    },
]


class LNNDash:
    def __init__(self, model, height=700, max_label_len=10):
        self.model = model
        self.max_label_len = max_label_len
        self.height = 700
        self.width = height

        self.graph = self.model.graph
        self.pos = get_pos(model)

        # create cytoscape elements
        self.graph_adj_list = self.get_adj_list()
        self.id_to_node = dict()
        self.node_index_map = dict()
        self.elements = self.get_elements()

        self.app = Dash(__name__)
        self.app.layout = self.get_app_layout()
        self.create_callbacks()

    def run(self, debug=False):
        self.app.run_server(debug=debug)

    def get_adj_list(self):
        children = {}
        for x, y in self.graph.edges():
            if x not in children:
                children[x] = set()
            children[x].add(y)
        return children

    def rescale_pos(self, x, y):
        layer_widths = {}
        for n in self.pos:
            yi = self.pos[n][1]
            layer_widths[yi] = layer_widths.get(yi, 0) + 1
        width_factor = max(layer_widths.values())

        labels = [len(n.name) for n in self.pos if not isinstance(n, str)]
        labels.remove(max(labels))
        num_layers = len(layer_widths)

        # flip coordinates
        # scale with some heuristics
        factor = max(num_layers * self.max_label_len * 5, width_factor * 10)
        xc = y * factor
        yc = x * factor
        # if width_factor < 4:
        #    xc *= len(self.pos)
        return xc, yc, factor

    def label_node(self, node):
        # if connective, label as connective character
        node_name = node.name
        # if len(name) short enough, keep full
        if self.max_label_len is None or len(node_name) <= self.max_label_len:
            return node_name
        name = ""
        # get first three upper case letters
        for i in range(len(node_name)):
            if node_name[i].isupper():
                name = name + node_name[i]
            if len(name) >= self.max_label_len:
                return name + "..."
        # if no uppercase letters, return first three letters
        if len(name) == 0:
            return node_name[: self.max_label_len] + "..."
        return name

    def node_marker(self, node):
        if hasattr(node, "connective_str"):
            return node.connective_str
        else:
            return ""

    def float_to_rgb(
        self, x, color0="#d4bbff", color1="#6929c4", min_int=0.1, max_int=1
    ):
        # calculate rgb, interpolation of #6929c4 (bigger) and #d4bbff (smaller)
        [r0, g0, b0] = ["0x" + color0[i : i + 2] for i in range(1, len(color0), 2)]
        [r1, g1, b1] = ["0x" + color1[i : i + 2] for i in range(1, len(color1), 2)]

        x = (x * (max_int - min_int)) + min_int

        r = (1 - x) * int(r0, 16) + (x) * int(r1, 16)
        g = (1 - x) * int(g0, 16) + (x) * int(g1, 16)
        b = (1 - x) * int(b0, 16) + (x) * int(b1, 16)

        result = "#%02x%02x%02x" % (int(r), int(g), int(b))
        return result

    def get_elements(self):
        elements = []
        # node elements
        for node in self.graph.nodes():
            x, y, factor = self.rescale_pos(self.pos[node][0], self.pos[node][1])
            if hasattr(node, "connective_str"):
                node_class = "connective"
            else:
                node_class = "propositional"
            # add parent node for full label
            elements.append(
                {
                    "data": {
                        "id": node.name + "parent",
                        "label": self.label_node(node),
                        "size": factor * 10,
                    },
                    "classes": "parent " + node_class,
                }
            )
            node_state_name = node.state().name
            node_class += " " + node_state_name
            color1 = node_state_color_map[node_state_name]
            certainty = 1 - abs(node.get_data()[0] - node.get_data()[1])
            if node_state_name == "CONTRADICTION":
                certainty = 1 - certainty
            color = self.float_to_rgb(certainty, color0="#ffffff", color1=color1)
            elements.append(
                {
                    "data": {
                        "id": node.name,
                        "label": self.node_marker(node),
                        "parent": node.name + "parent",
                        "opacity": 1,
                        "color": color,
                        "basecolor": color1,
                    },
                    "position": {"x": x, "y": y},
                    "classes": "" + node_class,
                }
            )
            self.node_index_map[node] = len(elements) - 2
            self.id_to_node[node.name] = node

        # edge elements
        edge_elements = []
        for u, v in self.graph.edges():
            weight = 1.0
            # find the edge weight
            if isinstance(u, _ConnectiveNeuron):
                # find edge weight relative to local sum
                weight = (
                    u.neuron.weights[u.operands.index(v)].item()
                    / u.neuron.weights.max().item()
                )
            goes_lower = ((self.pos[v][0] - self.pos[u][0])) - 0.5
            edge_elements.append(
                {
                    "data": {
                        "source": u.name,
                        "target": v.name,
                        "color": self.float_to_rgb(weight),
                        "weight": str(weight),
                        "cpdistance": str(-20 * goes_lower),
                    }
                }
            )
        self.edge_elements = edge_elements
        elements += edge_elements
        return elements

    def get_app_layout(self):
        layout = html.Div(
            [
                # sidebar
                html.Div(
                    [
                        # output
                        html.P(
                            id="output_text",
                            style={
                                "white-space": "pre-wrap",
                                "background-color": "#EEEEEE",
                                "width": "30%",
                                "height": str(self.height / 2) + "px",
                                "float": "left",
                            },
                        ),
                        html.Button(
                            "Show full graph",
                            id="btn-show-full-graph",
                            n_clicks_timestamp=0,
                        ),
                        html.Button(
                            "Show node subgraph",
                            id="btn-subgraph",
                            n_clicks_timestamp=0,
                        ),
                    ]
                ),
                # graph
                cyto.Cytoscape(
                    id="lnn-graph",
                    elements=self.elements,
                    layout={"name": "preset"},
                    style={
                        "background-color": "#EEEEEE",
                        "width": "70%",
                        "height": "calc(100vh - 50px)",
                        "float": "left",
                    },
                    stylesheet=default_stylesheet,
                ),
            ]
        )
        """html.Div([
                daq.BooleanSwitch(
                    id='name-label-toggle',
                    on=True,
                    label='Show node name instead of node label',
                    labelPosition='right',
                    style={}
                )
                ],
                id='settings',
                style={'width': '30%',
                       'float': 'left'}
        )"""

        return layout

    def create_callbacks(self):
        # display full node name in div
        @self.app.callback(
            Output("output_text", "children"),
            Input("lnn-graph", "selectedNodeData"),
            State("lnn-graph", "elements"),
        )
        def create_caption(nodeData, elements):
            # print(nodeData)
            # print(elements)
            if nodeData is None or len(nodeData) == 0:
                return "Click on a node to see full formula"
            text = "Selected node(s): \n"
            for n in nodeData:
                text += n["id"] + "\n"
            return text

        # toggl between node name and label
        """@self.app.callback(
            Output('lnn-graph', 'elements'),
            Input('name-label-toggle', 'value'),
            State('lnn-graph', 'elements')
        )
        def update_labels(value, elements):
            new_el = elements
            for el in new_el:

            return new_el"""

        @self.app.callback(
            Output("lnn-graph", "elements"),
            Input("btn-show-full-graph", "n_clicks_timestamp"),
            Input("btn-subgraph", "n_clicks_timestamp"),
            State("lnn-graph", "elements"),
            State("lnn-graph", "selectedNodeData"),
        )
        def update_elements(btn_show, btn_subgraph, elements, nodeData):
            if int(btn_show) > int(btn_subgraph):
                elements = self.elements
            elif int(btn_subgraph) > 0:
                if nodeData is not None and len(nodeData) == 1:
                    node_name = nodeData[0]["id"]
                    node = self.id_to_node[node_name]
                    include_indices = [
                        self.node_index_map[node],
                        self.node_index_map[node] + 1,
                    ]
                    included_node_names = set([node_name])
                    to_check = list(self.graph_adj_list[node])
                    while len(to_check) > 0:
                        node = to_check[0]
                        to_check = to_check[1:]
                        include_indices.append(self.node_index_map[node])
                        include_indices.append(self.node_index_map[node] + 1)
                        if node in self.graph_adj_list:
                            to_check += list(self.graph_adj_list[node])
                        included_node_names.add(node.name)
                    elements = [elements[i] for i in include_indices]
                    for edge in self.edge_elements:
                        source = edge["data"]["source"]
                        if source in included_node_names:
                            elements.append(edge)
            return elements
