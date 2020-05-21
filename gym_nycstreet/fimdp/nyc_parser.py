"""
Example related module for parsing electric vehicle energy consumption data
from a json file. 
"""

import consMDP
import json
from decimal import Decimal

def load_json(filename):
    with open(filename,'r') as f:
        return json.load(f)

def convert(g):
    states = []
    actions = dict()

    mdp = consMDP.ConsMDP()

    for node in g["nodes"]:
        if node["action"]:
            actions[node["label"]] = dict()
        else:
            states.append(node)

    for s in states:
        mdp.new_state(s["reload"], s["label"])

    for edge in g["edges"]:
        fr = edge["tail"]
        to = edge["head"]
        if to in actions:
            actions[to]["from"] = fr
            actions[to]["cons"] = edge["consumption"]
        else:
            dist = actions[fr].get("dist")
            to = mdp.state_with_name(to)
            if dist is None:
                actions[fr]["dist"] = dict()
            actions[fr]["dist"][to] = Decimal(f'{edge["probability"]}')

    for label, a in actions.items():
        fr = mdp.state_with_name(a["from"])
        mdp.add_action(fr, a["dist"], label, a["cons"])

    return mdp


def get_target_set(g, mdp):
    T = set()
    for t in g["T"]:
        T.add(mdp.state_with_name(t["label"]))
    return T

def parse(filename):
    js = load_json(filename)
    
    mdp = convert(js)
    target = get_target_set(js, mdp)
    
    return (mdp, target)
