"""
Core module defining the functions for converting a consumption Markov Decision 
Process from consMDP model to dot representation and present it. 
"""

import subprocess
import sys
dotpr = 'dot'
debug = False

from math import inf
#TODO build a list and join it in the end into string

tab_MI_style         = ' border="0" cellborder="0" cellspacing="0"' +\
                       ' cellpadding="1" align="center" valign="middle"' +\
                       ' style="rounded" bgcolor="#ffffff50"'
if debug:
    tab_MI_style         = ' border="1" cellborder="1" cellspacing="0" cellpadding="0"'

tab_state_cell_style = ' rowspan="{}"'

tab_MI_cell_style    = ' align="center" valign="middle"'
tab_MI_cell_font     = ' color="orange" point-size="10"'

tab_SR_cell_style    = tab_MI_cell_style
tab_SR_cell_font     = ' color="red" point-size="10"'

# Positive reachability
tab_PR_cell_style    = tab_MI_cell_style
tab_PR_cell_font     = ' color="deepskyblue" point-size="10"'

# Almost sure reachability
tab_AR_cell_style    = tab_MI_cell_style
tab_AR_cell_font     = ' color="dodgerblue4" point-size="10"'

# Reachability-safe
tab_RS_cell_style    = tab_MI_cell_style
tab_RS_cell_font     = ' color="blue4" point-size="10"'

# Büchi
tab_BU_cell_style    = tab_MI_cell_style
tab_BU_cell_font     = ' color="forestgreen" point-size="10"'

# Büchi-safe
tab_BS_cell_style    = tab_MI_cell_style
tab_BS_cell_font     = ' color="darkgreen" point-size="10"'

targets_style        = ', style="filled", fillcolor="#0000ff20"'
targets_Buchi_style  = ', style="filled", fillcolor="#00ff0020"'

default_options = "msrRb"

class consMDP2dot:
    """Convert consMDP to dot"""
    
    def __init__(self, mdp, options=""):
        self.mdp = mdp
        self.str = ""
        self.options = default_options + options

        self.act_color = "black"
        self.prob_color = "gray52"
        self.label_row_span = 2

        self.opt_mi = False # MinInitCons
        self.opt_sr = False # Safe levels
        self.opt_pr = False # Positive reachability
        self.opt_ar = False # Almost-sure reachability
        self.opt_bu = False # Büchi

        self.el = mdp.energy_levels

        if "m" in self.options:
            self.opt_mi = self.el is not None and self.el.mic_values is not None
        if "M" in self.options:
            mdp.get_minInitCons()
            self.opt_mi = True
            self.el = mdp.energy_levels

        if "s" in self.options:
            self.opt_sr = self.el is not None and self.el.safe_values is not None
        if "S" in self.options:
            mdp.get_safe()
            self.opt_sr = True
            self.el = mdp.energy_levels

        if "r" in self.options:
            self.opt_pr = self.el is not None and self.el.pos_reach_values is not None
        if "R" in self.options:
            self.opt_ar = self.el is not None and self.el.alsure_values is not None

        if "b" in self.options:
            self.opt_bu = self.el is not None and self.el.buchi_values is not None
            if self.opt_bu:
                self.label_row_span = 3
        #print(self.opt_bu,file=stderr)

    def get_dot(self):
        self.start()
        
        m = self.mdp
        for s in range(m.num_states):
            self.process_state(s)
            for a in m.actions_for_state(s):
                self.process_action(a)
        
        self.finish()
        return self.str
        
    def start(self):
        gr_name = self.mdp.name if self.mdp.name else ""
   
        self.str += f"digraph \"{gr_name}\" {{\n"
        self.str += "  rankdir=LR\n"
        
    def finish(self):
        self.str += "}\n"
        
    def get_state_name(self, s):
        name = s if self.mdp.names[s] is None else self.mdp.names[s]
        return name
    
    def process_state(self, s):
        self.str += f"\n  {s} ["

        # name
        state_str = self.get_state_name(s)

        # minInitCons
        if self.opt_mi or self.opt_sr or self.opt_pr or self.opt_bu:
            state_str = f"<table{tab_MI_style}>" + \
                        f"<tr><td{tab_state_cell_style.format(self.label_row_span)}>{state_str}</td>"

        if self.opt_mi:
            val = self.el.mic_values[s]
            val = "∞" if val == inf else val
            state_str += f"<td{tab_MI_cell_style}>" + \
                f"<font{tab_MI_cell_font}>{val}</font></td>"

        if self.opt_sr:
            val = self.el.safe_values[s]
            val = "∞" if val == inf else val
            state_str += f"<td{tab_SR_cell_style}>" + \
                f"<font{tab_SR_cell_font}>{val}</font></td>"

        if self.opt_mi or self.opt_sr or self.opt_pr or self.opt_bu:
            state_str += f"</tr><tr>"

            empty_row = True
            # positive reachability
            if self.opt_pr:
                empty_row = False
                val = self.el.pos_reach_values[s]
                val = "∞" if val == inf else val
                state_str += f"<td{tab_PR_cell_style}>" + \
                    f"<font{tab_PR_cell_font}>{val}</font></td>"

            # almost-sure reachability
            if self.opt_ar:
                empty_row = False
                val = self.el.alsure_values[s]
                val = "∞" if val == inf else val
                state_str += f"<td{tab_AR_cell_style}>" + \
                    f"<font{tab_AR_cell_font}>{val}</font></td>"
                val = self.el.reach_safe[s]
                val = "∞" if val == inf else val
                state_str += f"<td{tab_RS_cell_style}>" + \
                    f"<font{tab_RS_cell_font}>{val}</font></td>"

            if empty_row:
                state_str += "<td></td>"

            # buchi
            if self.opt_bu:
                state_str += f"</tr><tr>"
                empty_row = False
                val = self.el.buchi_values[s]
                val = "∞" if val == inf else val
                state_str += f"<td{tab_BU_cell_style}>" + \
                    f"<font{tab_BU_cell_font}>{val}</font></td>"
                val = self.el.buchi_safe[s]
                val = "∞" if val == inf else val
                state_str += f"<td{tab_BS_cell_style}>" + \
                    f"<font{tab_BS_cell_font}>{val}</font></td>"

            if empty_row:
                state_str += "<td></td>"

            state_str += "</tr></table>"

        self.str += f'label=<{state_str}>'

        # Reload states are double circled and target states filled
        if self.mdp.is_reload(s):
            self.str += ", peripheries=2"
        if (self.opt_pr or self.opt_ar or self.opt_bu) and s in self.el.targets:
            self.str += targets_style
        self.str += "]\n"

    def process_action(self, a):
        act_id = f"\"{a.src}_{a.label}\""
        
        # Src -> action-node
        self.str += f"    {a.src} -> {act_id}"
        self.str += f"[arrowhead=onormal,label=\"{a.label}: {a.cons}\""
        self.str += f", color={self.act_color}, fontcolor={self.act_color}]\n"
        
        # action-node
        self.str += f"    {act_id}[label=<>,shape=point]\n"
        
        # action-node -> dest
        for dst, p in a.distr.items():
            self.str += f"      {act_id} -> {dst}[label={p}, color={self.prob_color}, fontcolor={self.prob_color}]"
        

def dot_to_svg(dot_str):
    """
    Send some text to dot for conversion to SVG.
    """
    try:
        dot_pr = subprocess.Popen([dotpr, '-Tsvg'],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("The command 'dot' seems to be missing on your system.\n"
              "Please install the GraphViz package "
              "and make sure 'dot' is in your PATH.", file=sys.stderr)
        raise

    stdout, stderr = dot_pr.communicate(dot_str.encode('utf-8'))
    if stderr:
        print("Calling 'dot' for the conversion to SVG produced the message:\n"
              + stderr.decode('utf-8'), file=sys.stderr)
    ret = dot_pr.wait()
    if ret:
        raise subprocess.CalledProcessError(ret, 'dot')
    return stdout.decode('utf-8')
