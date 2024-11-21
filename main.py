import math
import numpy as np
import matplotlib.pyplot as plt


from engine import Value
from utils import *


def temp_func():
    # inputs 
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bias
    b = Value(6.8813735870195432, label="b")
    # x1w1 + x2w2 + b
    x1w1 = x1*w1; x1w1.label = "x1*w1"
    x2w2 = x2*w2; x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b; n.label='n'
    o = n.tanh(); o.label="o"
    return draw_dot(o)
    # dot.render("graph", view=True)

if __name__ == '__main__':
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    # dot = draw_dot(d)
    # dot.render('graph', view=True)
    temp_func()