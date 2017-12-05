# File: spanconverter.py
# Class file for convertering SPAN data to game state data
# see https://snap.stanford.edu/data/email-Eu-core.html for source data
# see game/reader for destination data

import numpy as np

class SnapConverter(object):
    """Convert the SNAP data to a """
    def __init__(self):
        # graph variables
        self.config = config
        self.max_con = 150
        self.cap_values = config.cap_values
        self.max_lfr = config.max_lfr
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = 1

