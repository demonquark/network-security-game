# test for reader.py

from game import State, Config, StateReader

class TestStateReader(object):
    """Tests for the Reader of the State and Config class"""
    def __init__(self):
        # create a new state
        self.reader = StateReader()

    def write_state(self, state):
        """Test for writing a config to a csv"""
        self.reader.write_state(state)

    def read_state(self):
        """Test for writing a config to a csv"""
        return self.reader.read_state()
