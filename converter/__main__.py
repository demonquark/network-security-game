"""
# File: Main code
# Runs the converter as requested
"""

import sys
from .snapconverter import SnapConverter


def main():
    """ Run the converter module"""

    # get the command line arguments
    valid_arguments = ["run"]
    arg_id = -1

    if len(sys.argv) > 1 and sys.argv[1] in valid_arguments:
        arg_id = valid_arguments.index(sys.argv[1])

    if arg_id == 0:
        # run the game for the requested state
        snap = SnapConverter(sys.argv[2] if len(sys.argv) > 2 else None)
        snap.read_state("snap_data.txt", "snap_data.csv")
    else:
        print("Invalid command line argument. Please add one of the following arguments: {}".format(valid_arguments))


# ------- START MAIN CODE --------
main()
