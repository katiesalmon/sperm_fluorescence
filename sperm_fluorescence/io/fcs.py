"""FCS access for the package: the standard-library readers under scripts/ (which must stay
importable on the server without the package) re-exported from one place."""
import sys

from .. import SCRIPTS

if SCRIPTS not in sys.path: sys.path.insert(0, SCRIPTS)
import fcs_data  # noqa: E402,F401
import fcs_probe  # noqa: E402,F401

read_text_segment = fcs_probe.read_text_segment
parameters = fcs_probe.parameters
read_matrix = fcs_data.read_matrix
column_index = fcs_data.column_index
