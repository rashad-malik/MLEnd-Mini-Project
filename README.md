## Data Setup
This project requires additional data that is not included in the repository due to size constraints.

To download the data, install the mlend library (make sure you have version 1.0.0.4):

`pip install mlend==1.0.0.4`

Import the library and it's functions:

`import mlend
from mlend import download_deception_small, deception_small_load`

Now you can download the data:

`datadir = download_deception_small(save_to='MLEnd', subset={}, verbose=1, overwrite=False)`