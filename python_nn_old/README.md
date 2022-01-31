# Neural network ball socket estimator

### Install

* Install Conda: 
`https://www.anaconda.com/products/individual`

* Create a conda environment:
`conda create -n roboy python=3.8`

* Activate the environment:
  `conda activate roboy`

* Install requirement:
`pip install -r requirement.txt`

### Run
* Run the trained network `python predict.py`

* Visualize the `rqt_plot` validation `python visualize_data.py --body_part BODY_PART`

* Generate data with `python particle_swarm.py --body_part BODY_PART`
 with `BODY_PART = {head, shoulder_left, shoulder_right, hand_left, hand_right}`