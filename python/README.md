# Neural network ball socket estimator

## Install

* Pytorch `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

* `pip install -r requirements.txt`

## Data collection

* Generate data using `python particle_swarm.py --body_part BODY_PART`
  
 with `BODY_PART = {head, shoulder_left, shoulder_right, hand_left, hand_right}`

## Training

* Convert `bags` to `csv` using `write_dataset_from_bag.ipynb`

* Run `data_analysis.ipynb` to make sure all the inputs and outputs are correct

* Train an LSTM network using `training.ipynb`

## Prediction

* Run the trained network `python predict_lstm.py`

