# Neural network ball socket estimator

## Install

* Pytorch `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

## Data collection

* Generate data with `python particle_swarm_.py --body_part BODY_PART`
  
 with `BODY_PART = {head, shoulder_left, shoulder_right, hand_left, hand_right}`

## Training

* Convert from `bags` to `csv` using `write_dataset_from_bag.ipynb`

* Data analysis to make sure all the inputs and outputs are correct using `data_analysis.ipynb`

* Training an LSTM network `training.ipynb`

## Prediction

* Run the trained network `python predict_lstm.py`

