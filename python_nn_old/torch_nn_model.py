import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTMNeuralNetworkModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, device="cpu", drop_prob=0.5):
        super(LSTMNeuralNetworkModel, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Tanh()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


class NeuralNetworkModel(ABC):
    def __init__(self, name, hidden_size=100):
        self.device = "cuda"
        self.lr = 0.005
        self.model = LSTMNeuralNetworkModel(input_size=12, output_size=6, hidden_dim=100, n_layers=2, device=self.device).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, x):
        pass

    def fit(self, x, y, x_val, y_val, save_path, iteration=3000, patience=100, batch_size=1000):
        cur_patience = 0
        best_mse = 999.
        n_train_batches = x.shape[0] // batch_size
        n_valid_batches = x_val.shape[0] // batch_size
        rand_idx = np.arange(x.shape[0])

        for epoch in range(iteration):

            training_loss = 0.
            valid_loss = 0.

            np.random.shuffle(rand_idx)
            rand_in_train = x[rand_idx]
            rand_out_train = y[rand_idx]

            for i in range(n_train_batches - 1):
                idx = slice(i * batch_size, i * batch_size + batch_size)

                minibatch_x = rand_in_train[idx]
                minibatch_y = rand_out_train[idx]

                # feed_dict = {
                #     self.x: minibatch_x,
                #     self.y: minibatch_y,
                # }

                # _, loss = self.session.run([self.step_op, self.mse], feed_dict)

                inputs, labels = minibatch_x.to(self.device), minibatch_y.to(self.device)
                self.model.zero_grad()
                output, h = self.model(inputs, h)
                loss = self.criterion(output.squeeze(), labels.float())
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()

                training_loss += loss / n_train_batches

            for i in range(n_valid_batches - 1):
                idx = slice(i * batch_size, i * batch_size + batch_size)

                minibatch_x = x_val[idx]
                minibatch_y = y_val[idx]

                # feed_dict = {
                #     self.x: minibatch_x,
                #     self.y: minibatch_y,
                # }

                # y_pred, loss = self.session.run([self.output, self.mse], feed_dict)

                valid_loss += loss / n_valid_batches

            if valid_loss < best_mse:
                best_mse = valid_loss
                self.save_model(save_path)
                cur_patience = 0
            else:
                cur_patience += 1

            if epoch % 10 == 0:
                print("#%04d: Traing loss %.5f, Valid loss %.5f" % (epoch, training_loss, valid_loss))

            if cur_patience >= patience:
                print("Stop training after %04d iterations" % epoch)
                break


lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)