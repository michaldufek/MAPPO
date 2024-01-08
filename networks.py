import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

T.autograd.set_detect_anomaly(True)

class CriticNetwork(nn.Module):
    def __init__(self, beta, fc1_dims, fc2_dims, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Convolutional layers using Sequential
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Calculate the size of the output from the last convolutional layer
        conv_output_size = self._calculate_conv_output_dims()

        # Fully connected layers using Sequential
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state):
        state = state.view(state.size(0), -1, state.size(3), state.size(4))
        conv_out = self.conv_layers(state)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten the output

        value = self.fc_layers(conv_out)

        return value.squeeze() # (mini_batch_size, 1) --> (mini_batch_size,) else mismatch of dims during backprop

    def _calculate_conv_output_dims(self):
        # Example input for one sample: (5 agent/channels, 128 height, 128 width)
        temp_input = T.zeros(1, 5, 128, 128).to(self.device)

        # Pass the temporary input through convolution layers
        conv_out = self.conv_layers(temp_input)
        # Calculate the flattened output size
        conv_output_size = conv_out.size(1) * conv_out.size(2) * conv_out.size(3) # product of channels, height, and width
        return conv_output_size

    def save_checkpoint(self):
        # Create directory if it doesn't exist 
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)

        T.save(self.state_dict(), self.chkpt_file)
        print("-- Critic saved")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        
        # Calculate the size of the output from the last convolutional layer
        conv_output_size = self._calculate_conv_output_dims(input_dims)
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x1 = self.conv1(state)
        x2 = self.conv2(x1)
        # Here is the issue
        x3 = x2.view(x2.size(0), -1)  # Flatten the output for the fully connected layers but preserve batch size
        #print(f'x3 shape is: {x3.shape}')
        x4 = self.fc1(x3)
        x5 = self.fc2(x4)
        pi = self.pi(x5)
        action_probabilities = T.softmax(pi, dim=1)

        return action_probabilities

    def _calculate_conv_output_dims(self, input_dims):
        # Example input: input_dims = (1, 128, 128) for a single-channel 128x128 input

        # Let's assume two convolutional layers for this example
        # Conv1 parameters
        conv1_filter_size = 3  # 3x3 filter
        conv1_stride = 1
        conv1_padding = 0

        # Conv2 parameters
        conv2_filter_size = 3
        conv2_stride = 1
        conv2_padding = 0

        # Calculate output size after first convolutional layer
        output_size = (input_dims[1] - conv1_filter_size + 2 * conv1_padding) // conv1_stride + 1

        # Calculate output size after second convolutional layer
        output_size = (output_size - conv2_filter_size + 2 * conv2_padding) // conv2_stride + 1

        # If the network includes pooling layers or additional convolutional layers, 
        # continue the calculation in the same manner

        # Since output will be flattened for a fully connected layer, 
        # calculate the total number of features
        num_final_conv_channels = 64
        total_output_features = output_size * output_size * num_final_conv_channels  # num_final_conv_channels is the number of output channels in the last convolutional layer

        return total_output_features


    def save_checkpoint(self):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)

        T.save(self.state_dict(), self.chkpt_file)
        print("-- Actor saved")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))