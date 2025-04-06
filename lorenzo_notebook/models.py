import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim, dropout=0.1, max_seq_len=24):
        super(TransformerDecoderOnly, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]

        # Project input to hidden dimension
        x = self.input_projection(x)

        # Prepare memory (self-attention within the decoder)
        memory = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)

        # Pass through the transformer decoder
        output = self.transformer_decoder(memory, memory)

        # Project to output dimension
        output = self.output_projection(output[-1])  # Use the last time step for prediction

        return output


class xLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(xLSTMModel, self).__init__()
        self.xlstm = xLSTM(input_dim, hidden_dim, num_layers)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()  # Ensure non-negative output

    def forward(self, x):
        # xLSTM expects input of shape (batch_size, seq_length, input_dim)
        out = self.xlstm(x)
        out = self.fc_1(out[:, -1, :])  # Use the last hidden state for prediction
        out = self.dropout(out)
        out = self.fc_2(out)
        out = self.relu(out)
        return out


class xLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(xLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create xLSTM layers
        self.layers = nn.ModuleList([xLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

    def forward(self, x):
        # Initialize hidden and cell states
        batch_size, seq_length, _ = x.size()
        hidden_states = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        cell_states = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        # Process sequence
        outputs = []
        for t in range(seq_length):
            input_t = x[:, t, :]
            for i, layer in enumerate(self.layers):
                hidden_states[i], cell_states[i] = layer(input_t, hidden_states[i], cell_states[i])
                input_t = hidden_states[i]
            outputs.append(hidden_states[-1])

        # Stack outputs along the sequence dimension
        outputs = torch.stack(outputs, dim=1)
        return outputs


class xLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(xLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Gates: Forget, Input, Output
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Candidate cell state
        self.cell_candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, hidden_state, cell_state):
        # Concatenate input and hidden state
        combined = torch.cat([x, hidden_state], dim=1)

        # Forget gate
        forget = torch.sigmoid(self.forget_gate(combined))

        # Input gate
        input_gate = torch.sigmoid(self.input_gate(combined))

        # Candidate cell state
        cell_candidate = torch.tanh(self.cell_candidate(combined))

        # Update cell state
        cell_state = forget * cell_state + input_gate * cell_candidate

        # Output gate
        output_gate = torch.sigmoid(self.output_gate(combined))

        # Compute new hidden state
        hidden_state = output_gate * torch.tanh(cell_state)

        return hidden_state, cell_state
