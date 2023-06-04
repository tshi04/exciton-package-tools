import torch
from torch.autograd import Variable


class EncoderRNN(torch.nn.Module):
    """RNN Encoder

    Args:
        embedding_size (int, optional): embedding dimension. Defaults to 256.
        hidden_size (int, optional): hidden size. Defaults to 256.
        n_layers (int, optional): number of layers. Defaults to 2.
        rnn_network (str, optional): RNN Network. Defaults to "lstm".
        bidirectional (bool, optional): Bi-directional. Defaults to True.
        device (str, optional): device. Defaults to "cpu".
    """

    def __init__(
        self,
        embedding_size: int = 768,
        hidden_size: int = 768,
        n_layers: int = 2,
        rnn_network: str = "lstm",
        bidirectional: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        device = torch.device(device)
        self.hidden_size = hidden_size
        self.rnn_network = rnn_network
        self.n_layers = n_layers
        self.device = device
        self.bidirectional = bidirectional

        if rnn_network == "lstm":
            self.encoder = torch.nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirectional,
            ).to(device)
        elif rnn_network == "gru":
            self.encoder = torch.nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirectional,
            ).to(device)

    def forward(self, input_data):
        """pipeline.

        Args:
            input_data (_type_): input data.

        Returns:
            _type_: output data
        """
        n_dk = 1
        if self.bidirectional:
            n_dk = 2
        batch_size = input_data.size(0)

        h0_encoder = Variable(
            torch.zeros(n_dk * self.n_layers, batch_size, self.hidden_size)
        ).to(self.device)
        if self.rnn_network == "lstm":
            c0_encoder = Variable(
                torch.zeros(n_dk * self.n_layers, batch_size, self.hidden_size)
            ).to(self.device)
            # encoding
            hy_encoder, (ht_encoder, ct_encoder) = self.encoder(
                input_data, (h0_encoder, c0_encoder)
            )
            return hy_encoder, (ht_encoder, ct_encoder)
        elif self.rnn_network == "gru":
            # encoding
            hy_encoder, ht_encoder = self.encoder(input_data, h0_encoder)
            return hy_encoder, ht_encoder
