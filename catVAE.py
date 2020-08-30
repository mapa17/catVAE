from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import random

class catDataSet(Dataset):
    """Dataset loader converting everything to categorical"""

    def __init__(self, csv_file):
        """
        Reads csv file and converts all columns into categories
        """
        df = pd.read_csv(csv_file)
        self.data, self.encoding = self._encode_dataframe(df)

    @staticmethod
    def _encode_dataframe(df):
        # Convert all fields to categories
        catDF = df.astype('category')

        # Collect information about the categories and extract codes
        encoding = {}
        codes = np.zeros(shape=catDF.shape)
        for idx, col in enumerate(catDF.columns):
            encoding[col] = {
                'categories': catDF[col].cat.categories,
                'cardinality': len(catDF[col].cat.categories)
            }
            codes[:, idx] = catDF[col].cat.codes
        return codes, encoding

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        #return torch.from_numpy(self.data[idx], dtype=torch.long)
        return self.data[idx]

    def decoding(self, onehot):
        values = []
        columns = []
        N = onehot.shape[0]
        offset = 0
        for column, encoding in self.encoding.items():
            columns.append(column)
            # Argmax
            #value = encoding['categories'][np.argmax(onehot[:, offset:offset+encoding['cardinality']], axis=1)].values

            # Sample from the categories based on the probabilities for each cardinality
            prediction = onehot[:, offset:offset + encoding['cardinality']]
            value = np.ravel([random.choices(k=1, population=encoding['categories'], weights=p) for p in prediction])

            values.append(value)
            offset += encoding['cardinality']
        return pd.DataFrame({c: v for c, v in zip(columns, values)})

class catVAE(nn.Module):
    def __init__(self, encoding, embedding_dimensions=None, latent_dim=None, hidden_dim=None):

        super().__init__()

        if not embedding_dimensions:
            # Default embedding size = 10
            embedding_dimensions = [(col['cardinality'], 10) for col in encoding.values()]

        # The input dimension is based on the width of all the concatenated embeddings
        # In the output layer we have one node for each categorical (onehot)
        # So the output dim is the total cardinality!
        self.input_dim = sum([w for N, w in embedding_dimensions])
        self.cardinalities = [N for N, w in embedding_dimensions]
        self.output_dim = sum(self.cardinalities)

        if not hidden_dim:
            hidden_dim = self.input_dim // 4

        if not latent_dim:
            latent_dim = hidden_dim // 4
        self.latent_dim = latent_dim//2

        print(f'Creating network with sizes:')
        print(f'input {self.input_dim}, hidden: {hidden_dim}, latent: {self.latent_dim} ')

        # Embedding layers (Cardinality, embedding size)
        self.emb_layers = nn.ModuleList([nn.Embedding(N, w) for N, w in embedding_dimensions])

        # Define activation function
        #activation = torch.nn.LeakyReLU
        activation = torch.nn.ReLU

        # Encoder and Decoder
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, self.latent_dim)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            activation(),
            #nn.Linear(hidden_dim, self.input_dim),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid(),
        )

        # Dropout Layer
        #self.emb_dropout_layer = nn.Dropout(emb_dropout)

    def reparameterise(self, mu, logvar):
        #std = logvar.mul(0.5).exp_()
        #eps = std.data.new(std.size()).normal_()
        #return eps.mul(std).add_(mu)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Calculate the onehot format for later
        x_onehot = self.codes2onehot(x)

        # run through embedding layers
        x = [emb_layer(x[:, i]) for i, emb_layer in enumerate(self.emb_layers)]

        # Concatenate embeddings and resize independent of batch size
        x = torch.cat(x, 1).view(-1, self.input_dim)

        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar, x_onehot

    def codes2onehot(self, codes):
        # Perform onehot encoding
        # NOTE: Needs to be float for the loss function (for the BCE calculation)
        onehot = torch.cat([torch.nn.functional.one_hot(codes[:, i], num_classes=N)
                            for i, N in enumerate(self.cardinalities)],
                           axis=1).float()
        return onehot


def loss_function(x, x_hat, mu, logvar):
    #CCE = nn.functional.cross_entropy(x_hat, x, reduction='sum')
    BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(logvar + 1 - logvar.exp() - mu.pow(2))
    return BCE + KLD


## Build Model

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


BS = 1024
trainDS = catDataSet('data/us-census-income.csv')
# NOTE: num_workers must be 0 for debugging in pycharm
dataloader = DataLoader(trainDS, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
nBatchesPerEpoch = len(dataloader.dataset) // BS
print(f'Created dataloader with BS: {BS}, nPatchesPerEpoch: {nBatchesPerEpoch}')

model = catVAE(trainDS.encoding).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

## Training

def train(epochs):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            x = data.to(device, dtype=torch.long)
            optimizer.zero_grad()
            x_hat, mu, logvar, x_onehot = model(x)
            loss = loss_function(x_onehot, x_hat, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % (nBatchesPerEpoch // 5) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(dataloader.dataset)))

train(10)

## Inference
model.eval()
N = 20
z = torch.randn((N, model.latent_dim)).to(device)
sample = model.decoder(z).detach().to('cpu').numpy()
sample_df = trainDS.decoding(sample)
print(sample_df)