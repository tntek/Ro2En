from torch import nn


class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		# self.encoder = nn.Sequential(nn.Linear(6000, 512),nn.PReLU(),nn.Linear(512, 256),nn.PReLU(),nn.Linear(256, 128),nn.PReLU(),nn.Linear(128, 28))
		self.encoder = nn.Sequential(nn.Linear(6000, 786),nn.PReLU(),nn.Linear(786, 512),nn.PReLU(),nn.Linear(512, 256),nn.PReLU(),nn.Linear(256, 60))
	def forward(self, x):
		x = self.encoder(x)
		return x

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		# self.decoder = nn.Sequential(nn.Linear(28, 128),nn.PReLU(),nn.Linear(128, 256),nn.PReLU(),nn.Linear(256, 512),nn.PReLU(),nn.Linear(512, 6000))
		self.decoder = nn.Sequential(nn.Linear(60, 256),nn.PReLU(),nn.Linear(256, 512),nn.PReLU(),nn.Linear(512, 786),nn.PReLU(),nn.Linear(786, 6000))
	def forward(self, x):
		x = self.decoder(x)
		return x


