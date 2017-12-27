import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class VAE(nn.Module):
	
	def build_network(self):
		self.latent_dim = 2
		self.encoder = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Linear(128, 24),
			nn.ReLU(),
			)
		self.mean_network = nn.Sequential(
			nn.Linear(24,self.latent_dim)
			)
		self.std_network = nn.Sequential(
			nn.Linear(24,self.latent_dim)
			)
		self.decoder = nn.Sequential(
			nn.Linear(self.latent_dim, 24),
			nn.ReLU(),
			nn.Linear(24, 128),
			nn.ReLU(),
			nn.Linear(128, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 28*28),
			nn.Sigmoid(),       # compress to a range (0, 1)
			)


	def Encoder(self,z):
		encoded_prime = self.encoder(z)
		mean = 	self.mean_network(encoded_prime)
		std = self.std_network(encoded_prime)
		return (mean,std)

	def Decoder(self,z):
		decoded = decoder(z)
		return decoded

	def Loss(self,z):
		mean,std = self.Encoder(z)
		n = z.size()[0]
		var = std*torch.randn()  
		latent_var = mean.expand_as(var) + var # Sampled from  latent distribution
		final = Decoder(latent_var)
		loss_fn = nn.BCELoss()
		generated_loss = loss_fn(z,final)
		latent_loss = torch.mean(0.5 * torch.sum(mean**2 - 2 + var + torch.log(var)))
		loss = generated_loss + latent_loss
		return loss

batch_size = 100

vae = VAE()
optimizer = torch.optim.SGD(vae.parameters(),lr=1e-4)


#---------------------------------x--------------------------------x--------------------------------#

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../datasets', train=True, download=True,
					transform=transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307,), (0.3081,))
					])),
	batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../datasets', train=False, transform=transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307,), (0.3081,))
					])),
	batch_size=test_batch_size, shuffle=True, **kwargs)

#---------------------------------x--------------------------------x--------------------------------#
for i in range epoch:
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		loss = vae.Loss()
		loss.backward()
		optimizer.step()
		#print(loss)