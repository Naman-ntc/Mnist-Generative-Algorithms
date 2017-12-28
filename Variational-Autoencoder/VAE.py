import torch
import numpy as np
import torch.nn autograds nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

torch.set_default_tensor_type('torch.cuda.FloatTensor')
dtype = torch.cuda.FloatTensor

latent_dim = 2

encoder = nn.Sequential(
	nn.Linear(28*28, 512),
	nn.LeakyReLU(),
	nn.Linear(512, 128),
	nn.ReLU(),
	nn.Linear(128, 24),
	nn.ReLU(),		
)
mean_network = nn.Sequential(
		nn.Linear(24,latent_dim)
)
std_network = nn.Sequential(
	nn.Linear(24,latent_dim)
)
decoder = nn.Sequential(
	nn.Linear(latent_dim, 24),
	nn.ReLU(),
	nn.Linear(24, 128),
	nn.ReLU(),
	nn.Linear(128, 512),
	nn.LeakyReLU(),
	nn.Linear(512, 28*28),
	nn.Sigmoid(),       # compress to a range (0, 1)
)


class VAE():

	def Encoder(self,z):
		encoded_prime = encoder(z)
		mean = 	mean_network(encoded_prime)
		std = std_network(encoded_prime)
		return (mean,std)

	def Decoder(self,z):
		decoded = decoder(z)
		return decoded

	def Loss(self,z):
		mean,std = self.Encoder(z)
		# print("a")	
		# print(mean)
		# print("b")
		# print(std)
		n = z.size()[0]
		var = std*Variable(torch.randn(n,latent_dim)).type(dtype)  
		latent_var = mean.expand_as(var) + var # Sampled from  latent distribution
		final = self.Decoder(latent_var)
		generated_loss = nn.functional.binary_cross_entropy(z,final)
		#print(type(final))
		#print(type(z))
		#generated_loss = loss_fn(final,z)
		latent_loss = torch.mean(0.5 * torch.sum(mean**2 - 2 + var + torch.log(var)))
		loss = generated_loss + latent_loss
		#print("a")
		#print(latent_loss)
		#print("b")
		#print(generated_loss.data)
		return loss

batch_size = 100

vae = VAE()
encoder.cuda()
decoder.cuda()
mean_network.cuda()
std_network.cuda()
params = list(encoder.parameters()) + list(decoder.parameters()) + list(mean_network.parameters()) + list(std_network.parameters())
optimizer = optim.SGD(params,lr=1e-4)


#---------------------------------x--------------------------------x--------------------------------#

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../datasets', train=True, download=True,
					transform=transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307,), (0.3081,))
					])),
	batch_size=batch_size, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
# 	datasets.MNIST('../datasets', train=False, transform=transforms.Compose([
# 					transforms.ToTensor(),
# 					transforms.Normalize((0.1307,), (0.3081,))
# 					])),
# 	batch_size=test_batch_size, shuffle=True)

#---------------------------------x--------------------------------x--------------------------------#
for i in range(10):
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data).type(dtype), Variable(target).type(dtype)
		optimizer.zero_grad()
		#print(data.size())
		loss = vae.Loss(data.view(batch_size,-1))
		loss.backward()
		optimizer.step()
		#print(loss)
