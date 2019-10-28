
from solver import Solver
import json


class Molgan():
	def __init__(self, config_dict='commandline_args.json'):
		"""
		A wrapper class for the MolGan model developed in 
		"MolGan: An implicit generative model for small molecular graphs"
		by De Cao and Kipf.

		This model adapts GAN, operate directly on graph-strucutred data, 
		and has a reinforcement learning objective using DDPG 
		(deep deterministic policy gradient)

		"""
		with open(config_dict, 'r') as f:
			config = json.load(f)
		print(config['mol_data_dir'])
		self.model = Solver(config)


	def train(self):
		self.model.train()
	
	def test(self):
		self.model.test()

	def save(self):
		self.model.save()


if __name__ == '__main__':
    #Debug
    molgan = Molgan()
    #molgan.train()
    molgan.test()
