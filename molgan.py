
from solver import Solver
import json


class Molgan():
	def __init__(self, config_dict='commandline_args.json'):
		with open(config_dict, 'r') as f:
			config = json.load(f)
		print(config['mol_data_dir'])
		self.model = Solver(config)


	def train(self):
		self.model.train()
	
	def test(self):
		self.model.test()


if __name__ == '__main__':
    #Debug
    molgan = Molgan()
    molgan.train()
    #molgan.train()
