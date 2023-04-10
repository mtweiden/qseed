import re
from os import listdir
import matplotlib.pyplot as plt

def get_calls(call_str : str) -> list[int]:
	return [int(x) for x in re.findall(r'\d+', call_str)]

if __name__ == '__main__':
	
	call_files = [f'../calls/{x}' for x in listdir('../calls/')]
	
	all_calls = []
	for call_file in call_files:
		with open(call_file, 'r') as f:
			all_calls += get_calls(f.readlines()[0])
	
	vmin = min([c for c in all_calls])
	vmax = max([c for c in all_calls])
	plt.hist(all_calls, bins=range(vmin, vmax+1), align='mid')
	plt.xticks(range(vmin,vmax+1))
	plt.show()
