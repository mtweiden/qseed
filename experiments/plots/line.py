import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hmean as mean
from csv import reader


def extract_name_size(name : str | list[str]) -> tuple[str,int] | list[tuple[str,int]]:
	if isinstance(name, list):
		names_sizes = []
		for x in name:
			n, s = x.split('.')[0].split('_')
			names_sizes.append((n, int(s)))
		return names_sizes
	else:
		name, size = name.split('.')[0].split('_')
		return name, int(size)

def get_color(name : str) -> str:
	color_maps = {
		'add' : 'red',
		'heisenberg' : 'darkorange',
		'hhl' : 'gold',
		'hubbard' : 'forestgreen',
		'mult' : 'lightseagreen',
		'qae' : 'cyan',
		'qft' : 'dodgerblue',
		'qml' : 'navy',
		'qpe' : 'mediumpurple',
		'shor' : 'purple',
		'tfim' : 'magenta',
		'vqe' : 'springgreen',
	}
	return color_maps[name]

#file_names=['add_17','add_41','add_65','heisenberg_3','heisenberg_4','heisenberg_8','heisenberg_16','heisenberg_32','heisenberg_64','hubbard_4','hubbard_8','hubbard_18','hubbard_50','mult_8','mult_16','mult_32','mult_64','qae_11','qae_33','qae_65','qft_3','qft_4','qft_8','qft_16','qft_32','qft_64','qml_4','qml_25','qml_60','shor_16','shor_32','shor_64','tfim_3','tfim_4','tfim_8','tfim_16','tfim_32','tfim_64']
file_names=['hubbard_4','hubbard_8','hubbard_18','hubbard_50','qml_4','qml_25','qml_60','shor_16','shor_32','shor_64']
#file_names=['heisenberg_3','heisenberg_4','heisenberg_8','heisenberg_16','heisenberg_32','heisenberg_64','mult_8','mult_16','mult_32','mult_64','qft_3','qft_4','qft_8','qft_16','qft_32','qft_64','tfim_3','tfim_4','tfim_8','tfim_16','tfim_32','tfim_64']

if __name__ == '__main__':
	
	qseed_time, qsearch_time   = {}, {}
	
	with open('stats.txt', 'r') as csvfile:
		csvreader = reader(csvfile)
		for name, algo, cx, u3, calls, time in csvreader:
			cx, u3, calls, time = int(cx), int(u3), float(calls), float(time)

			circ_name, size = name.split('_')[0], int(name.split('_')[1])
			
			if name not in file_names:
				continue

			if circ_name not in qseed_time:
				qseed_time[circ_name] = []
			if circ_name not in qsearch_time:
				qsearch_time[circ_name] = []

			if 'qseed' in algo:
				qseed_time[circ_name].append((size, time))
			if 'qsearch' in algo:
				qsearch_time[circ_name].append((size, time))

	fig, ax = plt.subplots()
	x_axis = np.arange(0, 65)

	for name in qseed_time.keys():
		sizes   = [s for (s,t) in qseed_time[name]]
		qseed   = [t for (s,t) in qseed_time[name]]
		qsearch = [t for (s,t) in qsearch_time[name]]

		diff = [a-b for (a,b) in zip(qsearch, qseed)]
		ratio = [a/b for (a,b) in zip(qsearch, qseed)]
		print(name)
		print(f'sizes:  {str(sizes)}')
		print(f'diffs:  {str(diff)}')
		print(f'ratios: {str(ratio)}')
	
		#ax.set_yscale('log')
		#ax.plot(sizes, diff, color=get_color(name), label=f'{name}')
		ax.set_ylim(0,4)
		ax.hlines(1.0, xmin=0, xmax=64, linestyle='--', color='grey')
		ax.plot(sizes, ratio, color=get_color(name), label=f'{name}')

		ax.set_xlabel('Circuit Width (qubits)', fontsize=16)
		ax.set_ylabel('Speed Up', fontsize=16)
	plt.legend()
	plt.show()

