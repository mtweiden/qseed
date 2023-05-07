import re
import pickle
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity

def get_calls(call_str: str) -> list[int]:
	return [int(x) for x in re.findall(r'\d+', call_str)]


def load_data() -> tuple[list]:
	qseed_files    = [x for x in listdir('../calls/') if 'qseed' in x]
	qsearch_files  = [x for x in listdir('../calls/') if 'qsearch' in x]
	random_files   = [x for x in listdir('../calls/') if 'random' in x]

	qseed_calls, qsearch_calls, random_calls = [], [], []
	qseed_cnots, qsearch_cnots, random_cnots = [], [], []
	for file in qseed_files:
		with open(f'../cnots/{file}.pickle', 'rb') as f:
			cnots = pickle.load(f)
		with open(f'../calls/{file}', 'r') as f:
			calls = get_calls(f.readlines()[0])
		# Because calls is appended to, not overwritten
		calls = calls[:len(cnots)]
		qseed_cnots += cnots
		qseed_calls += calls

	for file in qsearch_files:
		with open(f'../cnots/{file}.pickle', 'rb') as f:
			cnots = pickle.load(f)
		with open(f'../calls/{file}', 'r') as f:
			calls = get_calls(f.readlines()[0])
		# Because calls is appended to, not overwritten
		calls = [c-1 for c in calls[:len(cnots)]]
		qsearch_cnots += cnots
		qsearch_calls += calls

	for file in random_files:
		with open(f'../cnots/{file}.pickle', 'rb') as f:
			cnots = pickle.load(f)
		with open(f'../calls/{file}', 'r') as f:
			calls = get_calls(f.readlines()[0])
		# Because calls is appended to, not overwritten
		calls = [c-1 for c in calls[:len(cnots)]]
		random_cnots += cnots
		random_calls += calls

	return qsearch_calls, qseed_calls, random_calls, qsearch_cnots, qseed_cnots, random_cnots

	
def filter_outliers(data : list) -> list:
	threshold = 50
	np.random.shuffle(data)
	data = [d for d in data if d <= threshold]
	return [d if d > 0 else 1 for d in data]


def make_ridge(dataframe):
	dataframe.plot.density()
	plt.show()
	dataframe.hist()
	plt.show()

if __name__ == '__main__':

	qsearch_calls, qseed_calls, random_calls, qsearch_cnots, qseed_cnots, random_cnots = load_data()

	qseed_calls   = filter_outliers(qseed_calls)
	qsearch_calls = filter_outliers(qsearch_calls)
	random_calls  = filter_outliers(random_calls)
	min_calls = min([len(qseed_calls), len(qsearch_calls), len(random_calls)])

	qseed_calls = qseed_calls[:min_calls]
	qsearch_calls = qsearch_calls[:min_calls]
	random_calls = random_calls[:min_calls]
	
	
	#print(f'QSeed:   {np.mean():>0.2f}')
	#print(f'QSearch: {np.mean():>0.2f}')
	#print(f'Random:  {np.mean():>0.2f}')

	#qseed_cnots = filter_outliers(qseed_cnots)
	#qsearch_cnots = filter_outliers(qsearch_cnots)
	#random_cnots = filter_outliers(random_cnots)

	#qseed_calls = filter_outliers(qseed_calls)
	#qsearch_calls = filter_outliers(qsearch_calls)
	#random_calls = filter_outliers(random_calls)
	#for h_cx, , d_cx, r_cx, h_calls d_calls, r_calls in zip(qsearch_cnots, qseed_cnots, random_cnots, qsearch_calls, qseed_calls, random_calls):
	
	#calls = qsearch_calls + qseed_calls + random_calls
	#types = ['QSearch'] * len(qsearch_calls) + ['QSeed'] * len(qseed_calls) + ['Random'] * len(random_calls)
	cnots = qsearch_cnots + qseed_cnots + random_cnots
	types = ['QSearch'] * len(qsearch_cnots) + ['QSeed'] * len(qseed_cnots) + ['Random'] * len(random_cnots)
	data = {'cnots': cnots, 'synth': types}
	dataframe = pd.DataFrame(data)

	
	import matplotlib as mpl
	import matplotlib.gridspec as grid_spec


	#threshold = 20
	threshold = 2
	types = ['QSearch','QSeed','Random']

	gc = (grid_spec.GridSpec(len(types),1))
	fig = plt.figure(figsize=(8,6))

	ax_objs = []

	for i,synth in enumerate(types):
		ax_objs.append(fig.add_subplot(gc[i:i+1, 0:]))

		plot = (
			#dataframe[dataframe.synth == synth].calls.plot.hist(ax=ax_objs[-1], lw=0.5, bins=threshold)
			dataframe[dataframe.synth == synth].cnots.plot.kde(ax=ax_objs[-1], lw=1, color='#f0f0f0')
		)
		x = plot.get_children()[0]._x
		y = plot.get_children()[0]._y
		ax_objs[-1].fill_between(x,y)

		#ax_objs[-1].set_xlim(0,threshold+1)
		ax_objs[-1].set_xlim(0,2.5)

		#if i == 0:
		#	ax_objs[-1].set_ylim(0,1)
		#else:
		#	ax_objs[-1].set_ylim(0,2.75)
			
		if i == len(types)-1:
			ax_objs[-1].set_ylim(0,10)
		else:
			ax_objs[-1].set_ylim(0,20)
		

		rect = ax_objs[-1].patch
		rect.set_alpha(0)
		
		ax_objs[-1].set_yticks([])
		ax_objs[-1].set_yticklabels([])
		ax_objs[-1].set_ylabel('')
		

		if i == len(types)-1:
			ax_objs[-1].set_xlabel('Number of Instantiation Calls', fontsize=16)
			ax_objs[-1].set_xticks([x for x in range(threshold+1)])
		else:
			ax_objs[-1].set_xticks([])
			ax_objs[-1].set_xticklabels([])

		spines = ['top','right','left','bottom']
		for s in spines:
			ax_objs[-1].spines[s].set_visible(False)

	gc.update(hspace=-0.4)
	plt.tight_layout()
	plt.show()
	
