import re
import pickle
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def get_calls(call_str : str) -> list[int]:
	return [int(x) for x in re.findall(r'\d+', call_str)]
	#return [float(x) for x in re.findall(r'\d+.\d+',call_str)]

def exp_val(calls : list[float]) -> float:
	c_set = list(set(calls))
	probs = [calls.count(c)/len(calls) for c in c_set]
	expectation = 0.0
	for c, p in zip(c_set, probs):
		expectation += c * p
	return expectation

def make_violin(dataframe):
	fig, ax = plt.subplots(sharey=True)
	ax = sns.violinplot(
		#x=['Synthesis Type'], 
		data=dataframe,
		ax=ax, 
		scale='width',
		split=True
	)
	plt.show()
	
def filter_outliers(data : list) -> list:
	threshold = 1
	cutoff = 50
	num_samples = 301745
	data = [round(d,3) for d in data]
	data_set = set(data)
	data_set = [d for d in data_set if data.count(d) >= threshold]
	data = [d for d in data if d in data_set]
	data = [d for d in data if d <= cutoff]
	
	np.random.shuffle(data)
	return data[:num_samples]


if __name__ == '__main__':

	qseed_files    = [f'../calls/{x}' for x in listdir('../calls/') if 'qseed' in x]
	qsearch_files  = [f'../calls/{x}' for x in listdir('../calls/') if 'qsearch' in x]
	random_files   = [f'../calls/{x}' for x in listdir('../calls/') if 'random' in x]
	
	qseed_calls, qsearch_calls, random_calls = [], [], []

	for file in qseed_files:
		with open(file, 'r') as f:
			qseed_calls += get_calls(f.readlines()[0])
	for file in qsearch_files:
		with open(file, 'r') as f:
			qsearch_calls += get_calls(f.readlines()[0])
	for file in random_files:
		with open(file, 'r') as f:
			random_calls += get_calls(f.readlines()[0])
	qsearch_calls = [qc-1 for qc in qsearch_calls] # Don't count empty circuit calls
	random_calls  = [rc-1 for rc in random_calls] # Don't count empty circuit calls

	max_calls = max([max(qseed_calls), max(qsearch_calls), max(random_calls)])
	min_calls = min([min(qseed_calls), min(qsearch_calls), min(random_calls)])
	#print(max_calls)
	#print(min_calls)

	qseed_expectation   = np.median(qseed_calls) #exp_val(qseed_calls)
	qsearch_expectation = np.median(qsearch_calls) #exp_val(qsearch_calls)
	random_expectation  = np.median(random_calls) #exp_val(random_calls)

	print(f'QSeed:   {qseed_expectation:>0.2f}')
	print(f'QSearch: {qsearch_expectation:>0.2f}')
	print(f'Random:  {random_expectation:>0.2f}')

	qseed   = [x if x > 0 else 1 for x in qseed_calls   ]
	qsearch = [x if x > 0 else 1 for x in qsearch_calls ]
	random  = [x if x > 0 else 1 for x in random_calls  ]

	qseed = filter_outliers(qseed)
	qsearch = filter_outliers(qsearch)
	random = filter_outliers(random)

	data = {'QSearch':qsearch, 'QSeed':qseed, 'Random':random}

	#dataframe = pd.DataFrame([qsearch, qseed, random], rows='Synthesis Type')
	dataframe = pd.DataFrame(data)

	#qseed   = [np.log(x) for x in qseed  ]
	#qsearch = [np.log(x) for x in qsearch]	
	#random  = [np.log(x) for x in random ]
	#max_calls = max([max(qseed), max(qsearch), max(random)])
	#min_calls = min([min(qseed), min(qsearch), min(random)])
	#print(max_calls)
	#print(min_calls)

	make_violin(dataframe)
	#make_violin(qseed_calls, qsearch_calls, random_calls)

	#fig,ax = plt.subplots()
	#ax.hist(qseed_calls)
	#plt.show()
