import re
import pickle
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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

def make_violin(dataframe):
	fig, ax = plt.subplots(sharey=True)

	white='#e0ecf4'
	blue='#9ebcda'
	purple='#8856a7'

	pal={'QSearch':blue, 'QSeed':purple, 'Random':white}
	pal = [blue, purple, white]
	sns.set_palette(pal)

	sns.violinplot(
		#data=dataframe,
		x=dataframe['Calls'],
		y=dataframe['Synthesis Type'],
		scale='count',
		width=0.8,
		#inner='quartile',
		orient='h',
		#color=purple,
		#pallete=[blue,purple,white],
		pallete=pal,
		linewidth=2,
		fontsize=14,
		
	)
	ax.set_xlabel('Number of Instantiation Calls per Partition', fontsize=16)
	ax.set_ylabel('')
	ax.tick_params(axis='y', labelsize=14)
	plt.show()

def make_calls_violin(data):
	fig, ax = plt.subplots(sharey=True)
	fig.set_size_inches((8,6))

	white='#e0ecf4'
	blue='#9ebcda'
	purple='#8856a7'

	pal={'QSearch':blue, 'QSeed':purple, 'Random':white}
	pal = [white, white, white]

	parts = ax.violinplot(
		dataset=data,
		vert=False,
		showmeans=False,
		widths=0.8,
		showextrema=False,
		#showmedians=True,
		#quantiles=[[0.25,0.5,0.75]*3],
		#quantiles=[0.25,0.5,0.75]*3,
		bw_method='scott',
	)
	for i,pc in enumerate(parts['bodies']):
		if i == 0:
			pc.set_facecolor(blue)
		if i == 1:
			pc.set_facecolor(purple)
		if i == 2:
			pc.set_facecolor(white)
		pc.set_edgecolor('black')
		pc.set_alpha(0.8)


	quartile1, medians, quartile3 = np.percentile(data, [25,50,75],axis=1)
	inds = np.arange(1,len(medians)+1)
	ax.scatter(medians,inds, marker='o',color='white',s=30,zorder=3,edgecolor='black')

	ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)

	xticks = [_ for _ in range(1,20+1)]
	#xlabels = [x if x % 2 else '' for x in xticks]
	xlabels = xticks
	yticks = [1, 2, 3]
	ylabels = ['QSearch', 'QSeed', 'Random']

	ax.set_xlim(0,20+1)
	ax.set_xlabel('Number of Instantiation Calls', fontsize=16)
	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabels, fontsize=12)
	ax.set_yticks(yticks)
	ax.set_yticklabels(ylabels, fontsize=14)
	plt.show()

def make_cnots_violin(data):
	fig, ax = plt.subplots(sharey=True)
	fig.set_size_inches((8,6))

	white='#e0ecf4'
	blue='#9ebcda'
	purple='#8856a7'

	pal={'QSearch':blue, 'QSeed':purple, 'Random':white}
	#pal = [blue, purple, white]
	pal = [white, white, white]

	parts = ax.violinplot(
		dataset=data,
		vert=False,
		showmeans=False,
		widths=0.8,
		showextrema=False,
		#showmedians=True,
		#quantiles=[[0.25,0.5,0.75]*3],
		#quantiles=[0.25,0.5,0.75]*3,
		bw_method='scott',
	)
	for i,pc in enumerate(parts['bodies']):
		if i == 0:
			pc.set_facecolor(white)
		if i == 1:
			pc.set_facecolor(white)
		if i == 2:
			pc.set_facecolor(white)
		pc.set_edgecolor('black')
		pc.set_alpha(0.8)


	quartile1, medians, quartile3 = np.percentile(data, [25,50,75],axis=1)
	inds = np.arange(1,len(medians)+1)
	ax.scatter(medians,inds, marker='o',color='white',s=30,zorder=3,edgecolor='black')

	ax.hlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)


	thresh = 2
	gran = 0.25
	xticks = [gran*x for x in range(0,int((thresh+1)/gran)+1)]
	#xlabels = [x if x % 2 else '' for x in xticks]
	xlabels = xticks
	yticks = [1, 2, 3]
	ylabels = ['QSearch', 'QSeed', 'Random']

	ax.set_xlim(-0.1,thresh+1)
	ax.set_xlabel('Relative CNOT Gate Count per Partition', fontsize=16)
	ax.set_xticks(xticks)
	ax.set_xticklabels(xlabels, fontsize=12)
	ax.set_yticks(yticks)
	ax.set_yticklabels(ylabels, fontsize=14)
	plt.show()
	
def filter_outliers(data : list) -> list:
	threshold = 50
	np.random.shuffle(data)
	data = [d for d in data if d <= threshold]
	return [d if d > 0 else 1 for d in data]

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
	
	calls = qsearch_calls + qseed_calls + random_calls
	types = ['QSearch'] * len(qsearch_calls) + ['QSeed'] * len(qseed_calls) + ['Random'] * len(random_calls)
	data = {'Calls': calls, 'Synthesis Type': types}
	dataframe = pd.DataFrame(data)
	#make_violin(dataframe)

	data = [qsearch_calls, qseed_calls, random_calls]
	make_calls_violin(data)
	#data = [qsearch_cnots, qseed_cnots, random_cnots]
	#make_cnots_violin(data)

	#make_violin(qsearch_calls, qseed_calls, random_calls)
