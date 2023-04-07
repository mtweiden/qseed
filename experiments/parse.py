import re
from os import listdir
from argparse import ArgumentParser

def find_stats(lines : list[str]) -> tuple[int, int, int, float]:
	# Collect: 
	#  cx count
	#  u3 count
	#  instantiation calls
	#  compilation time
	cx_pat = r'CNOTGate Count'
	u3_pat = r'U3Gate Count'
	inst_pat = r'Instantiation Calls'
	time_pat = r'Optimization time'

	int_str = r'\d+'
	float_str = r'\d+\.\d+'

	cx, u3, inst, time = 0, 0, 0, 0.0

	for line in lines:
		if re.findall(cx_pat, line):
			cx += int(re.findall(int_str, line)[0])

		elif re.findall(u3_pat, line):
			u3 += int(re.findall(int_str, line)[0])

		elif re.findall(inst_pat, line):
			inst += int(re.findall(int_str, line)[0])

		elif re.findall(time_pat, line):
			time += float(re.findall(float_str, line)[0])
			break # For circuits run multiple times

	return cx, u3, inst, time

def sort_files(files : list[str]) -> list[str]:
	names = sorted(list(set([x.split('-')[-1].split('_')[0] for x in files])))
	sorted_files = []

	for name in names:
		sorted_files += sorted(
			[x for x in files if name in x],
			key=lambda x: int(re.findall(r'\d+', x)[0])
		)
	return sorted_files

if __name__ == '__main__':
	#parser = ArgumentParser()
	#parser.add_argument('log_file')
	#args = parser.parse_args()

	files = sort_files(listdir('logs/qseed'))

	for file in files:
		name = file.split('.')[0]
		stats_file = 'stats.txt'
		for algo in ['qseed', 'qsearch']:
			with open(f'logs/{algo}/{name}.log', 'r') as f:
				cx, u3, inst, time = find_stats(f.readlines())
			with open(stats_file, 'a') as f:
				f.write(f'{name}, {algo}, {cx}, {u3}, {inst}, {time}\n')
