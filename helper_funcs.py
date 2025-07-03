from ase.io import read, iread

import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress
from collections import Counter


from aseMolec import pltProps as pp
from aseMolec import anaAtoms as aa


def get_all_thermo(tag,xyz=False):
	"""This needs to be exhanged for Ioans version in order to be consistent with
	zenodo.
	"""
	import glob
	thermo = {}
	if xyz:
		flist = [name for name in glob.glob(tag) if '.xyz' in name]
	else:
		flist = [name for name in glob.glob(tag) if '.thermo' in name]

	#for name in glob.glob(tag):
	#    print(name)
	#flist = z.read_dat_files(tag, ext='.thermo')
	for f in flist:
		#print(f)
		aux = os.path.basename(f)
		key = os.path.splitext(aux)[0]
		#print(aux,key)
		if xyz:
			thermo.update({key: read(f, ':')})
		else:
			thermo.update({key: pp.loadtxttag(f)})
	return thermo

def get_dens(tags, root_path, std=False):
	thermo = dict()
	for tag in tags:
		thermo[tag] = get_all_thermo(f'{root_path}{tag}/*.thermo')
	
	densities = dict()
	densities_std = dict()
	for i, k in enumerate(thermo):
		tag = tags[i]
		densities[tag] = {}
		densities_std[tag] = {}
		for traj_name, traj_data in thermo[k].items():
			# Collect data
			dens, dunit = traj_data['Density']['data'], traj_data['Density']['units']
			#densities[tag]['units'] = dunit
			densities[tag][traj_name] = np.mean(dens)
			densities_std[tag][traj_name] = np.std(dens)
	if std:
		return densities, densities_std, dunit
	else:
		return densities, dunit


def get_xyz(tag):
	import glob
	flist = [name for name in glob.glob(tag)]
	atoms = []
	for f in flist:
		#print(f)
		if 'xyz' in f:
			#print(f,'xyz')
			return read(f, ':')
			#atoms.append(read(f, ':'))
	return atoms

def RMSE(x,y):
	rmse = np.sqrt(np.mean((x-y)**2))
	rrmse = rmse/np.sqrt(np.mean((x-np.mean(x))**2))
	#se = (x-y)**2
	#rmse = np.mean(se)**0.5
	#rrmse = rmse/(np.mean((x-np.mean(x))**2))**0.5
	return rmse, rrmse

def get_colour(name):
	import tol_colors as tc
	cset = tc.tol_cset('bright')

	colmap = {
		'EMC (298$\,$K)':'blue',
		'EC:EMC (3:7) (298$\,$K)':'green',
		'EC:EMC (7:3) (298$\,$K)':'yellow',
		'EC (313$\,$K)':'red',
	}

	tmap = {
		'npt_000conEC_298K':'EMC (298$\,$K)',
		'npt_033conEC_298K':'EC:EMC (3:7) (298$\,$K)',
		'npt_066conEC_298K':'EC:EMC (7:3) (298$\,$K)',
		'npt_100conEC_313K':'EC (313$\,$K)',
	}
	try:
		return getattr(cset, colmap[name])
	except:
		return getattr(cset, colmap[tmap[name]])

def get_average_density(dataframe, col, value):
	tmap = {
		'npt_000conEC_298K':'EMC (298$\,$K)',
		'npt_033conEC_298K':'EC:EMC (3:7) (298$\,$K)',
		'npt_066conEC_298K':'EC:EMC (7:3) (298$\,$K)',
		'npt_100conEC_313K':'EC (313$\,$K)',
	}

	# Filter out the 'other' column (we either average sample or seed)
	col_set = ['Seed', 'Sample']; col_set.remove(col)
	filtered_dataframe = dataframe[dataframe[col_set[0]] == value]

	# Get statistics
	df_list = []
	for comp, comp_df in filtered_dataframe.groupby('Composition'):
		for train, train_df in comp_df.groupby('Train data'):
			mean = train_df['Avg. density'].mean()
			variance = train_df['Avg. density'].var()
			agg_col = train_df.drop(col, axis=1)
			agg_col['Avg. density'] = mean
			agg_col['Stdev.'] = variance**0.5
			df_list.append(agg_col)

	# Merge data frames and remove duplicate rows
	merged_df = pd.concat(df_list, ignore_index=True)
	clean_df = merged_df.drop_duplicates()
	return clean_df

def get_density(tags, dynamics_path, std=False):
	conmap = {
		'000conEC':'EMC (298$\,$K)',
		'033conEC':'EC:EMC (3:7) (298$\,$K)',
		'066conEC':'EC:EMC (7:3) (298$\,$K)',
		'100conEC':'EC (313$\,$K)',
	}
	
	# Compute average densities
	dens = dict()
	if std:
		dens, denstd, dunit = get_dens(tags, dynamics_path, std=True)
	else:
		dens, dunit = get_dens(tags, dynamics_path)

	# Collect average densities for all tags
	data = {
		'Train data':[],
		'Functional':[],
		'Seed':[],
		'Sample':[],
		'Composition':[],
		'Avg. density':[],
	}
	if std:
		data |= {'Std. density':[]}

	for tag, d in dens.items():
		data_set, functional, seed, dset_sample = tag.split('/')
		seed = int(seed[-1]); dset_sample = int(dset_sample[-1])

		# Add one instance per composition
		for i in range(4):
			data['Train data'].append(data_set)
			data['Functional'].append(functional)
			data['Seed'].append(seed)
			data['Sample'].append(dset_sample)
		
		# Add average densities for each composition
		for traj_name, avg_dens in d.items():
			comp = traj_name.split('_')[1]
			data['Composition'].append(conmap[comp])
			data['Avg. density'].append(avg_dens)
			if std:
				data['Std. density'].append(denstd[tag][traj_name])
	#for key,val in data.items():
	#    print(key, len(val))
	return pd.DataFrame(data), dunit


def diffusion_coefficient(time, msd):
	"""Determines the diffusion coefficient in three dimensions
	from the slope of the MSD-curve.
	"""
	res = linregress(time,msd)
	diffusion_coeff = res.slope/6 #final unit: A^2/fs
	diffusion_coeff *= 1e-5 #final unit: m^2/s
	diffusion_err = res.stderr*1e-5 #final unit: m^2/s
	##D, offs = np.polyfit(t, msd, deg=1)
	##D /= 6 #final unit: A^2/ps
	##D *= 100 #final unit: 10^-10 m^2/s or 10^-6 cm^2/s

	from scipy.stats import t
	tinv = lambda p, df: abs(t.ppf(p/2, df))
	ts = tinv(0.05, len(time)-2)
	return diffusion_coeff, ts*diffusion_err, res.rvalue**2


def get_slope(time, msd):
	res = linregress(time,msd)
	return res.slope, res.intercept, res.stderr, res.rvalue**2

def get_start_index(time, msd, threshold=False, min_steps=0):
	if threshold:
		print(f'Using slope error thresold of: {threshold}')
		for i, t in enumerate(time):
			if i != 0:
				slope, intercept, err, r2 = get_slope(np.log(time[i:]),np.log(msd[i:]))
				if (abs(slope-1) < threshold) and (i+1 > min_steps):
					idx = i
					break
				else:
					idx = None
	else:
		slopes = []
		for i, t in enumerate(time):
			if i != 0:
				slope, intercept, err, r2 = get_slope(np.log(time[i:]),np.log(msd[i:]))
				slopes.append(abs(slope-1))
		slopes = np.array(slopes)
		slopes[slopes == np.nan] = 100
		#idx = np.argmin(slopes[~np.isnan(slopes)])
		idx = np.argmin(slopes)
	return idx, r2


def collect_comp(db):
	buf = {}
	for at in db:
		#at.info['Nmols'] 
		if at.info['Nmols'] in buf:
			try:
				buf[at.info['Nmols']] += [at.info['Comp']]
			except:
				buf[at.info['Nmols']] += [at.info['config_type']]
		else:
			try:
				buf[at.info['Nmols']] = [at.info['Comp']]
			except:
				buf[at.info['Nmols']] = [at.info['config_type']]

	comp = {}
	for b in buf:
		comp[b] = dict(Counter(buf[b]))
	return comp

def wrap_labels(labels):
	wrapped_labels = []
	for label in labels:
		parts = label.split(':')
		for i in range(1, len(parts) - 1, 2):
			parts[i] += '\n'
		wrapped_labels.append(':'.join(parts))
	return wrapped_labels

def flatten_comp(comp_dict):
	flattened_comp_dict = {}
	for csize, dist in comp_dict.items():
		#print(csize)
		#print(dist)
		tmp = {
			'EMC':0,
			'EC':0,
			'EC and EMC and other':0,
			'EC or EMC and other':0,
			'Other':0,
		}
		for comp, count in dist.items():
			#print(comp, count)
			part = comp.split(':')
			part = [p.split('(')[0] for p in part]
			part = [''.join([i for i in p if not i.isdigit()]) for p in part]
			#print(part)
			if ('EC' not in part) and ('EMC' not in part):
				#print('not EC not EMC', part)
				tmp['Other'] += count
			
			elif ('EC' in part) and ('EMC' in part):
				#print('not EC not EMC', part)
				tmp['EC and EMC and other'] += count
			
			elif ('EC' in part) and ('EMC' not in part):
				if len(set(part)) == 1:
					#print('only EC', part)
					tmp['EC'] += count
				else:
					#print('EC and other', part)
					tmp['EC or EMC and other'] += count

			elif ('EC' not in part) and ('EMC' in part):
				if len(set(part)) == 1:
					#print('only EMC', part)
					tmp['EMC'] += count
				else:
					#print('EMC and other', part)
					tmp['EC or EMC and other'] += count
			else:
				tmp['EC or EMC and other'] += count
			
		flattened_comp_dict[csize] = tmp
	df = pd.DataFrame(flattened_comp_dict).T
	return df.sort_index(inplace=False)

def rvalue_formatting(x):
    if x < 0.8:
        return 'background-color: red'
    elif x < 0.9:
        return 'background-color: orange'
    elif x < 0.95:
        return 'background-color: yellow'
    else:
        return None



def expectation(arr):
    cluster_sizes = np.arange(1,len(arr)+1)
    N = sum(arr)
    p = arr/N
    print(p*100)
    EX = sum([x*pi for x,pi in zip(cluster_sizes,p)])
    EX2 = sum([x**2*pi for x,pi in zip(cluster_sizes,p)])
    return EX, EX2-EX**2

def reduce_df(df, drop_list):
    df = df.drop(drop_list, axis=1)

    data = {}
    for i, (name, group) in enumerate(df.groupby('Composition')):
        dens = group['Avg. density'].mean()
        dens_err = (group['Std. density']**2).sum()**0.5/len(group)

        diff = group['Diffusion coefficient'].mean()
        diff_err = (group['Diff. fit error']**2).sum()**0.5/len(group)

        diff_start = group['Diff. start time /ns'].mean()
        tmp = {'Composition':name}
        for col in group.columns:
            try:
                if ('Std' in col) or ('error' in col):
                    tmp[col] = (group[col]**2).sum()**0.5/len(group)
                else:
                    tmp[col] = group[col].mean()
            except:
                pass
        data[i] = tmp
    dframe = pd.DataFrame.from_dict(data, orient='index')
    return dframe



def get_mol_positions(atoms):
	positions = []
	mol_set = set(atoms.arrays['molID'])
	for ID in mol_set:
		mol = atoms[atoms.arrays['molID'] == ID]
		com = mol.get_center_of_mass()
		positions.append(com)
	positions = np.array(positions)
	return positions, len(mol_set)

def mol_rdf(atoms, rcut, bins, return_num_mols=False):
	positions, num_mols = get_mol_positions(atoms)
	box_length = atoms.get_volume()**(1/3)
	if rcut is None: rcut=box_length
	for i, p in enumerate(positions):
		# PBC
		delta = positions[i+1:]-positions[i]
		delta -= box_length*np.round(delta/box_length)
		
		# Get counts
		dists = np.linalg.norm(delta, axis=1)
		s = sorted(dists[dists <= rcut])[1:]
		if i == 0:
			counts, bins = np.histogram(s, bins, (0,rcut))
		else:
			counts += np.histogram(s, bins, (0,rcut))[0]
	if return_num_mols:
		return counts, bins, num_mols
	else:
		return counts, bins

def compute_rdf(traj, rcut=None, bins=100):
	for i, atoms in enumerate(traj):
		if i == 0:
			counts, bins, nmols= mol_rdf(atoms, rcut=rcut, bins=bins, return_num_mols=True)			
			volume = atoms.get_volume()
			box_length = volume**(1/3)
			r = 0.5*(bins[1:]+bins[:-1])
			shell_volume = 4/3*np.pi*(bins[1:]**3-bins[:-1]**3)
			number_density = nmols/volume
			ideal_gas = number_density*shell_volume
		else:
			counts += mol_rdf(atoms, rcut=rcut, bins=bins)[0]
	g = counts/(ideal_gas*nmols)
	return g, r