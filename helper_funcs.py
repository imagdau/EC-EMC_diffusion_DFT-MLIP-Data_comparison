from ase.io import read, iread

import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress

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
  ##D, offs = np.polyfit(t, msd, deg=1)
  ##D /= 6 #final unit: A^2/ps
  ##D *= 100 #final unit: 10^-10 m^2/s or 10^-6 cm^2/s
  return diffusion_coeff, res.stderr, res.rvalue**2


def get_slope(time, msd):
  res = linregress(time,msd)
  return res.slope, res.intercept, res.stderr

def get_start_index(time, msd, threshold=False, min_steps=0):
    if threshold:
        print(f'Using slope error thresold of: {threshold}')
        for i, t in enumerate(time):
            if i != 0:
                slope, intercept, err = get_slope(np.log(time[i:]),np.log(msd[i:]))
                if (abs(slope-1) < threshold) and (i+1 > min_steps):
                    idx = i
                    break
                else:
                    idx = None
    else:
        slopes = []
        for i, t in enumerate(time):
            if i != 0:
                slope, intercept, err = get_slope(np.log(time[i:]),np.log(msd[i:]))
                slopes.append(abs(slope-1))
        slopes = np.array(slopes)
        slopes[slopes == np.nan] = 100
        #idx = np.argmin(slopes[~np.isnan(slopes)])
        idx = np.argmin(slopes)
    return idx
