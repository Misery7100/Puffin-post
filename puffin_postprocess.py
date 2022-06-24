import h5py
import numpy as np
import os
import re
import argparse
import shutil
from tqdm import tqdm

from powPrepare import prepare

# ------------------------- #

def concatenate_fields_1D(folder: str):

    files = os.listdir(folder)
    E_im = dict()
    E_re = dict()

    for f in files:
        if re.search(r'_aperp_\d+\.h5', f):
            t = float(f.split('_')[-1].split('.h5')[0])
            data = h5py.File(os.path.join(folder, f), 'r')
            E = np.array(data['aperp'])
            E_re[t] = E[0, :]
            E_im[t] = E[1, :]
    
    ts = np.array(sorted(E_re.keys()))
    E_re = np.concatenate([E_re[t].reshape(-1, 1) for t in ts], axis=1)
    E_im = np.concatenate([E_im[t].reshape(-1, 1) for t in ts], axis=1)

    return dict(times=ts, E_real=E_re, E_imag=E_im)

# ------------------------- #

def concatenate_density_1D(folder: str):
    
    files = os.listdir(folder)
    nenp = dict()
    z2s = dict()

    for f in files:
        if re.search('electrons', f):
            t = float(f.split('_')[-1].split('.h5')[0])
            data = h5py.File(os.path.join(folder, f), 'r')
            tb = np.array(data['electrons'])

            z2 = tb[:, 2]
            ne = tb[:, 6]

            nenp[t] = ne
            z2s[t] = z2
    
    ts = np.array(sorted(z2s.keys()))
    uniq_z2 = get_unique_z_1D(z2s)

    concatenated = np.zeros((uniq_z2.size, ts.size))

    for i, t in enumerate(tqdm(ts, desc='Extend z grid (el density data)')):
        check = extend_by_z_1D(z2s[t], uniq_z2, nenp[t])
        vals = sorted(check.items(), key=lambda x: x[0])
        vals = np.array(list(map(lambda x: x[1], vals)))
        concatenated[:, i] = vals
    
    for i in range(concatenated.shape[1]):
        mask = np.isnan(concatenated[:, i])
        concatenated[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), concatenated[~mask, i])
    
    return dict(nenp_2D=concatenated, z_values=uniq_z2, times=ts)

# ------------------------- #

def get_unique_z_1D(z2s: dict):

    unique_z2 = set()

    for v in z2s.values():
        unique_z2 = unique_z2.union(set(v))

    unique_z2 = np.array(sorted(list(unique_z2)))

    return unique_z2

# ------------------------- #

def extend_by_z_1D(z2, unique_z2, nenp):

    pairs = sorted(zip(z2, nenp), key=lambda x: x[0])
    z2 = list(map(lambda x: x[0], pairs))
    nenp = list(map(lambda x: x[1], pairs))
    pairs = dict(pairs)
    output = dict((z, pairs.get(z, None)) for z in unique_z2)

    return output

# ------------------------- #

def extract_power_1D(folder: str):

    files = os.listdir(folder)
    files = list(filter(lambda x: re.search('_integrated_all.vsh5', x), files))

    data = h5py.File(os.path.join(folder, files[0]), 'r')
    
    power_SI = np.array(data['power_SI'])
    power_SI_Norm = np.array(data['power_SI_Norm'])

    return dict(power_SI_Norm=power_SI_Norm, power_SI=power_SI)

# ------------------------- #

def decorated_save(folder, file, obj, decimals: int = 5):
    print(f'Saving {file.split(".dat")[0]} to {os.path.join(folder, file)}')
    np.savetxt(os.path.join(folder, file), obj, fmt=f'%1.{decimals}f')

# ------------------------- #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory',  required=True)
    parser.add_argument('-bn', '--basename', required=False, default='main')
    args = parser.parse_args()

    folder = os.path.abspath(args.directory)
    bname = args.basename

    fld = concatenate_fields_1D(folder)
    dens = concatenate_density_1D(folder)

    prepare(bname, folder)
    pwr = extract_power_1D(folder)

    output_directory = os.path.join(folder, 'post_output')

    if not os.path.exists(output_directory):

        print('Saving dir does not exist - create')
        os.makedirs(output_directory)
    
    else:

        print('Saving dir exists - clear')
        shutil.rmtree(output_directory)
        os.makedirs(output_directory)

    print('Saving result files')

    # 1: times, E_real, E_imag
    decorated_save(output_directory, 't_values.dat', fld['times'])
    decorated_save(output_directory, 'E_real_2D(t, z).dat', fld['E_real'])
    decorated_save(output_directory, 'E_imag_2D(t, z).dat', fld['E_imag'])

    # 2: nenp_2D=, z_values
    decorated_save(output_directory, 'nenp_2D(t, z).dat', dens['nenp_2D'])
    decorated_save(output_directory, 'nenp_z_values.dat', dens['z_values'])

    # 3: power_SI_Norm, power_SI
    decorated_save(output_directory, 'power_SI.dat', pwr['power_SI'])
    decorated_save(output_directory, 'power_SI_Norm.dat', pwr['power_SI_Norm'])

    print(f'Results saved to {output_directory}')