# W4-MR dataset

Script for extracting ground state energies:

```python
import numpy as np
data_dic = {}
with open("ene.txt") as f:
    for line in f:
        parts = line.split()
        species = parts[0]
        data_dic[species] = {}
        data_dic[species]["ref"] = float(parts[5])
        data_dic[species]["ccsdpt"] = (
            float(parts[1]) - data_dic[species]["ref"]
        ) * 1000
        data_dic[species]["afqmc_cisd"] = (
            float(parts[2]) - data_dic[species]["ref"]
        ) * 1000
        data_dic[species]["afqmc_cisd_stoc_err"] = float(parts[3]) * 1000
        data_dic[species]["ccsdt"] = (float(parts[4]) - data_dic[species]["ref"]) * 1000
        data_dic[species]["afqmc_hf"] = (
            float(parts[6]) - data_dic[species]["ref"]
        ) * 1000
        data_dic[species]["afqmc_hf_stoc_err"] = float(parts[7]) * 1000
        # make a latex label with subscripts for a plot
        data_dic[species]["label"] = r""
        for i in range(len(species)):
            if species[i].isdigit():
                data_dic[species]["label"] += f"$_{species[i]}$"
            else:
                data_dic[species]["label"] += species[i]

# get keys for data_dic in alphabetical order
mols = list(data_dic.keys())
mols.sort()
# move ClO3 to the end
mols.remove("ClO3")
mols.append("ClO3")
# get labels for the plot
labels = [data_dic[mol]["label"] for mol in mols]
# get data for the plot
afqmc_cisd = np.array([data_dic[mol]["afqmc_cisd"] for mol in mols])
afqmc_cisd_stoc_err = np.array([data_dic[mol]["afqmc_cisd_stoc_err"] for mol in mols])
# bootstrap to get stoc_err in rmsd
n_samples = 1000
afqmc_cisd_rmsd_samples = []
for i in range(n_samples):
    sample = np.random.normal(afqmc_cisd, afqmc_cisd_stoc_err, len(afqmc_cisd))
    afqmc_cisd_rmsd_samples.append(np.sqrt(np.mean(sample**2)))
afqmc_cisd_rmsd = np.mean(afqmc_cisd_rmsd_samples)
afqmc_cisd_rmsd_stoc_err = np.std(afqmc_cisd_rmsd_samples)
ccsdpt = np.array([data_dic[mol]["ccsdpt"] for mol in mols])
ccsdpt_rmsd = np.sqrt(np.mean(ccsdpt**2))
ccsdt = np.array([data_dic[mol]["ccsdt"] for mol in mols])
ccsdt_rmsd = np.sqrt(np.mean(ccsdt**2))
afqmc_hf = np.array([data_dic[mol]["afqmc_hf"] for mol in mols])
afqmc_hf_stoc_err = np.array([data_dic[mol]["afqmc_hf_stoc_err"] for mol in mols])
# bootstrap to get stoc_err in rmsd
n_samples = 1000
afqmc_hf_rmsd_samples = []
for i in range(n_samples):
    sample = np.random.normal(afqmc_hf, afqmc_hf_stoc_err, len(afqmc_hf))
    afqmc_hf_rmsd_samples.append(np.sqrt(np.mean(sample**2)))
afqmc_hf_rmsd = np.mean(afqmc_hf_rmsd_samples)
afqmc_hf_rmsd_stoc_err = np.std(afqmc_hf_rmsd_samples)
```

Script for extracting atomization energies:

```python
ene_dz = {}
with open("ene.dz.txt") as f:
    for line in f:
        parts = line.split()
        species = parts[0]
        ene_dz[species] = {}
        ene_dz[species]["hf"] = float(parts[4])
        ene_dz[species]["afqmc"] = float(parts[1]) - ene_dz[species]["hf"]
        ene_dz[species]["afqmc_stoc_err"] = float(parts[2])
        ene_dz[species]["ccsdpt"] = float(parts[3]) - ene_dz[species]["hf"]

ene_tz = {}
with open("ene.tz.txt") as f:
    for line in f:
        parts = line.split()
        species = parts[0]
        ene_tz[species] = {}
        ene_tz[species]["hf"] = float(parts[4])
        ene_tz[species]["afqmc"] = float(parts[1]) - ene_tz[species]["hf"]
        ene_tz[species]["afqmc_stoc_err"] = float(parts[2])
        ene_tz[species]["ccsdpt"] = float(parts[3]) - ene_tz[species]["hf"]

# these are in kcal/mol
atomization_corrections = {}
with open("ene.cc.txt") as f:
    for line in f:
        parts = line.split()
        species = parts[0]
        atomization_corrections[species] = {}
        atomization_corrections[species]["ccsdt"] = float(parts[1])
        atomization_corrections[species]["ccsdtq"] = float(parts[2])
        atomization_corrections[species]["ccsdtqp"] = float(parts[3])
        atomization_corrections[species]["ccsdtqph"] = float(parts[4])
```
