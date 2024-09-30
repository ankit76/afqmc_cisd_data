# HEAT dataset

Script for extracting ground state energies:

```python
import numpy as np
mols = [
"H2",
"CH",
"CH2",
"NH",
"CH3",
"NH2",
"OH",
"HF",
"H2O",
"NH3",
"C2H",
"CN",
"C2H2",
"CO",
"HCN",
"N2",
"HCO",
"CF",
"NO",
"HNO",
"O2",
"HO2",
"OF",
"H2O2",
"F2",
"CO2",
]
data_dic = {}
with open("ene.dz.txt") as f:
for line in f:
parts = line.split()
species = parts[0]
data_dic[species] = {}
data_dic[species]["ccsdtqp"] = float(parts[5])
data_dic[species]["ccsdtq"] = float(parts[4])
ref = np.array([data_dic[mol]["ccsdtqp"] for mol in mols])
ccsdtq = np.array([data_dic[mol]["ccsdtq"] for mol in mols])

data = np.loadtxt("ene.txt", delimiter=",")
afqmc*hf = (data[:, 1] - ref) * 1000
afqmc*hf_rmsd = np.sqrt(np.mean((afqmc_hf) \*\* 2))
afqmc_hf_stoc_err = data[:, 2] * 1000
ccsdtq = (ccsdtq - ref) _ 1000
ccsdtq_rmsd = np.sqrt(np.mean((ccsdtq) \*\* 2))
ccsdpt = (data[:, 4] - ref) _ 1000
ccsdpt_rmsd = np.sqrt(np.mean((ccsdpt) ** 2))
afqmc_cisd = (data[:, 5] - ref) \* 1000
afqmc_cisd_rmsd = np.sqrt(np.mean((afqmc_cisd) ** 2))
afqmc_cisd_stoc_err = data[:, 6] \* 1000
```

Script for extracting atomization energies:

```python
afqmc_tz = {}
with open("afqmc.tz.txt") as f:
    for line in f:
        data = line.split()
        species = data[0]
        energy = float(data[1])
        energy_stoc_err = float(data[2])
        afqmc_tz[species] = (energy, energy_stoc_err)

afqmc_qz = {}
with open("afqmc.qz.txt") as f:
    for line in f:
        data = line.split()
        species = data[0]
        energy = float(data[1])
        energy_stoc_err = float(data[2])
        afqmc_qz[species] = (energy, energy_stoc_err)

spin_frozen = {
    "C": [2, 1],
    "N": [3, 1],
    "O": [2, 1],
    "F": [1, 1],
    "H2": [0, 0],
    "N2": [0, 2],
    "CN": [1, 2],
    "F2": [0, 2],
    "O2": [2, 2],
    "CO": [0, 2],
    "C2H2": [0, 2],
    "C2H": [1, 2],
    "CH2": [2, 1],
    "CH": [1, 1],
    "CH3": [1, 1],
    "CO2": [0, 3],
    "H2O2": [0, 2],
    "H2O": [0, 1],
    "HCO": [1, 2],
    "HF": [0, 1],
    "HO2": [1, 2],
    "NO": [1, 2],
    "OH": [1, 1],
    "HNO": [0, 2],
    "HCN": [0, 2],
    "CF": [1, 2],
    "NH2": [1, 1],
    "NH3": [0, 1],
    "NH": [2, 1],
    "OF": [1, 2],
}

ene_tz = {}
with open("ene.tz.txt") as f:
    for line in f:
        # ignore first line
        if "#" in line:
            continue
        parts = line.split()
        species = parts[0]
        spin = spin_frozen[species][0]
        ene_tz[species] = {}
        if spin == 0:
            ene_tz[species]["ccsdpt"] = float(parts[3])
            ene_tz[species]["hf"] = float(parts[5])
        else:
            ene_tz[species]["ccsdpt"] = float(parts[4])
            ene_tz[species]["hf"] = float(parts[6])
        ene_tz[species]["afqmc"] = afqmc_tz[species][0] - ene_tz[species]["hf"]
        ene_tz[species]["afqmc_stoc_err"] = afqmc_tz[species][1]
        ene_tz[species]["ccsdpt"] = ene_tz[species]["ccsdpt"] - ene_tz[species]["hf"]

ene_qz = {}
with open("ene.qz.txt") as f:
    for line in f:
        # ignore first line
        if "#" in line:
            continue
        parts = line.split()
        species = parts[0]
        spin = spin_frozen[species][0]
        ene_qz[species] = {}
        if spin == 0:
            ene_qz[species]["ccsdpt"] = float(parts[3])
            ene_qz[species]["hf"] = float(parts[5])
        else:
            ene_qz[species]["ccsdpt"] = float(parts[4])
            ene_qz[species]["hf"] = float(parts[6])
        ene_qz[species]["afqmc"] = afqmc_qz[species][0] - ene_qz[species]["hf"]
        ene_qz[species]["afqmc_stoc_err"] = afqmc_qz[species][1]
        ene_qz[species]["ccsdpt"] = ene_qz[species]["ccsdpt"] - ene_qz[species]["hf"]
```
