# Transition metal oxide molecules

Script for extracting energies from the `ene.txt` file:

```python
import numpy as np
ene_tz = np.loadtxt("ene.tz.txt")
ene_qz = np.loadtxt("ene.qz.txt")
hf_5z = np.loadtxt("ene.5z.txt")

hf_tz = ene_tz[:, -1]
hf_qz = ene_qz[:, -1]
shci_tz_corr = ene_tz[:, 0] - hf_tz
shci_qz_corr = ene_qz[:, 0] - hf_qz
afqmc_msd_tz_corr = ene_tz[:, 1] - hf_tz
afqmc_msd_tz_stoc_err = ene_tz[:, 2]
afqmc_msd_qz_corr = ene_qz[:, 1] - hf_qz
afqmc_msd_qz_stoc_err = ene_qz[:, 2]
ccsd_tz_corr = ene_tz[:, 3] - hf_tz
ccsd_qz_corr = ene_qz[:, 3] - hf_qz
ccsdpt_tz_corr = ene_tz[:, 4] - hf_tz
ccsdpt_qz_corr = ene_qz[:, 4] - hf_qz
afqmc_uhf_tz = ene_tz[:, 5]
afqmc_uhf_qz = ene_qz[:, 5]
mask_afqmc_uhf = np.array(afqmc_uhf_qz) != 0
afqmc_uhf_tz_corr = afqmc_uhf_tz - hf_tz
afqmc_uhf_qz_corr = afqmc_uhf_qz - hf_qz
afqmc_uhf_tz_stoc_err = ene_tz[:, 6]
afqmc_uhf_qz_stoc_err = ene_qz[:, 6]
afqmc_cisd_tz = ene_tz[:, 7]
afqmc_cisd_qz = ene_qz[:, 7]
mask_afqmc_cisd = np.array(afqmc_cisd_qz) != 0
afqmc_cisd_tz_corr = afqmc_cisd_tz - hf_tz
afqmc_cisd_qz_corr = afqmc_cisd_qz - hf_qz
afqmc_cisd_tz_stoc_err = ene_tz[:, 8]
afqmc_cisd_qz_stoc_err = ene_qz[:, 8]

dmc_cbs_diss = -np.array(
    [
        0.254342134,
        0.256823202,
        0.231558857,
        0.1444383,
        0.128771035,
        0.141604044,
        0.098327667,
    ]
)
dmc_cbs_diss_stoc_err = np.array(
    [
        0.000285561,
        0.000423543,
        0.000961927,
        0.001037217,
        0.001296762,
        0.001339664,
        0.00157939,
    ]
)
```
