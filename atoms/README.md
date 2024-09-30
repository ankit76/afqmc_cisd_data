# Atomic energies

Script for extracting energies from the `ene.txt` file:

```python
import numpy as np
data = np.loadtxt("ene.txt")
ene_fci = data[:, 0]
ene_afqmc_cisd = (data[:, 1] - ene_fci) * 1000
ene_afqmc_cisd_rmsd = np.sqrt(np.mean(ene_afqmc_cisd**2))
ene_afqmc_cisd_err = data[:, 2]
ene_afqmc_hf = (data[:, 3] - ene_fci) * 1000
ene_afqmc_hf_rmsd = np.sqrt(np.mean(ene_afqmc_hf**2))
ene_afqmc_hf_err = data[:, 4]
ene_afqmc_ccsdpt = (data[:, 5] - ene_fci) * 1000
ene_afqmc_ccsdpt_rmsd = np.sqrt(np.mean(ene_afqmc_ccsdpt**2))
labels = ["Be", "B", "C", "N", "O", "F", "Ne"]
```
