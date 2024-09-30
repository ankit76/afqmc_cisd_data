# N<sub>2</sub> dissociation

Script for extracting energies from the `ene.txt` file:

```python
import numpy as np
data = np.loadtxt("ene.txt")
d = data[:, 0]
ref = data[:, 4]
afqmc_cisd = (data[:, 1] - ref) * 1000
afqmc_cisd_err = data[:, 2] * 1000
afqmc_cisd_npe = np.max(afqmc_cisd) - np.min(afqmc_cisd)
ccsdpt = (data[:, 5] - ref) * 1000
ccsdpt_npe = np.max(ccsdpt) - np.min(ccsdpt)
ccsdt = (data[:, 6] - ref) * 1000
ccsdt_npe = np.max(ccsdt) - np.min(ccsdt)
ccsdtq = (data[:, 7] - ref) * 1000
ccsdtq_npe = np.max(ccsdtq) - np.min(ccsdtq)
rccsdpt = (data[:, 10] - ref) * 1000
afqmc_hf = (data[:, -2] - ref) * 1000
afqmc_hf_err = data[:, -1] * 1000
afqmc_hf_npe = np.max(afqmc_hf) - np.min(afqmc_hf)
```
