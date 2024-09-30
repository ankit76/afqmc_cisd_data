import sys, os
import numpy as np
from pyscf import gto, scf, cc, lib
from ad_afqmc import pyscf_interface
import time

norb_frozen = 2
r = 2.7
mol = gto.M(atom=f"N 0 0 {-r/2}; N 0 0 {r/2}", verbose=4, basis='ccpvdz', unit="bohr")

umf = scf.UHF(mol)
umf.chkfile = 'umf.chk'
umf.level_shift = 0.5
umf.kernel()
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
umf.stability()
umf.analyze()

start_time = time.time()
mycc = cc.CCSD(umf)
mycc.frozen = norb_frozen
mycc.run()
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken for CCSD calculation: {time_taken:.2f} seconds")

pyscf_interface.prep_afqmc(mycc)

start_time = time.time()
et = mycc.ccsd_t()
end_time = time.time()
print(f"CCSD(T) energy: {mycc.e_tot + et}")
time_taken = end_time - start_time
print(f"Time taken for CCSD(T) calculation: {time_taken:.2f} seconds")

from ad_afqmc import config
config.afqmc_config["use_gpu"] = True
from ad_afqmc import run_afqmc
options = {
    "dt": 0.005,
    "n_eql": 3,
    "n_ene_blocks": 1,
    "n_sr_blocks": 5,
    "n_blocks": 200,
    "n_walkers": 200,
    "n_batch": 1,
    "walker_type": "rhf",
    "trial": "ucisd"
}
e_afqmc, err_afqmc = run_afqmc.run_afqmc(options=options)

