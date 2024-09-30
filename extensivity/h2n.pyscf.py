import sys, os
import numpy as np
from pyscf import gto, scf, cc, lib, ci
from ad_afqmc import pyscf_interface, wavefunctions
import time
import pickle

args = sys.argv
n = int(args[1])

d_1 = 2.0
d_2 = 10.0
geom = ""
for i in range(n):
    geom += f"H {i * d_2} 0 0; H {i * d_2} 0 {d_1}; "

mol = gto.M(
    atom=geom,
    basis="sto-6g",
    unit="bohr",
    verbose=4,
)
norb_frozen = 0

mf = scf.RHF(mol)
mf.chkfile = 'mf.chk'
mf.level_shift = 0.5
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mf.stability()
mf.analyze()

pyscf_interface.prep_afqmc(mf)

mycisd = ci.CISD(mf)
mycisd.frozen = norb_frozen
mycisd.run()

# pyscf_interface.prep_afqmc(mf, norb_frozen=norb_frozen)
ci0, ci1, ci2 = mycisd.cisdvec_to_amplitudes(mycisd.ci, mycisd.nmo, mycisd.nocc)
ci2 = ci2.transpose(0, 2, 1, 3) / ci0
ci1 = ci1 / ci0

trial = wavefunctions.cisd(sum(ci1.shape), (ci1.shape[0], ci1.shape[0]))
wave_data = {}
wave_data["ci1"] = ci1
wave_data["ci2"] = ci2
# write wavefunction to disk
with open("trial.pkl", "wb") as f:
    pickle.dump([trial, wave_data], f)


