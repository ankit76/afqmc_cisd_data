import numpy as np
from pyscf import gto, scf, cc
from ad_afqmc import pyscf_interface
import h5py

norb_frozen = 2
atomstring = "C 0 -1.24942055 0; O 0 0.89266692 0"

mol = gto.M(atom = atomstring, verbose=4, basis="augccpvqz", unit="bohr")
mf = scf.RHF(mol)
mf.chkfile = 'mf.chk'
mf.max_cycle = 500
mf.level_shift = 0.4
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mf.stability()

norb_frozen = 2
overlap = mf.get_ovlp()
h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)

# dipole moment
nuc_dipmom = [0.0, 0.0, 0.0]
for i in range(mol.natm):
  for j in range(3):
    nuc_dipmom[j] += mol.atom_charge(i) * mol.atom_coord(i)[j]

# spatial orbitals
dip_ints_ao = -mol.intor_symmetric('int1e_r', comp=3)
dip_ints_mo = np.empty_like(dip_ints_ao)
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo[i] = mf.mo_coeff.T.dot(dip_ints_ao[i]).dot(mf.mo_coeff)

mycc = cc.CCSD(mf)
mycc.frozen = norb_frozen
mycc.run()
et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)

pyscf_interface.prep_afqmc(mycc)
pyscf_interface.finite_difference_properties(mol, dip_ints_ao[1], observable_constant=nuc_dipmom[1], norb_frozen=norb_frozen)
pyscf_interface.finite_difference_properties(mol, dip_ints_ao[1], observable_constant=nuc_dipmom[1], norb_frozen=norb_frozen, relaxed=False)

# frozen orbitals
dip_ints_mo_act = np.zeros((dip_ints_ao.shape[0], mol.nao - norb_frozen, mol.nao - norb_frozen))
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo_act[i] = dip_ints_mo[i][norb_frozen:, norb_frozen:]
  nuc_dipmom[i] += 2. * np.trace(dip_ints_mo[i][:norb_frozen, :norb_frozen])
dip_ints_mo = dip_ints_mo_act
with h5py.File('observable.h5', 'w') as fh5:
    fh5['constant'] = np.array([ nuc_dipmom[1] ])
    fh5['op'] = dip_ints_mo[1].flatten()

from ad_afqmc import config
config.afqmc_config["use_gpu"] = True
from ad_afqmc import run_afqmc

options = {
    "dt": 0.005,
    "n_eql": 3,
    "n_ene_blocks": 2,
    "n_sr_blocks": 50,
    "n_blocks": 20,
    "n_walkers": 200,
    "orbital_rotation": False,
    "ad_mode": "forward",
    "walker_type": "rhf",
    "trial": "cisd",
}
e_afqmc, err_afqmc = run_afqmc.run_afqmc(options=options)

