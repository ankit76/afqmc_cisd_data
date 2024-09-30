import numpy as np
from pyscf import gto, scf, cc
from ad_afqmc import pyscf_interface

geom = open("geom.xyz").read().split("\n")[7:]
geom = "\n".join(geom)
mol = gto.M(atom=geom, verbose=4, basis='ccpvtz', spin=0, max_memory=10000)
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

norb_frozen = 3
mycc = cc.CCSD(mf)
mycc.frozen = norb_frozen
mycc.run()
et = mycc.ccsd_t()
print(f"CCSD(T) energy: {mycc.e_tot + et}")

umf = scf.UHF(mol)
mf.level_shift = 0.4
umf.kernel()
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
mo1 = umf.stability(external=True)[0]
umf = umf.newton().run(mo1, umf.mo_occ)
umf.stability()

mycc = cc.CCSD(umf)
mycc.frozen = norb_frozen
mycc.run()
et = mycc.ccsd_t()
print(f"UCCSD(T) energy: {mycc.e_tot + et}")

pyscf_interface.prep_afqmc(mycc)

