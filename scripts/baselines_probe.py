import time

import numpy as np

from qrl.chem import build
from qrl.sim import Circuit
from qrl.vqe import adapt, beam_search, random_order, uccsd

for name, r in [("H2",0.735),("LiH",1.6),("LiH",3.0),("H4",1.0),("H4",2.0)]:
    p = build(name, r); c = Circuit(p.n_qubits, p.n_electrons, p.H, p.pool)
    t=time.time(); u=uccsd(p,c); a=adapt(p,c); rr=random_order(p,c,np.random.default_rng(0)); t1=time.time()
    b=beam_search(p,c,upper=a); t2=time.time()
    print(p.tag, "pool",len(p.pool), "UCCSD",u["cnots"],f"{u['err']:.1e}", "ADAPT",a["cnots"],a["n_gates"],a["reached"],
          "RAND",rr["cnots"],rr["reached"], "BEAM",b.get("cnots"),b.get("n_gates"), f"t={t1-t:.1f}s beam={t2-t1:.1f}s")
