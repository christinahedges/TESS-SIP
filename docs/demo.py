import sys
sys.path.append('/Users/ch/K2/requests/ruth/tess-long-period-tools/TESS-SIP/src/')
from tess_sip import SIP
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

print(lk.__file__)

# Download target pixel files
# We're using TOI-700
tpfs = lk.search_targetpixelfile('TIC 150428135', mission='tess', sector=list(np.arange(14))).download_all()

# Run SIP
r = SIP(tpfs, min_period=10, max_period=80)

# Example plotting
fig, axs = plt.subplots(2, figsize=(4, 6))
axs[0].plot(r['periods'], r['power'], c='k')
axs[0].set(xlabel='Period [d]', ylabel='Power', title='Periodogram')
axs[0].axvline(r['period_at_max_power'], color='r', ls='--', label=f"{r['period_at_max_power']:0.2f} days")
axs[0].legend()
r['raw_lc'].plot(ax=axs[1], c='r', label='Raw', alpha=0.5)
r['corr_lc'].plot(ax=axs[1], c='k', label='Corrected', lw=0.1)
axs[1].set(ylim=(0.9, 1.1))
plt.savefig('demo.png', dpi=200, bbox_inches='tight')
