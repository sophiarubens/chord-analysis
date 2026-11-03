import sys, chord_pipeline
from chord_util import andata
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import color_sequences as cseq
from matplotlib.colors import BoundaryNorm
import os
import time
import cmasher as cmr

acq_dir="/home/sophiaru/projects/rrg-kmsmith/chord-data/kotekan_vis_files/subset/"
acq="acq_20261025_233145_198439631/"
# acq="acq_20261101_014834_710489386" # looks promising in the choco overview, but the data is not currently on Fir yet

reader = andata.CorrReader(acq_dir+acq+"vis_*.h5")
reader.select_input_type("dish")

chunk_width=100
band_chunk_floors=np.arange(300,1500,chunk_width)
N_BLOCKS = 8                                        # chunks across the band
N_chunks=len(band_chunk_floors)
crosspol_Calgary_vs_differential_MEAN=np.zeros((N_chunks,2))
crosspol_Calgary_vs_differential_MED=np.zeros((N_chunks,2))

fig,axs=plt.subplots(3,4, figsize=(12,8), layout="constrained", sharex=True, sharey=True)

for k,band_chunk_floor in enumerate(band_chunk_floors):
    t0=time.time()
    i,j=divmod(k,4)
    reader.select_freq_range(band_chunk_floor,band_chunk_floor+chunk_width)  # MHz
    data = reader.read() # as frustratingly loopy/repetitive as it seems to read multiple times, I will never reasonably get the ~3.5 TB RAM required to read everything in one go

    vis=data.vis
    freq_centre=data.freq
    n_freq=len(freq_centre)
    prods=data.prod
    n_prod=len(prods)
    labels=data.input["correlator_input"]
    n_el=len(labels)
    times=data.time
    frac_lost=data.flags["frac_lost"][:]
    frames_added=data.flags["frames_added"][:]
    t_unix=data.time

    C = n_freq//N_BLOCKS                                # channels per chunk
    
    valid = (frac_lost < 1) & (frames_added > 0)
    t_idx = int(np.argmax(valid.sum(axis=0)))           # best-covered time bin
    t_str = datetime.fromtimestamp(t_unix[0], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    chmid = int(np.argmin(np.abs(freq_centre - band_chunk_floor+chunk_width/2)))
    starts = sorted({int(c) // C * C for c in np.linspace(0, n_freq - 1, N_BLOCKS)}
                    | {chmid // C * C})
    chans, rows = [], []
    for s in starts:                                    # one whole chunk per read
        block = vis[s:s + C][:, :, t_idx]            # (chan, prod)
        ok = valid[s:s + C, t_idx]
        chans.append(np.arange(s, s + block.shape[0])[ok])
        rows.append(block[ok])
    chans, vis_t = np.concatenate(chans), np.concatenate(rows)
    print(f"read {len(starts)} chunks; {chans.size} valid channels")
    
    auto = np.empty(n_el, int)
    is_auto = prods["input_a"] == prods["input_b"]
    auto[prods["input_a"][is_auto]] = np.flatnonzero(is_auto)
    power = vis_t[:, auto].real                         # (chan, element)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.abs(vis_t) / np.sqrt(power[:, prods["input_a"]]
                                      * power[:, prods["input_b"]])
        coh = np.nanmean(np.where(np.isfinite(coh), coh, np.nan), axis=0)
    
    A = np.full((n_el, n_el), np.nan)
    A[prods["input_a"], prods["input_b"]] = coh
    A[np.diag_indices(n_el)] = 1                   # autos are 1 by definition
    np.save("vis_snapshot_{}_{}_{}_MHz.npy".format(acq[:-1],band_chunk_floor,band_chunk_floor+chunk_width), A)

    im = axs[i,j].imshow(A, norm=mcolors.LogNorm(vmin=1e-3, vmax=1), cmap=cmr.eclipse,
                        interpolation="nearest")
    plt.colorbar(im,ax=axs[i,j])
    axs[i,j].set_title("{:d}-{:d} MHz".format(band_chunk_floor,band_chunk_floor+chunk_width))

    half_n_el=n_el//2
    crosspol_vals_this_freq_chunk=np.zeros(half_n_el)
    for m in range(half_n_el):
        crosspol_vals_this_freq_chunk[m]=A[m,m+half_n_el]
    
    crosspol_Calgary_vs_differential_MEAN[k,:]=[np.mean(crosspol_vals_this_freq_chunk[:-2]),
                                                np.mean(crosspol_vals_this_freq_chunk[-2:])]
    crosspol_Calgary_vs_differential_MED[k,:]=[np.median(crosspol_vals_this_freq_chunk[:-2]),
                                               np.median(crosspol_vals_this_freq_chunk[-2:])]

    if k==N_chunks-1:
        ticks = np.arange(0, n_el, 8)
        axs[-1,0].set_xticks(ticks, [labels[i] for i in ticks], rotation=90)
        axs[-1,0].set_yticks(ticks, [labels[i] for i in ticks])

    del data, vis, prods, labels, times, frac_lost, frames_added, t_unix, valid, block, ok, \
        chans, rows, vis_t, auto, is_auto, power, coh, A, crosspol_vals_this_freq_chunk         # free up some RAM
    t1=time.time()
    print("processed chunk k={} in {:.2f} s".format(k,t1-t0))

axs[-1,0].set_xlabel("input b")
axs[-1,0].set_ylabel("input a")

fig.suptitle(f"CHORD-64 Tau A acquisition time snapshot at {t_idx} ({t_str})")
fig.tight_layout()
plt.savefig("vis_snapshots_freq_sweep.png")

colour_sequence=cseq["Accent"]

np.save("crosspol_{}_Calgary_vs_differential_MEAN.npy".format(acq[:-1]),crosspol_Calgary_vs_differential_MEAN)
np.save("crosspol_{}_Calgary_vs_differential_MEDIAN.npy".format(acq[:-1]),crosspol_Calgary_vs_differential_MED)
band_chunk_mids=band_chunk_floors+0.5*chunk_width
plt.figure()
plt.scatter(band_chunk_mids,20*np.log10(crosspol_Calgary_vs_differential_MEAN[:,0]),
            label="Calgary LNA",
            facecolor=colour_sequence[0])
plt.scatter(band_chunk_mids,20*np.log10(crosspol_Calgary_vs_differential_MEAN[:,1]),
            label="differential LNA",
            facecolor=colour_sequence[1])
plt.xlabel("freq (MHz)")
plt.ylabel("excitation amplitude (dB down from autocorr)")
plt.legend()
plt.title("Crosspol excitation for both CHORD LNAs (mean across antennas)")
plt.savefig("crosspol_chromaticity_mean.png")

plt.figure()
plt.scatter(band_chunk_mids,20*np.log10(crosspol_Calgary_vs_differential_MED[:,0]),
            label="Calgary LNA",
            facecolor=colour_sequence[0])
plt.scatter(band_chunk_mids,20*np.log10(crosspol_Calgary_vs_differential_MED[:,1]),
            label="differential LNA",
            facecolor=colour_sequence[1])
plt.xlabel("freq (MHz)")
plt.ylabel("excitation amplitude (dB down from autocorr)")
plt.legend()
plt.title("Crosspol excitation for both CHORD LNAs (median across antennas)")
plt.savefig("crosspol_chromaticity_median.png")