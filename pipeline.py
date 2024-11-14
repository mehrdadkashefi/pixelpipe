import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
from distutils.util import strtobool


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import spikeinterface.full as si
from matplotlib import gridspec, rcParams
import pandas as pd


# Define a custom argument type for a list of strings
def list_of_strs(arg):
    return list(map(str, arg.split(',')))

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-time", type=str, default=datetime.now().strftime("%Y_%m_%d_%H%M%S"),
                            help="time of processing")
    # Get the path of the recording
    parser.add_argument("--f", type=str,
                            help="path to the recording folder")
    parser.add_argument("--preprocess", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                            help="If True, preprocess the data") 
    parser.add_argument("--motion", type=list_of_strs, default=[],
                            help="If true, estimate motion")
    parser.add_argument("--sort", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                            help="If true, sort the data")
    
    args = parser.parse_args()
    return args

def plot_sorting_summary(sorter_path):
    results_dir = Path(sorter_path / 'sorter_output')
    ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
    camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
    contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
    chan_map =  np.load(results_dir / 'channel_map.npy')
    templates =  np.load(results_dir / 'templates.npy')
    chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
    chan_best = chan_map[chan_best]
    amplitudes = np.load(results_dir / 'amplitudes.npy')
    st = np.load(results_dir / 'spike_times.npy')
    clu = np.load(results_dir / 'spike_clusters.npy')
    firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
    dshift = ops['dshift']
    cluster_label = pd.read_csv(results_dir / 'cluster_KSLabel.tsv', sep='\t')['KSLabel'].values


    fig = plt.figure(figsize=(10,10), dpi=100)
    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)
    if dshift is not None:
        ax = fig.add_subplot(grid[0,0])
        ax.plot(np.arange(0, ops['Nbatches'])*2, dshift);
        ax.set_xlabel('time (sec.)')
        ax.set_ylabel('drift (um)')

    ax = fig.add_subplot(grid[0,1:])
    t0 = 0 
    t1 = np.nonzero(st > ops['fs']*5)[0][0]
    ax.scatter(st[t0:t1]/30000., chan_best[clu[t0:t1]], s=0.5, color='k', alpha=0.25)
    ax.set_xlim([0, 5])
    ax.set_ylim([chan_map.max(), 0])
    ax.set_xlabel('time (sec.)')
    ax.set_ylabel('channel')
    ax.set_title('spikes from units')

    ax = fig.add_subplot(grid[1,0])
    nb=ax.hist(firing_rates, 20, color='gray')
    ax.set_xlabel('firing rate (Hz)')
    ax.set_ylabel('# of units')

    ax = fig.add_subplot(grid[1,1])
    nb=ax.hist(camps, 20, color='gray')
    ax.set_xlabel('amplitude')
    ax.set_ylabel('# of units')

    ax = fig.add_subplot(grid[1,2])
    nb=ax.hist(np.minimum(100, contam_pct), np.arange(0,105,5), color='grey')
    ax.plot([10, 10], [0, nb[0].max()], 'k--')
    ax.set_xlabel('% contamination')
    ax.set_ylabel('# of units')
    ax.set_title('< 10% = good units'+ '\n' + str(sum(cluster_label == 'good'))+'/'+str(len(cluster_label)))

    for k in range(2):
        ax = fig.add_subplot(grid[2,k])
        is_ref = contam_pct<10.
        ax.scatter(firing_rates[~is_ref], camps[~is_ref], s=3, color='r', label='mua', alpha=0.25)
        ax.scatter(firing_rates[is_ref], camps[is_ref], s=3, color='b', label='good', alpha=0.25)
        ax.set_ylabel('amplitude (a.u.)')
        ax.set_xlabel('firing rate (Hz)')
        ax.legend()
        if k==1:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('loglog')
    fig.savefig(Path.joinpath(results_dir, 'summary.pdf'), format='pdf')
    plt.close()

    # Plot summary of good and multiunit waveforms
    probe = ops['probe']
    # x and y position of probe sites
    xc, yc = probe['xc'], probe['yc']
    nc = 16 # number of channels to show
    good_units = np.nonzero(contam_pct <= 0.1)[0]
    mua_units = np.nonzero(contam_pct > 0.1)[0]


    gstr = ['good', 'mua']
    for j in range(2):
        units = good_units if j==0 else mua_units 
        fig = plt.figure(figsize=(12,3), dpi=150)
        grid = gridspec.GridSpec(2,20, figure=fig, hspace=0.25, wspace=0.5)

        for k in range(40):
            wi = units[np.random.randint(len(units))]
            wv = templates[wi].copy()  
            cb = chan_best[wi]
            nsp = (clu==wi).sum()
            
            ax = fig.add_subplot(grid[k//20, k%20])
            n_chan = wv.shape[-1]
            ic0 = max(0, cb-nc//2)
            ic1 = min(n_chan, cb+nc//2)
            wv = wv[:, ic0:ic1]
            x0, y0 = xc[ic0:ic1], yc[ic0:ic1]
            amp = 4
            for ii, (xi,yi) in enumerate(zip(x0,y0)):
                t = np.arange(-wv.shape[0]//2,wv.shape[0]//2,1,'float32')
                t /= wv.shape[0] / 20
                ax.plot(xi + t, yi + wv[:,ii]*amp, lw=0.5, color='k')

            ax.set_title(f'{nsp}', fontsize='small')
            ax.axis('off')
        fig.savefig(Path.joinpath(results_dir, 'summary_wave_'+gstr[j]+'.pdf'), format='pdf')
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    n_jobs = 10
    process_name = f"{args.process_time}"
    print(process_name)
    print(f"Looking for recording in: {args.f}")
    base_folder = Path(args.f)
    folder_name = args.f.split('/')[-1]
    print(f"Assuming binaries are in: {folder_name}")
    # Get folder name and binary streams
    spikeglx_folder = base_folder / (folder_name +'_g0/')
    # Find folders in the recording folder
    subfolders = [f for f in sorted(spikeglx_folder.iterdir()) if f.is_dir()]
    print(subfolders)
    print(f"Found {len(subfolders)} probes in the recording folder.")
    # loop on probes
    for probe_i in range(len(subfolders)):
        print(f"Processing probe {probe_i + 1}/{len(subfolders)}")
        probe_folder = subfolders[probe_i]
        # Get the streams in the folder
        stream_names, stream_ids = si.get_neo_streams('spikeglx', probe_folder)
        print(F"Found the following streams:", stream_names)

        # make a plot directory if it does not exist
        plot_folder = probe_folder / 'plots'
        if not plot_folder.exists():
            plot_folder.mkdir()

        # Detect probe type and channel map
        raw_rec = si.read_spikeglx(probe_folder, stream_name='imec'+ str(probe_i) +'.ap', load_sync_channel=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        si.plot_probe_map(raw_rec, ax=ax, with_channel_ids=True)
        ax.set_ylim(-100, 200)
        sns.despine(trim=True)
        plt.savefig(probe_folder / plot_folder/ 'probe_map.png')
        plt.close()

        # Preprocess the data
        if args.preprocess:
            print('Preprocessing the data')
            if not (probe_folder / 'preprocess').exists():
                #(probe_folder / 'preprocess').mkdir()
                rec1 = si.highpass_filter(raw_rec, freq_min=400.)
                bad_channel_ids, channel_labels = si.detect_bad_channels(rec1)
                rec2 = rec1.remove_channels(bad_channel_ids)
                print('bad_channel_ids', bad_channel_ids)
                print('Num bad channels:' ,len(bad_channel_ids))
                rec3 = si.phase_shift(rec2)
                rec = si.common_reference(rec3, operator="median", reference="global")

                # Plot raw and preprocessed traces
                fig, axs = plt.subplots(ncols=3, figsize=(10, 5))
                si.plot_traces(raw_rec, backend='matplotlib',  clim=(-50, 50), ax=axs[0])
                si.plot_traces(rec1, backend='matplotlib',  clim=(-50, 50), ax=axs[1])
                si.plot_traces(rec, backend='matplotlib',  clim=(-50, 50), ax=axs[2])
                for i, label in enumerate(('Raw', 'High pass', 'CAR')):
                    axs[i].set_title(label)
                plt.savefig(probe_folder / plot_folder/ 'preprocess.png')
                plt.close()

                # Estimate noise levels
                noise_levels_microV = si.get_noise_levels(rec, return_scaled=True)
                fig, ax = plt.subplots()
                _ = ax.hist(noise_levels_microV, bins=np.arange(5, 30, 2.5))
                ax.set_xlabel('noise  [microV]')
                plt.savefig(probe_folder / plot_folder/ 'noise_level.png')
                plt.close()


                # plot some channels
                fig, ax = plt.subplots(figsize=(20, 10))
                some_chans = rec.channel_ids[[20, 150, 250, ]]
                si.plot_traces({'Raw':raw_rec, 'Final': rec}, backend='matplotlib', mode='line', ax=ax, channel_ids=some_chans)
                plt.savefig(probe_folder / plot_folder/ 'sample_channels.png')
                plt.close()

                job_kwargs = dict(n_jobs=n_jobs, chunk_duration='1s', progress_bar=True)
                rec = rec.save(folder= probe_folder / 'preprocess', format='binary', **job_kwargs)

            else: 
                print('Preprocess folder already exists')
                rec = si.read_binary_folder(probe_folder / 'preprocess')
                print(rec)

        if len(args.motion) != 0:
            print('Following motion estimation methods will be used:', args.motion)
            # Estimate motion
            from spikeinterface.sortingcomponents.peak_detection import detect_peaks
            from spikeinterface.sortingcomponents.peak_localization import localize_peaks

            if not (probe_folder / 'preprocess').exists():
                print('No preprocess folder found. Run pre-processing on data!')
                assert False
            else:
                print('Found preprocess folder. Estimating motion from preprocessed data!')
                rec = si.read_binary_folder(probe_folder / 'preprocess')
                print(rec)

            # Estimate motion
            motion_folder = probe_folder / 'motion'
            if not motion_folder.exists():
                motion_folder.mkdir()
                print('Found no motion folder. Estimating peaks and their locations. This may take a while! Hang tight!')
                # Detect, extract, and localize peaks
                peaks = detect_peaks(recording=rec, method="locally_exclusive", n_jobs=n_jobs)
                peak_locations = localize_peaks(recording=rec, peaks=peaks, method="monopolar_triangulation", n_jobs=n_jobs)
                np.save(motion_folder / 'peaks.npy', peaks)
                np.save(motion_folder / 'peak_locations.npy', peak_locations)
                # Plotting the motion
                fs = rec.sampling_frequency
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(peaks['sample_index'] / fs, peak_locations['y'], color='k', marker='.',  alpha=0.002)
                ax.set_xlabel('Time (s)')
                plt.savefig(motion_folder / 'motion.png')
                plt.close()
                # Plot peaks on the probe map
                fig, ax = plt.subplots(figsize=(15, 10))
                si.plot_probe_map(rec, ax=ax, with_channel_ids=True)
                ax.scatter(peak_locations['x'], peak_locations['y'], color='purple', alpha=0.002)
                ax.set_ylim(-50, 1000)
                plt.savefig(motion_folder / 'peak_site.png')
                plt.close()

            else:
                peaks = np.load(motion_folder / 'peaks.npy')
                peak_locations = np.load(motion_folder / 'peak_locations.npy', allow_pickle=True)
                print('Found motion folder. Just loaded Peaks and their locations.')
            
            # Estimate motion ---------------------------------------------------------------------
            ## Estimate motion using MEDiCINe
            if 'medicine' in args.motion:
                import medicine
                print('Estimating motion using medicine method')
                # Create directory to store MEDiCINe outputs for this recording
                medicine_output_dir = Path(motion_folder / 'medicine')
                if not medicine_output_dir.exists():
                    medicine_output_dir.mkdir(parents=True, exist_ok=True)
                    # Run MEDiCINe to estimate motion
                    medicine.run_medicine(
                        peak_amplitudes=peaks['amplitude'],
                        peak_depths=peak_locations['y'],
                        peak_times=peaks['sample_index'] / rec.get_sampling_frequency(),
                        output_dir=medicine_output_dir) 

                    motion = np.load(medicine_output_dir / 'motion.npy')
                    time_bins = np.load(medicine_output_dir / 'time_bins.npy')
                    depth_bins = np.load(medicine_output_dir / 'depth_bins.npy')  
                    # Plotting the motion
                    plt.figure(figsize=(2,5))
                    plt.plot(time_bins, motion + depth_bins, color='k')
                    plt.ylabel('depth (um)')
                    plt.xlabel('time (sec)')
                    sns.despine(trim=True)
                    plt.savefig(medicine_output_dir / 'motion.png')
                    plt.close()
                else:
                    print('Found medicine folder already, skipping!')
            if 'decenter' in args.motion:
                from spikeinterface.sortingcomponents.motion.motion_estimation import estimate_motion
                print('Estimating motion using decenteralization method')
                decenter_output_dir = Path(motion_folder / 'decenter')
                if not decenter_output_dir.exists():
                    decenter_output_dir.mkdir(parents=True, exist_ok=True)
                    motion = estimate_motion(recording=rec,
                                             peaks=peaks,
                                             peak_locations=peak_locations,
                                             method="decentralized",
                                             progress_bar=True)
                    np.save(decenter_output_dir / "temporal_bins_s.npy", motion.temporal_bins_s)
                    np.save(decenter_output_dir / "displacement.npy", motion.displacement)
                    np.save(decenter_output_dir / "spatial_bins_um.npy", motion.spatial_bins_um)
                    # Plotting the motion
                    plt.figure(figsize=(2,5))
                    plt.plot(motion.temporal_bins_s[0], motion.displacement[0] + motion.spatial_bins_um, 'k')
                    plt.ylabel('depth (um)')
                    plt.xlabel('time (sec)')
                    sns.despine(trim=True)
                    plt.savefig(decenter_output_dir / 'motion.png')
                    plt.close()
                else:
                    print('Found Decenter folder already, skipping!')

        if args.sort:
            from spikeinterface.sorters import run_sorter
            print('Sorting spikes!')
            ##############################
            # Sort KiloSort4 from raw data
            ##############################
            if (probe_folder / 'kilosort4').exists():
                print('kilosort4 folder already exists, Anicha!')
            else:
                run_sorter(sorter_name='kilosort4', recording=raw_rec, folder= probe_folder / 'kilosort4',
                        nblocks= 3, batch_size= 60000, sig_interp= 20, verbose=True)
                plot_sorting_summary(Path(probe_folder / 'kilosort4'))
            ######################################################################
            if (probe_folder / 'preprocess').exists():
                #######################################
                # Sort KiloSort4 from preprocessed data
                #######################################
                if (probe_folder / 'kilosort4_pre').exists():
                    print('kilosort4_pre folder already exists, Anicha!')
                else:
                    rec = si.read_binary_folder(probe_folder / 'preprocess')
                    run_sorter(sorter_name='kilosort4', recording=rec, folder= probe_folder / 'kilosort4_pre',
                            nblocks= 3, batch_size= 60000, sig_interp= 20, verbose=True)
                    plot_sorting_summary(Path(probe_folder / 'kilosort4_pre'))
                ######################################################################
                # Sort KiloSort4 from preprocessed data and medicine motion estimation
                ######################################################################
                if (probe_folder / 'motion/medicine').exists():
                    from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording
                    from spikeinterface.sortingcomponents.motion import motion_utils
                    # Sort KiloSort4 after motion correction
                    medicine_output_dir = Path(motion_folder / 'medicine')
                    if (probe_folder / 'kilosort4_pre_medicine').exists():
                        print('kilosort4_pre_medicine folder already exists, Anicha!')
                    else:
                        rec = si.read_binary_folder(probe_folder / 'preprocess')
                        rec_float = si.astype(recording=rec, dtype="float32")
                        # Load motion estimated by MEDiCINe
                        motion = np.load(medicine_output_dir / 'motion.npy')
                        time_bins = np.load(medicine_output_dir / 'time_bins.npy')
                        depth_bins = np.load(medicine_output_dir / 'depth_bins.npy')
                        motion_object = motion_utils.Motion(
                            displacement=motion,
                            temporal_bins_s=time_bins,
                            spatial_bins_um=depth_bins,
                        )
                        # Use interpolation to correct for motion estimated by MEDiCINe
                        recording_motion_corrected = InterpolateMotionRecording(
                            rec_float,
                            motion_object,
                            border_mode='force_extrapolate')
                        run_sorter(sorter_name='kilosort4', recording=recording_motion_corrected, folder= probe_folder / 'kilosort4_pre_medicine',
                                    batch_size= 60000, verbose=True, do_correction= False)
                        plot_sorting_summary(Path(probe_folder / 'kilosort4_pre_medicine'))
                ##############################################################################
                # Sort KiloSort4 from preprocessed data and decentralization motion estimation
                ##############################################################################
                if (probe_folder / 'motion/decenter').exists():
                    from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording
                    from spikeinterface.sortingcomponents.motion import motion_utils
                    decenter_output_dir = Path(motion_folder / 'decenter')
                    # Sort KiloSort4 after motion correction
                    if (probe_folder / 'kilosort4_pre_decenter').exists():
                        print('kilosort4_pre_decenter folder already exists, Anicha!')
                    else:
                        rec = si.read_binary_folder(probe_folder / 'preprocess')
                        rec_float = si.astype(recording=rec, dtype="float32")
                        # Load motion estimated by decentralization
                        displacement = np.load(decenter_output_dir / 'displacement.npy')[0]
                        temporal_bins_s = np.load(decenter_output_dir / 'temporal_bins_s.npy')[0]
                        spatial_bins_um = np.load(decenter_output_dir / 'spatial_bins_um.npy')
                        motion_object = motion_utils.Motion(
                            displacement=displacement,
                            temporal_bins_s=temporal_bins_s,
                            spatial_bins_um=spatial_bins_um,
                        )
                        # Use interpolation to correct for motion estimated by decentralization
                        recording_motion_corrected = InterpolateMotionRecording(
                            rec_float,
                            motion_object,
                            border_mode='force_extrapolate')
                        run_sorter(sorter_name='kilosort4', recording=recording_motion_corrected, folder= probe_folder / 'kilosort4_pre_decenter',
                                    batch_size= 60000, verbose=True, do_correction= False)
                        plot_sorting_summary(Path(probe_folder / 'kilosort4_pre_decenter'))