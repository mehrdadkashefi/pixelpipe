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
    parser.add_argument("--preprocess", type=bool, default=True,
                            help="If True, preprocess the data") 
    parser.add_argument("--motion", type=list_of_strs, default=[],
                            help="If true, estimate motion")
    parser.add_argument("--sort", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                            help="If true, sort the data")
    
    args = parser.parse_args()
    return args

    


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

        # Estimate motion
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_localization import localize_peaks

        if not (probe_folder / 'preprocess').exists():
            print('No preprocess folder found. Estimating motion from raw data!')
            rec = si.read_spikeglx(probe_folder, stream_name='imec'+ str(probe_i) +'.ap', load_sync_channel=False)
        else:
            print('Found preprocess folder. Estimating motion from preprocessed data!')
            rec = si.read_binary_folder(probe_folder / 'preprocess')
            print(rec)

        if len(args.motion) != 0:
            print('Following motion estimation methods will be used:', args.motion)
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
                        output_dir=medicine_output_dir,)   
                else:
                    print('Found medicine folder already, skipping!')

        if args.sort:
            from spikeinterface.sorters import run_sorter
            print('Sorting spikes!')
            # Sort KiloSort4 from raw data
            if (probe_folder / 'kilosort4').exists():
                print('kilosort4 folder already exists, Anicha!')
            else:
                run_sorter(sorter_name='kilosort4', recording=raw_rec, folder= probe_folder / 'kilosort4',
                        nblocks= 3, batch_size= 60000, sig_interp= 20, verbose=True)
                
            # Sort KiloSort4 from preprocessed data
            if (probe_folder / 'kilosort4_pre').exists():
                print('kilosort4_pre folder already exists, Anicha!')
            else:
                rec = si.read_binary_folder(probe_folder / 'preprocess')
                run_sorter(sorter_name='kilosort4', recording=rec, folder= probe_folder / 'kilosort4_pre',
                        nblocks= 3, batch_size= 60000, sig_interp= 20, verbose=True)
                
            if (probe_folder / 'motion/medicine').exists():
                from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording
                from spikeinterface.sortingcomponents.motion import motion_utils

                print('Found medicine motion estimation, sorting with it!')
                # Sort KiloSort4 after motion correction
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
                        nblocks= 3, batch_size= 60000, sig_interp= 20, verbose=True, do_correction= False)

                    
                    


            

            





