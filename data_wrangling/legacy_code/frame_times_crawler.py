import os
import numpy as np
from pathlib import Path

# From Helpyr
import data_loading
from helpyr_misc import nsplit
from helpyr_misc import ensure_dir_exists
from logger import Logger
from crawler import Crawler




class FrameTimesCrawler (Crawler):
    # The FrameTimesCrawler navigates through the backup data drives and 
    # collects all the image names. The names are timestamps for when images 
    # where taken, therefore can be used to figure out the frame rate for any 
    # particular second. Save the image times in text files

    def __init__(self, destination_dir, log_filepath="./log-files/frame-name-crawler.txt"):
        logger = Logger(log_filepath, default_verbose=True)
        Crawler.__init__(self, logger)

        self.mode_dict['collect_frame_times'] = self.collect_frame_times

        self.set_target_names('*.tif')
        self.destination_dir = destination_dir
        ensure_dir_exists(destination_dir, self.logger)

    def end(self):
        Crawler.end(self)
        self.logger.end_output()

    def collect_frame_times(self):
        self.collect_names(verbose_file_list=False)
        print()
        paths = self.file_list

        # Get the run parameters and frame times. Store in dict for now.
        self.logger.write(f"Extracting run info")
        time_dict = {}
        n_paths = len(paths)
        print_perc = lambda p: print(f"{p:4.0%}", end='\r')
        i_tracker = 0
        for i, path in enumerate(paths):
            if (i / n_paths) >= i_tracker:
                print_perc(i_tracker)
                i_tracker += 0.1
            _, exp_code, step, period, file = nsplit(path, 4)
            time_str, ext = file.rsplit('.', 1)
            
            key = (exp_code, step, period)
            if key in time_dict:
                time_dict[key].append(np.float(time_str))
            else:
                time_dict[key] = [np.float(time_str)]

        self.logger.write(f"Writing times files")
        self.logger.increase_global_indent()
        npy_paths = []
        for key in time_dict:
            (exp_code, step, period) = key
            times = np.sort(np.array(time_dict[key]))

            self.logger.write(f"Found {len(times)} images for {exp_code} {step} {period}")

            times_filename = f"{exp_code}_{step}_{period}_frame_times.npy"
            times_filepath = os.path.join(self.destination_dir, times_filename)

            np.save(times_filepath, times)
            npy_paths.append(times_filepath)

        self.logger.decrease_global_indent()

        npy_list_filepath = os.path.join(self.destination_dir, 'npy_list.txt')
        npy_path = Path(npy_list_filepath)

        if npy_path.is_file():
            # Add to existing records. Ignores duplicates.
            with npy_path.open() as fid:
                existing_paths = fid.read().splitlines()

            self.logger.write(f"Adding {len(time_dict)} files to {len(existing_paths)} existing files.")
            npy_paths = list(filter(None, set(existing_paths) | set(npy_paths)))

        else:
            self.logger.write(f"{len(time_dict)} files written")

        with npy_path.open('w') as fid:
            fid.write('\n'.join(npy_paths))
        self.logger.write("Done!")


if __name__ == "__main__":
    destination_dir = '/home/alex/feed-timing/data/extracted-lighttable-results/frame-times'
    crawler = FrameTimesCrawler(destination_dir)
    exp_root = '/run/media/alex/Alex4/lighttable-data'
    crawler.set_root(exp_root)
    crawler.run()
