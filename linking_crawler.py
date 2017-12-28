import os

from crawler import Crawler

class LinkingCrawler (Crawler):

    def __init__(self, log_filepath="./log-crawler.txt"):
        Crawler.__init__(self, log_filepath)
        self.set_links_dir("./links/")

        self.mode_dict['2m-photos'] = self.run_2m_photos
        self.mode_dict['8m-photos'] = self.run_8m_photos
        self.mode_dict['8m-lasers'] = self.run_8m_lasers
        self.mode_dict['manual']    = self.run_manual_data


    def set_links_dir(self, links_dir):
        self.links_dir = links_dir
        self.write_log(["Links dir set to " + links_dir])

    def populate_links_dir(self, make_subdirs=True, rename_fu=None):
        # Redefine the Crawler version to allow for subdirectories based on 
        # experiment names (1A, 1B, 2A, etc.)
        
        if not os.path.isdir(self.links_dir):
            msg = "Making links directory at {}.".format(self.links_dir)
            self.write_log([msg], True)
            os.makedirs(self.links_dir, exist_ok=True)

        if make_subdirs:
            sub_dirs = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '5A',]
            for sub_dir in sub_dirs:
                sub_path = os.path.join(self.links_dir, sub_dir)
                if not os.path.isdir(sub_path):
                    msg = "Making links directory at {}.".format(sub_path)
                    self.write_log([msg], True)
                    os.makedirs(sub_path, exist_ok=True)

        for filepath in self.file_list:
            path, filename = os.path.split(filepath)
            sub_dir = filename[0:2] if make_subdirs else ''
            if rename_fu:
                filename = rename_fu(filename, path)
            linkpath = os.path.join(self.links_dir, sub_dir, filename)
            if os.path.exists(linkpath):
                msg = "Link for {} exists already. Skipping.".format(linkpath)
                self.write_log([msg], True, indent=4*' ')
            else:
                msg = "Making symlink for {}.".format(linkpath)
                self.write_log([msg], True, indent=4*' ')
                os.symlink(filepath, linkpath)


    def rename_cart_file(self, filename, source_path):
        try:
            exp, step, time, length, name = filename.split('-')
        except ValueError:
            print(filename)
            raise ValueError
        limb = step[0]
        flow = int(step[1:-1])
        time = int(time[1:])
        length = length[0]

        limb_bin = 0 if 'r' == limb else 1

        order = flow + 2 * limb_bin * (100 - flow) + int(time/10)

        new_name = "{}-{:03d}-{}{}L-t{}-{}m-{}".format(
                exp, order, limb, flow, time, length, name)

        msg = "Renaming {} to {}".format(filename, new_name)
        self.write_log([msg], True, 4*' ')
        return new_name

    def rename_manual_file(self, filename, source_path):
        exp, name = filename.split('-', 1)
        return "{}-{}.xlsx".format(name[:-5], exp)


    def run_2m_photos(self):
        self.write_log_section_break()
        self.write_log(["Running 2m photos"], verbose=True)
        self.set_target_names(['*2m-Composite.JPG',])
        self.set_links_dir('/home/alex/ubc/research/feed-timing/data/data-links/2m-photos')
        self.collect_names()
        self.populate_links_dir(rename_fu=self.rename_cart_file)
        self.end()

    def run_8m_photos(self):
        self.write_log_section_break()
        self.write_log(["Running 8m photos"], verbose=True)
        self.set_target_names(['*8m-Composite.JPG',])
        self.set_links_dir('/home/alex/ubc/research/feed-timing/data/data-links/8m-photos')
        self.collect_names()
        self.populate_links_dir(rename_fu=self.rename_cart_file)
        self.end()

    def run_8m_lasers(self):
        self.write_log_section_break()
        self.write_log(["Running 8m laser scans"], verbose=True)
        self.set_target_names(['*beddatafinal.txt',
                                  '*laser.*',
                                 ])
        self.set_links_dir('/home/alex/ubc/research/feed-timing/data/data-links/8m-lasers')
        self.collect_names()
        self.populate_links_dir(rename_fu=self.rename_cart_file)
        self.end()

    def run_manual_data(self):
        self.write_log_section_break()
        self.write_log(["Running manual data"], verbose=True)
        self.set_root('/home/alex/ubc/research/feed-timing/data/manual-data')
        self.set_target_names(['??-flow-depths.xlsx',
                               '??-masses.xlsx',
                              ])
        self.set_links_dir('/home/alex/ubc/research/feed-timing/data/data-links/manual-data')
        self.collect_names()
        self.populate_links_dir(make_subdirs=False, rename_fu=self.rename_manual_file)
        self.end()



if __name__ == "__main__":
    crawler = LinkingCrawler()
    exp_root = '/home/alex/ubc/research/feed-timing/data/{}'
    crawler.set_root(exp_root.format('cart'))
    crawler.run('manual')
