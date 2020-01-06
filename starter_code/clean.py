import argparse
import csv
import os

EXP_ROOT = '/Users/michaelchang/Documents/Researchlink/Berkeley/auction_project/runs'

# clean the checkpoints
# clean the gifs

parser = argparse.ArgumentParser()
parser.add_argument('--for-real', action='store_true')
args = parser.parse_args()

class Cleaner():
    def __init__(self, exp_subroot):
        self.exp_subroot = exp_subroot

    def clean_checkpoints(self, exp_dirs):
        for exp_dir in exp_dirs:
            print('Removing from {}'.format(exp_dir))
            fp_exp_dir = os.path.join(EXP_ROOT, self.exp_subroot, exp_dir)
            for seed_dir in [x for x in os.listdir(fp_exp_dir) if 'seed' in x]:
                print('Removing from {}'.format(seed_dir))
                fp_seed_dir = os.path.join(fp_exp_dir, seed_dir)
                for mode_dir in [x for x in os.listdir(os.path.join(fp_seed_dir, 'group_0')) if 'train' in x or 'test' in x]:
                    fp_mode_dir = os.path.join(fp_seed_dir, 'group_0', mode_dir)
                    print('Removing from {}'.format(mode_dir))
                    with open(os.path.join(fp_mode_dir, 'checkpoints', 'summary.csv'), 'r') as f:
                        ckpt_csv_reader = csv.DictReader(f)
                        last_row = list(ckpt_csv_reader)[-1]
                        recent_ckpt = last_row['recent']
                        best_ckpt  = last_row['best']

                        ckpt_files = [x for x in os.listdir(os.path.join(fp_mode_dir, 'checkpoints')) if '.pth.tar' in x]
                        # print(ckpt_files)
                        for ckpt_file in ckpt_files:
                            if ckpt_file not in [recent_ckpt, best_ckpt]:
                                fp_ckpt_file = os.path.join(
                                    fp_mode_dir, 'checkpoints', ckpt_file)
                                # remove    
                                print('Removing {}'.format(fp_ckpt_file))
                                if args.for_real:
                                    os.remove(fp_ckpt_file)

    def clean_gifs(self, exp_dirs):
        def get_ckpt_epoch(ckpt_file_name):
            return int(ckpt_file_name[len('ckpt_batch'):-len('.pth.tar')])

        for exp_dir in exp_dirs:
            print('Removing from {}'.format(exp_dir))
            fp_exp_dir = os.path.join(EXP_ROOT, self.exp_subroot, exp_dir)
            for seed_dir in [x for x in os.listdir(fp_exp_dir) if 'seed' in x]:
                print('Removing from {}'.format(seed_dir))
                fp_seed_dir = os.path.join(fp_exp_dir, seed_dir)
                for mode_dir in [x for x in os.listdir(os.path.join(fp_seed_dir, 'group_0')) if 'train' in x or 'test' in x]:
                    fp_mode_dir = os.path.join(fp_seed_dir, 'group_0', mode_dir)
                    print('Removing from {}'.format(mode_dir))
                    with open(os.path.join(fp_mode_dir, 'checkpoints', 'summary.csv'), 'r') as f:
                        ckpt_csv_reader = csv.DictReader(f)
                        last_row = list(ckpt_csv_reader)[-1]
                        ##### basically everything above is the same #####
                        recent_ckpt_epoch = get_ckpt_epoch(last_row['recent'])
                        best_ckpt_epoch  = get_ckpt_epoch(last_row['best'])

                        gif_files = [x for x in os.listdir(os.path.join(fp_mode_dir, 'qualitative')) if '.gif' in x]

                        for gif_file in gif_files:
                            trimmed_gif_file = gif_file[:gif_file.rfind('_')]
                            gif_epoch = int(trimmed_gif_file[trimmed_gif_file.rfind('_')+1:])

                            if gif_epoch not in [recent_ckpt_epoch, best_ckpt_epoch]:
                                fp_gif_file = os.path.join(fp_mode_dir, 'qualitative', gif_file)
                                print('Removing {}'.format(fp_gif_file))
                                if args.for_real:
                                    os.remove(fp_gif_file)

def clean_1_2_20_debug_plot():
    c = Cleaner(exp_subroot='debug_plot')
    c.clean_checkpoints([
        'CP-0_plr4e-05_optadam_ppo_aucv_red2_ec0',
        'CP-0_plr4e-05_optadam_ppo_aucbb_red2_ec0'
        ])

    c.clean_gifs([
        'CP-0_plr4e-05_optadam_ppo_aucv_red2_ec0', 
        'CP-0_plr4e-05_optadam_ppo_aucbb_red2_ec0'])


def clean_1_2_20_debug_plot_hdim32():
    c = Cleaner(exp_subroot='debug_plot_hdim32')
    c.clean_checkpoints([
        'CP-0_plr4e-05_optadam_ppo_aucbb_red2_ec0',
        'CP-0_plr4e-05_optadam_ppo_aucv_red2_ec0'
        ])

    c.clean_gifs([
        'CP-0_plr4e-05_optadam_ppo_aucbb_red2_ec0', 
        'CP-0_plr4e-05_optadam_ppo_aucv_red2_ec0'])

if __name__ == '__main__':
    clean_1_2_20_debug_plot_hdim32()
