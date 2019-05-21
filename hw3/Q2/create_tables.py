# CS2770 - Nils Murrugarra
import re
import pandas as pd
import numpy as np

# Read all lines from a file
def read_file(in_file, remove_enter=False, flag_strip=False):
    file_ID = open(in_file, "r")
    lines = [line for line in file_ID.readlines()] # all lines
    if remove_enter:
        lines = map(lambda x: x.replace('\n', ''), lines)
    if flag_strip:
        lines = map(lambda x: x.strip(), lines)
    file_ID.close()
    return lines

folder = './'
# files = ['test.txt']
files = ['c1.txt', 'c2.txt', 'c3.txt', 'c4.txt', 'c5.txt', 'c6.txt', 'c7.txt', 'c8.txt']
# parse
for name_file in files:
    lines = read_file(folder + name_file)
    # Get lines that contains "Train Perplexity" and "Valid Perplexity" You can use filter function from python.
    lines_train = filter(lambda x: 'Train Perplexity' in x, lines)
    lines_val = filter(lambda x: 'Valid Perplexity' in x, lines)

    reg_exp_train = 'Epoch: \d+ Train Perplexity: (\d+\.\d+)'
    reg_exp_val = 'Epoch: \d+ Valid Perplexity: (\d+\.\d+)'

    nb_lines_train = len(lines_train)
    nb_lines_val = len(lines_val)
    assert nb_lines_train == nb_lines_val

    perp_train = []
    perp_val = []
    for i in range(nb_lines_val):
        line_train_i = lines_train[i]
        line_val_i = lines_val[i]

        # Next, you need to extract perplexity measures. You can do it with regular expressions.
        re_train = re.match(reg_exp_train, line_train_i)
        re_val = re.match(reg_exp_val, line_val_i)

        perp_train_i = float(re_train.group(1))
        perp_val_i = float(re_val.group(1))
        perp_train.append(perp_train_i)
        perp_val.append(perp_val_i)

    ar_data = np.array([perp_train, perp_val])
    df = pd.DataFrame(ar_data.T, columns=['perplexity_train', 'perplexity_val'])
    df_file = name_file.replace('.txt', '.csv')
    df.to_csv(df_file)
