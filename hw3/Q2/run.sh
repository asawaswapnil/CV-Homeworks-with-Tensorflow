#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/opt/cuda-8.0-cuDNN5.1/lib64
source /own_files/cs2770_hw3/rnn/python_rnn/bin/activate

#python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=small # time=18:44.01
# TODO3: run new 8 configurations
# Example: python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c1 > c1.txt
#python ptb_word_lm.py --data_python=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c1 > c1.txt

python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c1 > c1.txt
python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c2 > c2.txt
python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c3 > c3.txt
python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c4 > c4.txt
python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c5 > c5.txt
python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c6 > c6.txt
python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c7 > c7.txt
python ptb_word_lm.py --data_path=/own_files/cs2770_hw3/rnn/simple-examples/data/ --model=c8 > c8.txt
deactivate
