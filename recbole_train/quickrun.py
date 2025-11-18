
# from recbole.quick_start import run_recbole
from run_fine_tuning import run_recbole
from merge_model import merge_model

# run_recbole(model='GRU4Rec', config_file_list=['/data/hcy/recbole/z_yaml/quickgru.yaml'])
# run_recbole(model='SASRec', config_file_list=['/data/hcy/recbole/z_yaml/quicksasrec.yaml'])
run_recbole(model='Mamba4Rec', config_file_list=['/data/hcy/recbole/z_yaml/invmamba.yaml'])
# run_recbole(model='CL4SRec', config_file_list=['/data/hcy/recbole/z_yaml/quicksasrec.yaml'])
# run_recbole(model='CL4GRU', config_file_list=['/data/hcy/recbole/z_yaml/quickgru.yaml'])
# run_recbole(model='CLMamba', config_file_list=['/data/hcy/recbole/z_yaml/invmamba.yaml'])
# run_recbole(model='FMLP', config_file_list=['/data/hcy/recbole/z_yaml/quicksasrec.yaml'])
# run_recbole(model='STOSASAS', config_file_list=['/data/hcy/recbole/z_yaml/quicksasrec.yaml'])
# run_recbole(model='STOSAGRU', config_file_list=['/data/hcy/recbole/z_yaml/quickgru.yaml'])

# merge_model(model='Mamba4Rec', config_file_list=['/data/hcy/recbole/z_yaml/invmamba.yaml'])
# merge_model(model='GRU4Rec', config_file_list=['/data/hcy/recbole/z_yaml/quickgru.yaml'])

# 1.model in saved directory
# 2.tensorboard --logdir=/data/hcy/recbole/log_tensorboard --port=10086 --host 114.213.208.253 
# 3.yaml is config file