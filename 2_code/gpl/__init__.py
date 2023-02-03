import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)) ).parent.parent # GraphPatternLearning/
CONFIG_DIR = ROOT_DIR/'0_config'
DATA_DIR = ROOT_DIR/'1_data'
LOG_DIR = ROOT_DIR/'5_output'/'log' # to .log files
TBLOG_DIR = ROOT_DIR/'5_output'/'tb_log'
TBLOG_HPARAMS_DIR = ROOT_DIR/'5_output'/'tb_log_hparams'
CKPT_DIR = ROOT_DIR/'5_output'/'ckpts'
RESULT_DIR = ROOT_DIR/'6_result' # maybe some csv files
FIGURES_DIR = ROOT_DIR/'7_figure'
TMP_DIR = ROOT_DIR/'8_tmp'

