from data_generate.data import Data
from data_generate.data_generation import DataGeneration
from log.log import Log

data = DataGeneration().read_data()
data.pre_hemodynamic_state = data.hemodynamic_state
data.post_calculation(cal_los=False)
Log().save_mat(data=data, epoch=119999, is_train=False)

