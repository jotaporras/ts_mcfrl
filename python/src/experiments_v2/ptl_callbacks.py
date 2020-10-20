import logging
import shutil

from pytorch_lightning import Callback
import os
import wandb

from experiment_utils import report_generator


class MyPrintingCallback(Callback):
    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_epoch_end(self,trainer,pl_module):
        print("epoch is done")

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')

class WandbDataUploader():
    def __init__(self, base="data/results"):
        self.base = base

    def upload(self, experiment_name):
        '''
            Copies the base directory of experiment results to the wandb directory for upload
        :param experiment_name:
        :return:
        '''
        # copy base to wandb.
        shutil.copytree(os.path.join(self.base,experiment_name), os.path.join(wandb.run.dir,self.base,experiment_name))


class ShippingFacilityEnvironmentStorageCallback(Callback):
    '''
        Stores the information objects into CSVs for debugging Environment and actions.
    '''
    def __init__(self, experiment_name,base,experiment_uploader:WandbDataUploader):
        self.experiment_name = experiment_name
        self.base = base
        self.experiment_uploader = experiment_uploader

    def on_train_end(self, trainer, pl_module):
        logging.info("Finished training, writing environment info objects")
        report_generator.write_single_df_experiment_reports(pl_module.episodes_info, self.experiment_name,self.base)
        report_generator.write_generate_physical_network_valid_dcs(pl_module.env.environment_parameters.network, self.experiment_name,self.base)

        self.experiment_uploader.upload(self.experiment_name)
