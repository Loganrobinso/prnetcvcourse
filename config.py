import os
import json
from jsonschema import validate
from prnet.utils import setup_logger

# # Setup a logger
# logger = setup_logger(__name__)

# **********************************************************************
# Description:
#   Handle configuration of experiments 
# Parameters:
#   config_file - File containing desired user parameters
#   mode - Either training mode of testing mode
# Notes:
#   -
# **********************************************************************
class Config:
    def __init__(self, config_file, mode = 'train'):
        
        self.mode = mode
        self.load_defaults(config_file)
        # self.validate_schema()

        pretty_config = json.dumps(self.config_dic, indent = 3)
        # logger.info(pretty_config)

        # Create the checkpoint and log directories if they are missing
        if not os.path.exists(self.config_dic['log_dir']):
            os.mkdir(self.config_dic['log_dir'])

        if not os.path.exists(self.config_dic['checkpoint_dir'] ):
            os.mkdir(self.config_dic['checkpoint_dir'] )


    # **********************************************************************
    # Description:
    #   Loads default values for unspecified values in the config file
    # Parameters:
    #   config_file - File containing desired user parameters
    # Notes:
    #   The default settings json is assumed to be in the configs folder.
    # **********************************************************************
    def load_defaults(self, config_file):
        # logger.info('Reading default config file')
        f = open('./configs/default_settings.json')
        self.config_dic = json.load(f)
        f.close()

        # logger.info('Reading user config file')
        f = open(config_file)
        user_settings = json.load(f)
        f.close()

        # Override user preferences in default settings
        for key, value in user_settings.items():
            self.config_dic[key] = value