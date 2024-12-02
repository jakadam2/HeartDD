import os
import configparser

CONFIG_FILE = "config.ini"
root_dir = os.path.join(os.path.dirname(__file__), "..")
config_path = os.path.abspath(CONFIG_FILE)

parser = configparser.ConfigParser()
config = parser.read(config_path)