import configparser

__cf = configparser.ConfigParser()
__cf.read("./config.cfg")
TRAIN_IMAGE_PATH = __cf.get("data_path", "train_image_path")
TEST_IMAGE_PATH = __cf.get("data_path", "test_image_path")

