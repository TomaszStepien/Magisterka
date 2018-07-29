import os

import defaults

def create_folder(folder_structure):
    """divides datasets for x and y

        :param folder_structure: list of strings which will create directory in given order
        :return: ready x and y for given type of dataset
        """
    folder_struct = ""
    for folder in folder_structure:
        folder_struct = os.path.join(folder_struct, folder.replace('/', '').replace('\\', ''))
        try:
            os.makedirs(folder_struct)
        except:
            pass
    return folder_struct

def prepare_folder_structure():
    create_folder([defaults.SAVED_FILES_PATH, "images"])
    create_folder([defaults.SAVED_FILES_PATH, "models"])

