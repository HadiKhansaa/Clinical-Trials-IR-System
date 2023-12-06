import shutil
import os
import json

def copy_file(src, dst):
    """
    Copy a file from the source path to the destination path.

    :param src: str - The path of the source file.
    :param dst: str - The path of the destination directory.
    """
    try:
        shutil.copy(src, dst)
        print(f"File copied successfully from {src} to {dst}")
    except FileNotFoundError:
        print(f"File not found: {src}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    PATH_TO_TRIALS = 'trials'

    # get documents with relevence feedback
    with open('documents.json', 'r') as file:
        docs = json.load(file)
    documents = set(docs)
    i = 1
    for folder in os.listdir(PATH_TO_TRIALS):
        for sub_folder in os.listdir(os.path.join(PATH_TO_TRIALS, folder)):
            for file in os.listdir(os.path.join(PATH_TO_TRIALS, folder, sub_folder)):
                with open(os.path.join(PATH_TO_TRIALS, folder, sub_folder, file), 'r') as f:
                    if file.endswith('.xml') and file[:-4] in documents: # check if file is in documents with relevence feedback
                        #copy the file to trials_query1 folder
                        copy_file(os.path.join(PATH_TO_TRIALS, folder, sub_folder, file), 'trials_query1')
                        print(f"Processed {i} files")
                        i += 1

