import os
import kagglehub
import shutil

def ensure_path(path: str = "./dataset") -> str:
    """
    Ensure the given path exists. If it doesn't exist, create it (as a directory).

    Path defaults to './dataset', due to the design as a part of an ML project.

    :param path: Optional, path to check. Defaults to './dataset', due to the designe as a part of an ML project.
    :return: String corresponding to the provided or default path
    """

    if not os.path.exists(path):
        os.makedirs(path)
        
    return path

def check_path_not_empty(path: str = "./dataset") -> bool:
    """
    Check if given path is NOT empty.

    - If the path is a file, return True if the file size is greater than 0.
    - If the path is a directory, return True if it contains any files or subdirectories.

    :param path: Optional, path to check. Defaults to './dataset', due to the designe as a part of an ML project.
    :return: True if the path is NOT empty, False otherwise.
    """

    if os.path.isfile(path):
        return os.path.getsize(path) > 0

    if os.path.isdir(path):
        return any(os.scandir(path))

    return False

def search_for_csv(path: str = "./dataset") -> str:
    """
    Searches a path for CSV files.

    :param path: Optional, path to check. Defaults to './dataset', due to the design as a part of an ML project.
    :return: Path to the first CSV file found.
    :raises FileNotFoundError: If no CSV file is found.
    """
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No CSV file found in the directory: {path}")

def download_dataset(dataset: str, destination: str = "./dataset") -> str:
    """
    Downloads dataset and moves it to the destination.

    :param dataset: Kagglehub dataset
    :param destination: Optional, destination to which downloaded dataset is moved. Defaults to './dataset', due to the designe as a part of an ML project.
    :return: Path to the CSV file in the downloaded dataset.
    """

    destination = ensure_path(destination)

    if check_path_not_empty(destination):
        print(f"The destination '{destination}' is not empty. Assuming it contains the desired dataset. Skipping download.")
    else:
        # Download the dataset using Kagglehub
        temp_path = kagglehub.dataset_download(dataset)

        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Dataset could not be downloaded from Kagglehub: {dataset}")

        # Move the dataset to the destination
        for item in os.listdir(temp_path):
            source_item = os.path.join(temp_path, item)
            destination_item = os.path.join(destination, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
            else:
                shutil.copy2(source_item, destination_item)

        print(f"Dataset '{dataset}' has been downloaded and moved to '{destination}'.")

    # Search for a CSV file in the destination
    return search_for_csv(destination)

    

