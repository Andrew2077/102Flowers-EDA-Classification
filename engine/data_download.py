import requests, os, tarfile
from tqdm import tqdm


def download(
    url: str, filename: str, force_redownload: bool = False, chunk_size=1024
) -> None:
    """
    Download a file from a url to a local directory, with a progress bar
    Args:
        url (str): download url
        filename (str): filename to save as
        chunk_size (int, optional): chunk size to download at a time. Defaults to 1024.
        force_redownload (bool, optional): whether to force redownload if file already exists. Defaults to False.
    """
    if os.path.exists(os.path.join("data", filename)) and not force_redownload:
        print(f"File {filename} already exists")
    else:
        print(f"Downloading {filename}...")
        request = requests.get(url, stream=True)
        # * Get the total file size in bytes
        total_size = int(request.headers.get("content-length", 0))
        with open(os.path.join("data", filename), "wb") as file, tqdm(
            desc=filename,
            total=total_size,
            unit="iB",  # * unit of measurement, bytes
            unit_scale=True,  # * scale to human readable units
            unit_divisor=chunk_size,  # * divide by chunk size
        ) as bar:
            # * progress bar
            for data in request.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
        # print("Download complete")


def extract_tgz(file_name):
    if os.path.exists(os.path.join("data", file_name.split(".")[0])):
        print("File already extracted")
        return
    else:
        file = tarfile.open(os.path.join("data", file_name))
        samples = file.getmembers()
        print(f"Extracting {file_name}...")
        for sample in tqdm(
            samples, desc=file_name, unit="files", total=len(samples), unit_scale=False
        ):
            file.extract(sample, f"data/{file_name.split('.')[0]}")
        file.close()
        
        
def download_extrac_all(dataset_url, labels, splits, tzg_path, OVERWRITE=False):
    download(dataset_url, dataset_url.split("/")[-1], force_redownload=OVERWRITE)
    download(labels, labels.split("/")[-1], force_redownload=OVERWRITE)
    download(splits, splits.split("/")[-1], force_redownload=OVERWRITE)
    extract_tgz(tzg_path)


