import requests
import tarfile

def download_file(url, filename):
    print(f'Downloading from {url} to {filename}')
    response = requests.get(url)
    with open(filename,  'wb') as ofile:
        ofile.write(response.content)


def unzip_data(fname, outpath):
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(outpath)
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall(outpath)
        tar.close()
