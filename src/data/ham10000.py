import requests
import zipfile
import io
import os


HAM10000_IMAGES = "https://zip.isic-archive.com/download?zsid=eyJ1c2VyIjpudWxsLCJxdWVyeSI6bnVsbCwiY29sbGVjdGlvbnMiOlsyMTJdfQ:1sBkJK:3hl7ErREGxQzNTAPeiLC6O3aXIfUzbMpB3jQgQzh-Bw"
HAM10000_METADATA = "https://api.isic-archive.com/collections/212/metadata/"


def folder_exists(folder):
    return os.path.isdir(folder)


def download_zip(url, folder_name="data/ham10000/images"):
    r = requests.get(url)
    print(r)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(folder_name)


def main():
    # Where the images are stores
    image_directory = "data/ham10000/images"
    # HAM10000 Images
    if not folder_exists(image_directory):
        print(f"Downloading HAM10000 Images...")
        download_zip(HAM10000_IMAGES, image_directory)
    else:
        print(f"Image folder exists. Skipping HAM10000 Images...")
    # HAM10000 Metadata
    print(f"Downloading HAM10000 Metadata...")
    r = requests.get(HAM10000_METADATA)
    with open("data/ham10000/metadata.csv", "wb") as f:
        f.write(r.content)
    print(f"HAM10000 Dataset Downloaded!")


if __name__ == "__main__":
    main()
