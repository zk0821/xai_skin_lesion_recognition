import requests
import zipfile
import io
import os

HAM10000_PART_ONE = "https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/DBW86T/163c862634d-663b42fb343d?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27HAM10000_images_part_1.zip&response-content-type=application%2Fzip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240527T224840Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20240527%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=99d00da1ae32bb9344b919bd60f53f21b8ae16ca9bd73210114164de5c787c82"
HAM10000_PART_TWO = "https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/DBW86T/163c85dbd0a-b565e4ed4c74?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27HAM10000_images_part_2.zip&response-content-type=application%2Fzip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240527T225845Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20240527%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8f29d33b5ee80b5619ceef2b749107b2a9e958774003e9968506032dd5cf3e71"
HAM10000_METADATA = "https://dvn-cloud.s3.amazonaws.com/10.7910/DVN/DBW86T/1774dfb2763-002d253ecea5.orig?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27HAM10000_metadata&response-content-type=text%2Fcsv&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240527T225930Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Credential=AKIAIEJ3NV7UYCSRJC7A%2F20240527%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=0b7fdc6e8bae66f89c7533edbdfe5dc801f1f056d50f272344f31e0197703a65"


def folder_exists(folder):
    return os.path.isdir(folder)


def download_zip(url, folder_name="data/ham10000/images"):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(folder_name)


def main():
    # Where the images are stores
    image_directory = "data/ham10000/images"
    # HAM10000 Part One
    if not folder_exists(image_directory):
        print(f"Downloading HAM10000 Part One...")
        download_zip(HAM10000_PART_ONE, image_directory)
    else:
        print(f"Image folder exists. Skipping HAM10000 Part One...")
    # HAM10000 Part Two
    if not folder_exists(image_directory):
        print(f"Downloading HAM10000 Part Two...")
        download_zip(HAM10000_PART_TWO, image_directory)
    else:
        print(f"Image folder exists. Skipping HAM10000 Part Two...")
    # HAM10000 Metadata
    print(f"Downloading HAM10000 Metadata...")
    r = requests.get(HAM10000_METADATA)
    with open("data/ham10000/metadata.csv", "wb") as f:
        f.write(r.content)
    print(f"HAM10000 Dataset Downloaded!")
    # Assertions
    _, _, images = next(os.walk(image_directory))
    assert len(images) == 10_015
    assert os.path.isfile("data/ham10000/metadata.csv")


if __name__ == "__main__":
    main()
