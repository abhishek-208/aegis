import modal
import urllib.request

app = modal.App("aegis-data-setup")
data_volume = modal.Volume.from_name("aegis-dataset", create_if_missing=True)

@app.function(volumes={"/mnt/data_drive": data_volume})
def fetch_dataset():
    print("--- Initiating cloud-to-cloud transfer ---")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    dest = "/mnt/data_drive/cifar10.tar.gz"
    
    # Download using native Python to bypass missing OS tools
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    
    # Lock the data in
    data_volume.commit()
    print("--- Transfer Complete. CIFAR-10 secured in Cloud Volume. ---")

@app.local_entrypoint()
def main():
    print("Commanding Modal to fetch dataset...")
    fetch_dataset.remote()