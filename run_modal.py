#modal run --detach run_modal.py::execute_aegis
#modal volume get aegis-results / "D:\IITD\MTP 2\Results"
#modal app logs aegis-simulation --tail 50000 > aegis_logs.txt
import modal
import subprocess
import os

app = modal.App("aegis-simulation")

# 1. Define Environment (Code Only)
# CRITICAL: Do not miss the opening parenthesis on the line below
# 1. Define Environment (Code Only)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "scikit-learn", "matplotlib") 
    .add_local_dir(".", remote_path="/root/aegis", ignore=["venv", ".git", "__pycache__", "saved_models", "aegis-results"])
)
# 2. Attach Cloud Hard Drives
results_volume = modal.Volume.from_name("aegis-results", create_if_missing=True)
data_volume = modal.Volume.from_name("aegis-dataset") 

# 3. Define Execution
@app.function(
    image=image, 
    gpu="T4", 
    timeout=36000, 
    volumes={
        "/root/aegis/saved_models": results_volume,
        "/mnt/data_drive": data_volume
    }
)
def execute_aegis():
    os.chdir("/root/aegis")
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    
    # Extract the tar.gz file from the cloud volume
    print("--- Extracting CIFAR-10 from dedicated data volume ---")
    subprocess.run(["tar", "-xzf", "/mnt/data_drive/cifar10.tar.gz", "-C", "./data/"], check=True)
    
    print("--- Launching AEGIS Simulation ---")
    subprocess.run(["python", "main.py"], check=True)
    
    results_volume.commit()
    print("--- Simulation Complete. Results secured in Modal Volume. ---")