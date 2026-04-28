import subprocess
import sys
import re


# Max CUDA version for each PyTorch version supported by PyG
# Check the list shown on https://data.pyg.org/whl/
# Last updated Oct 2025
TORCH_MAX_CUDA = {
    "2.8": "12.9",
    "2.7": "12.8",
    "2.6": "12.6",
    "2.5": "12.4",
}


def main():
    max_torch = list(map(int, sorted(TORCH_MAX_CUDA.keys())[-1].split('.')))

    try:
        import torch
        torch_version_str = torch.__version__.split('+')[0]
    except ImportError:
        print("PyTorch not installed. Please install PyTorch first:")
        print(f'  pip install "torch<={max_torch[0]}.{max_torch[1]}"')
        sys.exit(1)

    vmatch = re.match(r"(\d+\.\d+)", torch_version_str)
    cur_torch_str = vmatch.group(1)
    cur_torch = list(map(int, cur_torch_str.split('.')))

    if cur_torch[0] > max_torch[1] \
            or (cur_torch[0] == max_torch[0] and cur_torch[1] > max_torch[1]):
        print("Unsupported version of PyTorch. "
              "Please install a supported version:")
        print(f'  pip install "torch<={max_torch[0]}.{max_torch[1]}"')
        sys.exit(1)

    max_cuda = list(map(int, TORCH_MAX_CUDA[cur_torch_str].split(".")))
    cuda_ver_supported = False
    if torch.cuda.is_available():
        cur_cuda = list(map(int, torch.version.cuda.split('.')))
        if cur_cuda[1] < max_cuda[1] \
                or (cur_cuda[0] == max_cuda[0] and cur_cuda[1] <= max_cuda[1]):
            cuda_ver_supported = True
        else:
            print("Warning: This CUDA version "
                  f"(together with PyTorch {cur_torch_str}) "
                  "is unsupported by PyG.")
            print("\tCheck the list shown on https://data.pyg.org/whl/ "
                  "for supported PyTorch-CUDA versions.")
            print("\tFalling back to the CPU version of CUDA for now...")

    cuda_version_str = f"cu{cur_cuda[0]}{cur_cuda[1]}" \
        if cuda_ver_supported else "cpu"

    print(f"Detected PyTorch version: {torch_version_str}")
    print(f"Detected CUDA version: {cuda_version_str}")

    # Construct URL based on torch and CUDA versions
    find_links = "https://data.pyg.org/whl/" \
        f"torch-{torch_version_str}+{cuda_version_str}.html"

    print(f"Installing from: {find_links}")

    # The PyG C++ extensions are only needed by the optional group_averaging
    # module (torch_scatter is imported in group_averaging/model/blocks.py and
    # torch_cluster is needed by torch_geometric.nn.radius_graph in
    # group_averaging/utils/ga_utils.py). The core BAM-torch path uses the
    # pure-PyTorch scatter in bam_torch.utils.scatter and does not require any
    # of these extensions. Run this script only if you intend to use
    # `pip install -e ".[group_averaging]"`.
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--find-links", find_links,
            "torch_scatter", "torch_cluster"
        ])
        print("Successfully installed torch_scatter and torch_cluster")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        print(f"You may need to manually install from: {find_links}")
        sys.exit(1)


if __name__ == "__main__":
    main()
