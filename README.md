# Bladebit Chia Plotter

[![Release Builds](https://github.com/Chia-Network/bladebit/actions/workflows/build-release.yml/badge.svg?branch=master&event=push)](https://github.com/Chia-Network/bladebit/actions/workflows/build-release.yml)

A high-performance **k32-only**, Chia (XCH) plotter.

Bladebit supports 3 plotting modes:
 - Fully In-RAM (no drives required), CPU-based mode.
 - GPU (CUDA-based) mode. Both fully in-RAM or disk-hybrid mode.
 - Disk-based mode

## Usage
Run `bladebit --help` to see general help. For command-specific help, use `bladebit help <command_name>`.

## Requirements

**CUDA**

An NVIDIA GPU is required for this mode. This mode is exposed via the `cudaplot` command in a separate executable "bladebit_cuda". This mode has mainly been tested on consumer cards from the **10xx** series and up. 

| Mode                           | OS             | DRAM | VRAM | CUDA capability 
|--------------------------------|----------------|------|------|----------------
| In-RAM                         | Linux, Windows | 256G | 8G   | 5.2 and up
| Disk-hybrid 128G               | Linux, Windows | 128G | 8G   | 5.2 and up
| Disk-hybrid 16G (WIP)          | Linux          | 16G  | 8G   | 5.2 and up

> *NOTE: 16G mode currently a work in progress and at this stage it only works in Linux and direct I/O is unavailable in this mode.*


**CPU RAM-Only**

Available on Linux, Windows and macOS.
Requires at least **416G** of system DRAM.


**Disk**

Available on Linux, Windows and macOS.

A minimum of **4 GiB of RAM** is required, with lower bucket counts requiring up to 12 GiB of RAM. Roughly **480 GiB of disk space** is required in the default mode, or around **390 GiB of disk space** with `--alternate` mode enabled.

The exact amounts of RAM and disk space required may vary slightly depending on the system's page size and the target disk file system block size (block-alignment is required for direct I/O).

SSDs are highly recommended for disk-based plotting.


## Compressed Plots

Compressed plots are supported in CUDA mode and in RAM-only mode. CPU Disk-based mode does **NOT** currently support compressed plots.

Compressed plots are currently supported for compression levels from **C1** to **C7**. Note that bladebit compression levels are not compatible with other plotter compression levels. These compression levels are based on the *number of bits dropped from an entry excluding the minimum bits required to fully drop a table*. At `k=32` a the first table is fully excluded from the plot at 16 bits dropped.

> *NOTE: Although higher compression levels are available, support for farming them has not been currently implemented and are therefore disabled. They will be implemented in the future.*

Compression levels are currently roughly equivalent to the following plot sizes.

| Compression Level | Plot Size
|-------------------|-------------
| C1                | 87.5 GiB
| C2                | 86.0 GiB
| C3                | 84.4 GiB
| C4                | 82.8 GiB
| C5                | 81.2 GiB
| C6                | 79.6 GiB
| C7                | 78.0 GiB

These might be optimized in the future with further compression optimizations.


## Requirements

## **GPU (CUDA) Plotter Requirements**


**Supported system configurations for alpha:**
 
|||
|------------|-------------------------------------------------------------------------------
| **OS**     | Windows and Linux                                                             
| **Memory** | **256GB** of system DRAM                                                      
| **GPUs**   | NVIDIA GPUs with CUDA capability **5.2** and up with at least **8GB** of vRAM 
|            |                                                                               

> See https://developer.nvidia.com/cuda-gpus for compatible GPUs.

<br/>

### In-RAM
**416 GiB of RAM are required** to run it, and a few more megabytes for stack space and small allocations.

64-bit is supported only, for obvious reasons.


### Disk-based
A minimum of **4 GiB of RAM** is required, with lower bucket counts requiring up to 12 GiB of RAM. 

Around **480 GiB** of total temporary space is required when plotting to disk in the default mode, or around 390 GiB with `--alternate` mode enabled.

The exact amounts of RAM and disk space required may vary slightly depending on the system's page size and the target disk file system block size (block-alignment is required for direct I/O).

SSDs are highly recommended for disk-based plotting.


## Prerequisites
Linux, Windows and macOS (both Intel and ARM) are supported.


### Linux

Install the following packages:
```bash
# CentOS or RHEL-based
sudo yum group install -y "Development Tools"
sudo yum install -y cmake gmp-devel numactl-devel

# Ubuntu or Debian-based
sudo apt install -y build-essential cmake libgmp-dev libnuma-dev
```

### Windows
Must have at least [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) or its build tools installed.

### macOS
Must have Xcode or Xcode build tools installed.
`brew install cmake`

Optionally install `gmp`:
`brew install gmp`


## Building

```bash

# Clone the repo & its submodules
git clone https://github.com/Chia-Network/bladebit.git && cd bladebit

# Create a build directory for cmake and cd into it
mkdir -p build && cd build

# Generate config files & build
cmake ..
cmake --build . --target bladebit --config Release
```

The resulting binary will be found under the `build/` directory.
On Windows it will be under `build/Release/`.

For **bladebit_cuda**, the CUDA toolkit must be installed. The target name is `bladebit_cuda`.

For simplicity the `build.sh` or `build-cuda.sh` scripts can be used to build. On Windows this requires gitbash or similar bash-based shell to run.

## Usage
Run **bladebit** (or **bladebit_cuda**) with the `-h` for complete usage and command line options:

```bash
# Linux & macOS
build/bladebit -h

# Windows
build/Release/bladebit.exe -h
```
The bladebit CLI uses the format `bladebit <GLOBAL_OPTIONS> <command> <COMMAND_OPTIONS>`.

Use the aforementioned `-h` parameter to get the full list of commands and `GLOBAL_OPTIONS`. 
The `command`-specific `COMMAND_OPTIONS` can be obtained by using the `help` sub command with the desired command as the parameter: 

```bash
bladebit help cudaplot
bladebit help ramplot
bladebit help diskplot
```

### CUDA
Basic `cudaplot` usage:
```bash
# OG plots
./bladebit_cuda -f <farmer_public_key> -p <pool_public_key> cudaplot <output_directory>

# Portable plots
./bladebit_cuda -f <farmer_public_key> -c <pool_contract_address> cudaplot <output_directory>

# Compressed plots
./bladebit_cuda -z <copression_level> -f <farmer_public_key> -c <pool_contract_address> cudaplot <output_directory>

# 128G disk-hybrid mode
./bladebit_cuda -z <copression_level> -f <farmer_public_key> -c <pool_contract_address> cudaplot --disk-128 -t1 <temp_dir> <output_directory>
```

### In-RAM
Basic `ramplot` usage:
```bash
# OG plots
./bladebit -f <farmer_public_key> -p <pool_public_key> ramplot <output_directory>

# Portable plots
./bladebit -f <farmer_public_key> -c <pool_contract_address> ramplot <output_directory>

# Compressed plots
./bladebit -z <copression_level> -f <farmer_public_key> -c <pool_contract_address> ramplot <output_directory>
```

### Disk-Based
Basic `diskplot` usage:
```bash

# OG plots
./bladebit -f <farmer_public_key> -p <pool_public_key> diskplot -t1 <temp_directory> <output_directory>

# Portable plots
./bladebit -f <farmer_public_key> -c <pool_contract_address> diskplot -t1 <temp_directory> <output_directory>

# Differing temp directories:
./bladebit -f ... -c ... diskplot -t1 /path/to/temp_1 -t2 /path/to/temp2 /my/output/dir

# With a 100 GiB temp2 cache and alternating mode
./bladebit -f ... -c ... diskplot -a --cache 100G -t1 /path/to/temp_1 -t2 /path/to/temp2 /my/output/dir

# With fine-grained thread control depending on the workload
./bladebit -f ... -c ... diskplot --f1-threads 12 --fp-threads 32 -t1 /path/to/temp_1  /my/output/dir
```


## License
Licensed under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). See [LICENSE](LICENSE).


Copyright 2023 Harold Brenes, Chia Network Inc

