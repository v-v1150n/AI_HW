# Installation Instructions

1. Creat the virtual env:
    ```bash
    conda create -y -n ai_hw6 python=3.10
    ```

2. Activate the virtual env:
    ```bash
    conda activate ai_hw6
    ```

3. Install pytorch based on your cuda version:
    ```bash
    nvidia-smi #(verify the cuda version)
    ```
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4. Install the required packages:
    ```bash
    pip install --no-deps trl peft accelerate bitsandbytes
    ```

    ```bash
    pip install tqdm packaging wandb
    ```

5. Based on cuda version install the correct version of unsloth:
    For Pytorch 2.3.0: Use the "ampere" path for newer RTX 30xx GPUs or higher.
    ```bash
    pip install "unsloth[cu118-torch230] @ git+https://github.com/unslothai/unsloth.git"
    pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
    pip install "unsloth[cu118-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
    pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
    ```

6. Verify if the installation was successful.:
    ```bash
    nvcc
    python -m xformers.info
    python -m bitsandbytes
    ```

# Training the Model

Please follow the steps below to train your model:

1. Navigate to the code directory:
    ```bash
    cd python_file/AI2024-hw6/
    ```

2. Run the training script:
    ```bash
    bash run.sh DPO unsloth/llama-3-8b-Instruct-bnb-4bit e3fff50ea094a01b2517ed5ab652f25906a498f0 5
    ```

    ```bash
    bash run.sh ORPO unsloth/llama-3-8b-Instruct-bnb-4bit e3fff50ea094a01b2517ed5ab652f25906a498f0 5
    ```

3. Run the inference script:
    ```bash
    bash inference.sh unsloth/llama-3-8b-Instruct-bnb-4bit e3fff50ea094a01b2517ed5ab652f25906a498f0 
    ```

