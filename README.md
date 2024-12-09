# Gymalyze

## Project Description
Gymalyze is a project aimed at analyzing exercise data using machine learning models, specifically LSTM networks. The project includes data preprocessing, model training, evaluation, and visualization.

## Installation

### Prerequisites
- Python 3.12 or higher
- `pip` (Python package installer)

### Clone the Repository
```sh
git clone https://github.com/barrtek02/Gymalyze.git
cd Gymalyze
```

### Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### Install Requirements
Install the required packages using pip.


```sh
pip install -r requirements.txt
```
Additionally, to use the GPU version of PyTorch, install the appropriate version using the following command:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124```
```

## Usage

Run the `main.py` script.

License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.