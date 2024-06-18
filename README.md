# Transformer Model for Time Series Prediction

This repository contains a simplified implementation of a Transformer model for time series prediction. The purpose of this project is to provide an easy-to-understand example of how Transformer models can be applied to time series data, making it accessible for beginners.

## Features

- Simple implementation with minimal files
- Generates random input data for demonstration purposes
- Allows users to save the trained model
- Clear and concise code to facilitate learning

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Zdong104/TimeSeriesTransformer.git
    cd TimeSeriesTransformer
    ```

2. Install Miniconda:

    Miniconda is a minimal installer for conda. You can download and install it from the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html). Scroll to the button, use Quick install. 

    After downloading the installer, run it and follow the instructions to complete the installation.

3. Create a virtual environment and activate it:
    ```sh
    conda create --name transformer-ts python=3.8
    conda activate transformer-ts
    ```

4. Install the required packages:
    ```sh
    pip install numpy torch
    ```

## Usage

1. Run the training script:
    ```sh
    python TimeSeriesTransformer.py
    ```

    This script will:
    - Generate random input data
    - Train the Transformer model on this data
    - Save the trained model to a file

2. You can modify the `TimeSeriesTransformer.py` script to use your own time series data by replacing the random data generation section.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or improvements, please open an issue or submit a pull request.

## License

No License, if you like it give me a star 🌟
