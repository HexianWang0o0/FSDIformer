# 🌬️ A Novel Deep Learning Wind Power Forecasting Model

This project proposes a novel deep learning architecture for accurate wind power forecasting. It leverages historical data, advanced neural network design, and scalable training scripts to deliver strong performance across various wind farm scenarios.

## 🚀 Get Started

Follow the steps below to run the project:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Download the input feature dataset from:

🔗 [Renewable energy generation input feature variables (GitHub)](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis)

Place the downloaded data into an appropriate directory, for example:

```
./datasets/
```

### 3. Train the model

All training scripts are located in the `./scripts` directory.

For **distributed multi-machine training**, please:

- Specify the **master node IP address**
- Set the **model save path**

Then execute the training script:

```bash
sh ./scripts/windfarm1.sh
```

You can adjust the hyperparameters based on your needs by editing the script or passing arguments directly.

---

## 📁 Project Structure

```
.
├── scripts/                # Training scripts
├── models/                 # Model definitions
├── layers/                 # Custom neural network layers
├── data_provider/          # Data loading and processing
├── datasets/               # Dataset (place your files here)
├── exp/                    # Experiment configuration and logging
├── utils/                  # Utility functions and helpers
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🧠 Key Features

- 📊 Designed for wind power prediction using deep learning  
- 🌍 Supports both onshore and offshore wind farms  
- 🔁 Suitable for ultra-short-term and short-term wind power forecasting  
- ⚡ Scalable to multi-GPU and multi-machine setups  
- 📦 Easy-to-use scripts for reproducible experiments  

---

## 📫 Contact

For any questions, suggestions, or collaborations, feel free to reach out:

📧 **hexianwang429@gmail.com**

---

**Enjoy forecasting the wind power! 💨**
