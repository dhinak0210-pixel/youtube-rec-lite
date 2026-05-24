# 🚀 RecoStream: Step-by-Step Installation Tutorial

Welcome to the beginner-friendly setup guide. Follow these step-by-step instructions to get the YouTube-scale recommendation system running on your local machine.

---

## 📋 Table of Contents
1. [Step 1: Verify Python Installation](#step-1-verify-python-installation)
2. [Step 2: Scaffold Directory Structure](#step-2-scaffold-directory-structure)
3. [Step 3: Create Project Placeholder Files](#step-3-create-project-placeholder-files)
4. [Step 4: Initialize Virtual Environment](#step-4-initialize-virtual-environment)
5. [Step 5: Install Package Dependencies](#step-5-install-package-dependencies)
6. [Step 6: Run the Unified Master Pipeline](#step-6-run-the-unified-master-pipeline)
7. [Step 7: Run Integration Tests](#step-7-run-integration-tests)
8. [Step 8: Run Performance Latency Benchmarks](#step-8-run-performance-latency-benchmarks)
9. [Step 9: Run the Interactive Demo App](#step-9-run-the-interactive-demo-app)

---

### Step 1: Verify Python Installation
First, verify that your machine has Python 3.10 or 3.11 installed.

1. **Open your terminal**:
   * **Mac**: Press `Cmd + Space`, type `terminal`, and press `Enter`.
   * **Windows**: Press `Win + R`, type `cmd`, and press `Enter`.
   * **Linux**: Press `Ctrl + Alt + T`.

2. **Check your python version**:
   Type the following command in your terminal and press `Enter`:
   ```bash
   python --version
   ```
   *You should see output similar to:* `Python 3.10.x` or `Python 3.11.x`.
   
   > [!IMPORTANT]
   > If you see an error (e.g. `command not found`), or a version older than `3.10`, navigate to [python.org/downloads](https://www.python.org/downloads/) to download and install Python 3.11.

---

### Step 2: Scaffold Directory Structure
If you are initializing a fresh project folder structure, create the core directories using terminal commands:

1. In your terminal, make sure you are in your project directory:
   ```bash
   cd "recommantation systeam"
   ```

2. Run these commands to create all the required folders:
   * **Linux / macOS**:
     ```bash
     mkdir -p config data models services streaming evaluation training demo tests benchmarks
     ```
   * **Windows (Command Prompt)**:
     ```cmd
     mkdir config data models services streaming evaluation training demo tests benchmarks
     ```

3. **Verify it worked**:
   Type this command and press `Enter`:
   ```bash
   # On Mac / Linux:
   ls
   
   # On Windows:
   dir
   ```
   *You should see all the folders listed in your active workspace!* ✅

---

### Step 3: Create Project Placeholder Files
Create the initial Python modules inside each folder to construct the application layout:

* **On Mac / Linux (Terminal)**:
  ```bash
  touch config/settings.py config/__init__.py data/schemas.py data/generator.py data/__init__.py models/collaborative_filtering.py models/matrix_factorization.py models/sequential_recommender.py models/graph_neural_network.py models/multi_objective_ranker.py models/cold_start.py models/model_registry.py models/__init__.py services/feature_store.py services/ab_testing.py services/__init__.py streaming/pipeline.py streaming/__init__.py evaluation/metrics.py evaluation/__init__.py training/pipeline.py training/__init__.py demo/app.py demo/__init__.py tests/test_all.py tests/__init__.py benchmarks/run_benchmarks.py main.py
  ```

* **On Windows (Command Prompt)**:
  ```cmd
  type nul > config\settings.py
  type nul > config\__init__.py
  type nul > data\schemas.py
  type nul > data\generator.py
  type nul > data\__init__.py
  type nul > models\collaborative_filtering.py
  type nul > models\matrix_factorization.py
  type nul > models\sequential_recommender.py
  type nul > models\graph_neural_network.py
  type nul > models\multi_objective_ranker.py
  type nul > models\cold_start.py
  type nul > models\model_registry.py
  type nul > models\__init__.py
  type nul > services\feature_store.py
  type nul > services\ab_testing.py
  type nul > services\__init__.py
  type nul > streaming\pipeline.py
  type nul > streaming\__init__.py
  type nul > evaluation\metrics.py
  type nul > evaluation\__init__.py
  type nul > training\pipeline.py
  type nul > training\__init__.py
  type nul > demo\app.py
  type nul > demo\__init__.py
  type nul > tests\test_all.py
  type nul > tests\__init__.py
  type nul > benchmarks\run_benchmarks.py
  type nul > main.py
  ```

* **Verify the files were created successfully**:
  Type this command and press `Enter`:
  * **Mac / Linux**:
    ```bash
    find . -name "*.py" | grep -v "/venv/" | head -30
    ```
  * **On Windows**:
    ```cmd
    dir /s /b *.py
    ```
  *You should see all your newly initialized Python files listed in your terminal!* ✅

---

### Step 4: Initialize Virtual Environment
A virtual environment keeps your project packages isolated from other libraries on your machine.

1. Initialize the virtual environment named `venv`:
   ```bash
   python -m venv venv
   ```

2. **Activate the environment**:
   * **Mac / Linux**:
     ```bash
     source venv/bin/activate
     ```
   * **Windows**:
     ```bash
     venv\Scripts\activate
     ```

3. **Verify activation**:
   * You should see `(venv)` appear at the start of your terminal line. This means it worked! ✅
   * *Example:* `(venv) your-name@computer:~/recommantation-systeam$`

---

### Step 5: Install Package Dependencies
With your virtual environment active, update your package manager and install all the required libraries:

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install dependencies (PyTorch, FastAPI, Gradio, fakeredis, pandas, etc.)
pip install -r requirements.txt
```

> [!NOTE]
> This command takes around 3-5 minutes to complete. You will see a lot of package download and build text scrolling down your terminal. Once it stops and returns you to the active command line prompt, your environment is fully ready! ✅

---

### Step 6: Run the Unified Master Pipeline
Spin up both the FastAPI backend server and the Gradio Web UI dashboard in one click:

```bash
python run.py
```

* **Gradio Web Interface**: Open `http://localhost:7860` in your web browser.
* **FastAPI Swagger API Docs**: Open `http://localhost:8000/docs` in your web browser.

---

### Step 7: Run Integration Tests
To prove that all retrieval models, ranking gates, and streaming metrics are performing correctly:

```bash
pytest
```
*You should see all 16 tests pass successfully.*

---

### Step 8: Run Performance Latency Benchmarks
To run inference speed trials and print latency reports:

```bash
python -m benchmarks.run_benchmarks
```
*You should see millisecond stats for all retrieval algorithms and the neural ranker!* ✅

---

### Step 9: Run the Interactive Demo App
Spin up the web app dashboard built using custom dark-themed CSS and interactive tab layouts:

```bash
python demo/app.py
```
*Open `http://localhost:7865` in your browser to interact with the finished system!* ✅
