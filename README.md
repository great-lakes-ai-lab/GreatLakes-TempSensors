# GreatLakes-TempSensors

## Project Overview
This repository is dedicated to the Great Lakes Summer Fellows Program project focused on optimizing the placement of temperature sensors across the Great Lakes using advanced machine learning techniques. By leveraging the DeepSensor framework, we aim to improve the spatial network design for environmental monitoring, providing valuable insights into the Great Lakes' surface temperature variability. This project will serve as a proof-of-concept, allowing for expansion into other key variables, such as nutrients.

## Background
The Great Lakes are a critical natural resource, providing drinking water, transportation routes, recreational opportunities, and supporting a diverse ecosystem. However, monitoring such a vast area is challenging due to logistical constraints and resource limitations. It is crucial to make the most efficient use of available observing platforms (e.g. buoys, research vessels). This project seeks to utilize convolutional Gaussian neural processes, as formulated in the DeepSensor tool, to propose a strategic placement of temperature sensors, thereby optimizing the observation network.

### Overarching Goal
Our overall objective is to develop a quantitative framework for strategic placement of the next generation of Great Lakes observing stations in order to best capture surface temperature variability. To phrase this goal as a research question, “where should the next generation of temperature measurement sensors be placed in order to most efficiently improve our quantitative understanding of Great Lakes surface temperature variability?”

### Project Activities
The summer fellow will use DeepSensor, an open source Python package for probabilistically modeling environmental data with neural processes, to characterize Great Lakes surface temperature and to make informed suggestions for future temperature sensor locations. Specifically, the student will:

- Use existing observational and model data to train DeepSensor on Great Lakes surface temperature
- Create a list of target observing sites that would most efficiently reduce uncertainties in our quantitative representation of Great Lakes surface temperature variability
- Prepare a brief report and final presentation to visualize, characterize, and document the results

## Getting Started
### Prerequisites
- Python 3.x
- Familiarity with machine learning concepts and data analysis
- Access to high-performance computing resources (provided for the project)

## DeepSensor Environment Setup [Non-GitHub Version]

Below are some instructions for setting up your environment for the first time and using it thereafter in the interactive Jupyter notebook environment on the Great Lakes HPC platform. This should prevent you from having to `pip install deepsensor` every time that you want to run a notebook. (This version does not use GitHub).

### Environment Setup (First Time)

1. **Activate Your Virtual Environment:**
   
   If you don't have a virtual environment already, create one using `python -m venv` or `conda create`. For example, with `venv`:
   
   ```bash
   module load python3.10-anaconda/2023.03  # Load the Anaconda module on U-M HPC
   python -m venv ~/deepsensor_env  # Create a virtual environment named "deepsensor_env" in your home directory
   source ~/deepsensor_env/bin/activate  # Activate the virtual environment
   ```

2. **Install Required Packages:**
   
   Install the required packages listed in the `requirements.txt` file:

   ```bash
   pip install deepsensor==0.3.6 tensorflow==2.16.1 tensorflow-io-gcs-filesystem==0.37.0 tensorflow-probability==0.24.0
   ```

   This will install all the necessary dependencies for the DeepSensor project. Optionally, if you want to use pytorch, install that instead:

   ```bash
   pip install deepsensor==0.3.6 torch==2.3.0
   ```

3. **Create a Setup File:**

   Create a setup file that will activate your virtual environment. Save this file somewhere in your home directory, for example, `~/setup_deepsensor_env.sh`:

   ```bash
   echo 'module load python3.10-anaconda/2023.03' > ~/setup_deepsensor_env.sh
   echo 'source ~/deepsensor_env/bin/activate' >> ~/setup_deepsensor_env.sh
   chmod +x ~/setup_deepsensor_env.sh
   ```

### Using the Environment (Thereafter)

After setting up the environment for the first time, follow these steps to use it thereafter:

1. **[If you want to work in the terminal] Activate Your Virtual Environment:**
   
   If you want to work in the terminal, activate your virtual environment if it's not already activated:

   ```bash
   source ~/deepsensor_env/bin/activate  # Activate the virtual environment
   ```

   Alternatively, you can run the setup script that you created above:
   ```bash
   ~/setup_deepsensor_env.sh
   ```

   If, instead, you want to use a notebook, skip this step and proceed to Step 2. 

1. **[If you wan to use Jupyter Notebook] Start an Interactive Jupyter Notebook Session:**

    If you want to use a Jupyter Notebook interfact, use these steps instead:

   - Log in to the Great Lakes HPC platform.
   - Navigate to the "Interactive Apps" section.
   - Select "Jupyter Notebook" and configure the job submission form as needed.
   - Specify the Anaconda Python module (`python3.10-anaconda/2023.03`), Slurm account, partition, number of hours, cores, memory, and source the setup file (`~/setup_deepsensor_env.sh`).
   - Submit the job and wait for it to start.
   - Once the job starts, open the provided link to access the Jupyter Notebook interface.
   - Your virtual environment will be activated automatically, providing access to the DeepSensor project and its dependencies.

4. **Run Your DeepSensor Notebooks:**
   
   Within the Jupyter Notebook interface, navigate to your DeepSensor project directory and open the desired notebook.
   
5. **Deactivate Your Virtual Environment (Optional):**
   
   When you're done working, you can deactivate your virtual environment:
   
   ```bash
   deactivate
   ```

   This will return you to your base environment.

## DeepSensor Environment Setup [GitHub Version]

Below are some instructions for setting up your environment for the first time and using it thereafter in the interactive Jupyter notebook environment on the Great Lakes HPC platform. This should prevent you from having to `pip install deepsensor` every time that you want to run a notebook.

### Environment Setup (First Time)

Follow these steps to set up your environment for the first time:

1. **Clone the Repository:**
   
   ```bash
   git clone https://github.com/your-username/deepsensor.git
   cd deepsensor
   ```

2. **Activate Your Virtual Environment:**
   
   If you don't have a virtual environment already, create one using `python -m venv` or `conda create`. For example, with `venv`:
   
   ```bash
   module load python3.10-anaconda/2023.03  # Load the Anaconda module
   python -m venv ~/deepsensor_env  # Create a virtual environment named "deepsensor_env" in your home directory
   source ~/deepsensor_env/bin/activate  # Activate the virtual environment
   ```

3. **Install Required Packages:**
   
   Install the required packages listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all the necessary dependencies for the DeepSensor project.

### Using the Environment (Thereafter)

After setting up the environment for the first time, follow these steps to use it thereafter:

1. **[If you want to work in the terminal] Activate Your Virtual Environment:**
   
   If you want to work in the terminal, activate your virtual environment if it's not already activated:

   ```bash
   source ~/deepsensor_env/bin/activate  # Activate the virtual environment
   ```

   If, instead, you want to use a notebook, skip this step and proceed to Step 2. 

2. **[If you wan to use Jupyter Notebook] Start an Interactive Jupyter Notebook Session:**

    If you want to use a Jupyter Notebook interfact, use these steps instead:

   - Log in to the Great Lakes HPC platform.
   - Navigate to the "Interactive Apps" section.
   - Select "Jupyter Notebook" and configure the job submission form as needed.
   - Specify the Anaconda Python module (`python3.10-anaconda/2023.03`), Slurm account, partition, number of hours, cores, memory, and source the setup file (`~/setup_deepsensor_env.sh`).
   - Submit the job and wait for it to start.
   - Once the job starts, open the provided link to access the Jupyter Notebook interface.
   - Your virtual environment will be activated automatically, providing access to the DeepSensor project and its dependencies.

4. **Run Your DeepSensor Notebooks:**
   
   Within the Jupyter Notebook interface, navigate to your DeepSensor project directory and open the desired notebook.
   
5. **Deactivate Your Virtual Environment (Optional):**
   
   When you're done working, you can deactivate your virtual environment:
   
   ```bash
   deactivate
   ```

   This will return you to your base environment.

## Usage
Instructions on how to train the DeepSensor model, analyze the data, and propose sensor locations will be provided in the `docs` directory or as separate markdown files within this repository.

## Contributing
Contributions to this project are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute effectively.

## Project Milestones
We've set several [Milestones](https://github.com/CIGLR-ai-lab/GreatLakes-TempSensors/milestones) to organize the progress of the project, including a review of the literature and other resources, data preparation and visualization, model training and runing, analysis and site recommendation, reporting and documentation, and final presentation.

The detailed timeline is available under the repository's [Projects](https://github.com/CIGLR-ai-lab/GreatLakes-TempSensors/projects) tab.

## Mentors & Contributors
- Dani Jones (CIGLR)
- Russ Miller (CIGLR)
- Shelby Brunner (Great Lakes Observing System)
- David Cannon (CIGLR)
- Hazem Abdelhady (CIGLR)

For a full list of contributors, please see the [contributors](https://github.com/CIGLR-ai-lab/GreatLakes-TempSensors/graphs/contributors) page.

## Connections with the DeepSensor development community
The Alan Turing Institute hosts the development of this software, in part by maintaining a Slack channel. For info on how to join the Slack channel, [visit the DeepSensor Repository and check out the README](https://github.com/alan-turing-institute/deepsensor).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Thank you to the NOAA SOAR funding initiative for supporting this research work.

## References
For details on the methodologies and algorithms used in this project, refer to the following paper:

Andersson, T., Bruinsma, W., Markou, S., Requeima, J., et al. (2023). Environmental sensor placement with convolutional Gaussian neural processes. Environmental Data Science, 2, E32. [doi:10.1017/eds.2023.22](https://doi.org/10.1017/eds.2023.22)

Further resources included in the DeepSensor repository:  
[https://github.com/alan-turing-institute/deepsensor/resources.html](https://alan-turing-institute.github.io/deepsensor/resources.html)




   
