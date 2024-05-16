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

### Installation

Clone the repository and install the required Python packages:
 
    git clone https://github.com/CIGLR-ai-lab/GreatLakes-TempSensors.git
    cd GreatLakes-TempSensors
    pip install -r requirements.txt

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

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Thank you to the NOAA SOAR funding initiative for supporting this research work.

## References
For details on the methodologies and algorithms used in this project, refer to the following paper:

Andersson, T., Bruinsma, W., Markou, S., Requeima, J., et al. (2023). Environmental sensor placement with convolutional Gaussian neural processes. Environmental Data Science, 2, E32. [doi:10.1017/eds.2023.22](https://doi.org/10.1017/eds.2023.22)

Further resources included in the DeepSensor repository:  
[https://github.com/alan-turing-institute/deepsensor/resources.html](https://alan-turing-institute.github.io/deepsensor/resources.html)




   
