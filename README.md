# MIMIC Triage Preprocessing


The project leverages the SciSpaCy library, developed by the Allen Institute for AI, to construct a comprehensive dataset aimed at training a medical diagnostic model. By integrating custom pipes and models, it focuses on optimizing the processing of scientific texts within the realms of medicine and biology.
## Installation

Before diving into the project, ensure you have the necessary packages installed. This project relies on SciSpaCy and a specific language model tailored for scientific literature. Follow the steps below to set up your environment:

1. Install SciSpaCy:
   ```bash
   pip install scispacy

2. Install the `en_core_sci_sm` model (version 0.5.0) for scientific English:
    ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
For more information about SciSpaCy, including documentation and additional models, visit the [SciSpaCy GitHub repository](https://github.com/allenai/scispacy)

## Data Source

This project employs data from the MIMIC-IV-ED dataset, specifically leveraging the triage and diagnosis modules. MIMIC-IV-ED is a comprehensive dataset that includes emergency department visits and contains rich medical information useful for various research and analysis tasks.

Access the dataset at: [MIMIC-IV-ED Dataset](https://physionet.org/content/mimic-iv-ed/2.2/)
