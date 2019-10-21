Getting started
===============

Python environment:

1. Create a new python environment by executing make environment. This creates a new python
virtual environment (or Conda environment if Conda is installed) with the same name as the
project.

2. Activate the new environment in the command line. If you have conda installed this
will be conda activate phone-sentiment-analysis.

3. Install need packages by executing make requirements. This will install all the
packages described in requirements.txt to the currently active environment (be sure that
the right one is active!).


Environmental Variables:

Create a file called .env in the root directory and add write variables
S3_BUCKET, EMR_JOB, SUBNET_ID there. E.g. S3_BUCKET="some-bucket".
These environmental variables are used to create the compute instance in AWS.


Configuring AWS Client:

Execute aws configure in the shell and give your aws account information if you
have not done so already (if you have aws client previously installed).


Creating the data with EMR process:

1. Execute make run_emr in the shell. This will start an EMS process in the AWS servers.
You can monitor this process trough the aws website. The process should take more than 10
hours but less than 24.

BEWARE! Using Amazon EMS is not free! Processing the data will cost around 10 euros (21.10.2019 pricing)

2. Download and preprocess the data by executing make download_data. The data will appear
in data/raw folder. With two separate files. With one having the actual data and one
containing the urls of the websites where the data comes from.


Processing the Data for Analysis:

Execute make data in the shell. This will do all the preprocessing and cleaning
outlined in src/make_dataset.py
