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

1. Create json job step descriptions by executing make job_json. This will break down
the Common Crawl WET-paths to chunks used as processing steps.

2. Execute make create_bucket. This will create a S3 bucket to hold all necessary data
and scripts for the EMR jobs.

3-5. Execute make run_job1-3 in shell. Each of these will start a separate EMR job with
specific steps dictated by different json files. Although it is possible to put all the
three scripts running at once, it is probably not actually feasible as the the limit of
computing instances in AWS is by default 12 and each job takes up 5 instances.

BEWARE! Using Amazon EMS is not free! Processing the data will cost around 25 euros (21.10.2019 pricing)

6. Download and pre-process the data by executing make download_data. The data will appear
in data/raw folder. With two separate files. With one having the actual data and one
containing the urls of the websites where the data comes from.


Processing the Data for Analysis:

Execute make data in the shell. This will do all the preprocessing and cleaning
outlined in src/make_dataset.py
