RA2CE
=====

This is built based on the single network RA2CE repository -the Resilience Assessment and Adaptation for Critical infrastructurE Toolkit Python Package eveloped by Deltares. It aims to expand to multi-network problem configuration and analysis.

**Contact** Margreet van Marle (Margreet.vanMarle@Deltares.nl)

Installation
---------------------------
RA2CE can be operated via the command-line interface with two commands. Before RA2CE can be used, the correct Python environment needs to be installed (see *environment.yml*). Anaconda is a well-known environment manager for Python and can be used to install the correct environment and run RA2CE via its command-line interface. It is recommended to install Anaconda, instead of `miniconda`, so that you have all required packages already available during the following steps.

CLI only
+++++++++++++++++++++++++++
If only interested in using the tool via command-line interface follow these steps:
::
  pip install git+https://github.com/sahand-asgarpour/ra2ce_multi_network.git
::

Development mode
+++++++++++++++++++++++++++
When running a development environment with Anaconda, the user may follow these steps in command line:
::
  cd <to the main repository ra2ce_multi_network folder>
  conda env create -f .config\environment.yml
  conda activate ra2ce_env
  poetry install
::
