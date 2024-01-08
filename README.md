The presented package aims to build upon the existing Python package RA2CE (Deltares/ra2ce, Resilience Assessment and Action perspective for Critical infrastructurE) used for the road network and extend its functionalities to be used for assessing the resilience of multi-modal transport networks. Initial steps are taken to include railway network with this regard.

The most important packages used are:
- OSM-FLEX: Required for downloading the PBF (Protocolbuffer Binary Format) files, extracting relevant railway elements, and clipping to the region of interest. (https://github.com/osm-flex/)

- Trail: A Python toolkit for Trade and tRAnsport Impact and fLow analysiS (https://github.com/ElcoK/trails).

- snkit: https://github.com/tomalrussell/snkit

- RA2CE (Resilience Assessment and Action perspective for Critical infrastructurE): Focuses on mapping the exposure, criticality, and vulnerability as well as the forthcoming prioritisation of locations to take actions based on cost-benefit assessment. ra2ce_multi_modal_network uses a branch of RA2CE called chore/185-ra2ce-multi. (https://github.com/Deltares/ra2ce)

# Installation
1. Clone the repository:

```
git clone https://github.com/sahand-asgarpour/ra2ce_multi_network.git
```

2. Create the ra2ce_multi_env:

```
cd <to the main repository RA2CE folder>
conda env create -f .config\environment.yml
conda activate ra2ce_multi_env
```
3. Manually download binaries from https://wiki.openstreetmap.org/wiki/Osmconvert#Windows . Then rename it to osmconvert.exe. Then place the .exe in the osm folder created by osm_flex during the installation.
4. For updating the Trail package, Clone and place the Trail's src folder in the ra2ce_multi|_network folder, and then import accordingly in .py files.

# Overview of the main features
The following features (using the mentioned packages, modified, or developed here) are available:

1. A rail network for a specific transport purpose (e.g., freight) can be created and simplified for the region of interest. The railway network simplification includes the following:
- Dropping hanging nodes.
- Merging tracks: segments with node degrees of 2 are merged while allowing for excluding bridges and tunnels to be merged (also other types of road links)

2. The package allows for introducing terminals or detecting possible terminals in the region under study. Moreover, the introduced or identified terminals are mapped on the railway network.
- Detecting possible terminals: Identifying hanging nodes with the track type of spur.
- Introducing terminals: Input by the user in GEOJSON format. It should include information about the origin or destination IDs and modes to which the terminals deliver services.

3. A multi-modal object (from the MultiModal class) can be created that contains information about the input graphs (NetworkX type) networks (road and rail). Moreover, this object creates a multi-modal network that connects networks at terminals serving these networks. For instance, if we have rail and road networks and a number of multi-modal terminals serving both networks, then the multi-modal graph consists of the rail and road networks connected at the multi-modal nodes. The weight (i.e., travel time or distance) of the connector edges linking separate graphs are defined statically and can be defined through the connector_weight parameter of the create_multi_modal_graph method of the MultiModal class. Such weights will be used later in finding the optimal routes.

4. After creating a multi-modal object (from the MultiModal class), it is possible to explore the optimal routes among origins and destinations through the introduced graphs (rail and road) as well as the multi-modal graph. The shortest path is calculated using transport time or distance (weights). The optimal route finding is connected to the RA2Ce package for the future integration of the ra2ce_multi_modal_network and RA2CE

In the [examples](https://github.com/sahand-asgarpour/ra2ce_multi_network/tree/master/ra2ce_multi_network/examples) folder, examples are provided for the mentioned features.

