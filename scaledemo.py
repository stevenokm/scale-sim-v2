# %% [markdown]
# <a href="https://colab.research.google.com/github/scalesim-project/scalesim-tutorial-materials/blob/main/scaledemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# #**SCALE Sim Demo**
#
# This notebook demonstrates step to setup SCALE-Sim from scratch and launch a run

# %% [markdown]
# #**Get the scalesim package**
#
# If your project does not require modifying scalesim then this should be sufficient. Although when you need it, the code is just one clone away on [github](https://github.com/scalesim-project/scale-sim-v2)!

# %%
#!pip install scalesim

# %% [markdown]
# #**Linking the inputs**
#
# SCALE-Sim uses two inputs to work
#
# 1.   First is a config file like `scale.cfg` below. The config file describes the architecture, dataflow and a few modes for the tool to run.
#
#
# 2.   Second is a CSV file, which specifies workloads as neural network layer parameters. In our example, this file is the `alexnet_part.csv`, which captures the description of the first convolution layer in Alexnet.
#
#

# %%
from scalesim.scale_sim import scalesim

content_path = "."
config = content_path + "/configs/scale.cfg"
topo = content_path + "/topologies/conv_nets/alexnet_full.csv"


# %% [markdown]
# #**Instantiating the simulator object**
#
# Here we create an object of the SCALE-Sim calls, and pass the paths to input files. We also set the tool to run in a verbose mode and signal it to generate trace files as well (save_disk_space=True, will suppress trace generation)

# %%
top = "./test_runs"
s = scalesim(save_disk_space=False, verbose=True, config=config, topology=topo)


# %% [markdown]
# #**Run Simulation**
#
# That's it, now we run the simulation with the following call.

# %%
s.run_scale(top_path=top)
