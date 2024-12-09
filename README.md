# NeuroPixels preprocessing toolkit

A set of codes and tools based on spikeinterface and kilorsort for preprocessing, spike sorting, and lfp extraction from binary AP data.


#### Getting Started

To setup a fresh conda environment and all the requirements run:

```sh
conda create -n pixelpipe --file requirements.txt
```

Here is a simple code sample for:

##### Simple sorting
```python
python pipeline.py --f 'NP_FOLDER_PATH' --sort
```

##### Preprocess and sorting with Kilosort 4 (with Kilosort 4 builtin motion correction):

Remove bad channels ->
High pass filter ->
Phase shift ->
CAR -> Sort

```python
python pipeline.py --f 'NP_FOLDER_PATH' --preprocess --sort
```

##### Preprocess, motion correction and sorting with Kilosort 4:

Remove bad channels ->
High pass filter ->
Phase shift ->
CAR -> 
Motion correction ->
Sort

```python
python pipeline.py --f 'NP_FOLDER_PATH' --preprocess --motion 'medicine' --sort
```

##### Preprocess, multiple motion correction methods, and sorting with Kilosort 4:

Remove bad channels ->
High pass filter ->
Phase shift ->
CAR -> 
Motion correction ->
Sort

```python
python pipeline.py --f 'NP_FOLDER_PATH' --preprocess --motion 'medicine,decenter' --sort
```



