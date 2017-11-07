# Research Project: Exploiting Structure For Classification Of Handwritten Japanese Characters

This was a research project for CSC412 (Probabilistic Learning and Reasoning) course at University of Toronto, taken Winter 2017. 

Due to licensing, the dataset is not published. It can be acquired at http://etlcdb.db.aist.go.jp/. The path to the dataset needs to be inserted in /code/load_data.py and /code/load_data_rad.py

The models are all contained in different files because they were all ran on different AWS EC2 machines simultaneously and this was the easiest solution, even though they shared a lot of code. Similarly with load_data and load_data_rad, they have matching code, but were tweaked separately and ended up never put together into a single file.
