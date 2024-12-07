# LC_SEG

Music structure segmentation via novelty curves.

Complementary repo for the paper "Structural Analysis of Live Coding Performances Through Novelty-based MIR Methodologies", ICLC 2025. 


## Installation

Download the code, navigate to the folder, and install the required libraries:

```bash
pip install requirements.txt
```

## Usage

Move the desired file/s (.wav or .mp3) to be analyzed in the audio folder. 

Then run:

```bash
python segmentation.py
```

In the analysis folder, the script will generate separate folders for every audio analyzed. Within such folders, : 

1) spectrograms;
2) feature plots for each feature;
3) self similarity matrices for each feature;
4) a cumulative plot of single novelty curves for each features and relative peaks + global novelty curve and peaks;
5) segment similarity matrices for each feature + global;
6) an audio track with the original audio (left channel) + audio clicks in correspondence of the retrieved boundaries (right channel) 
7) a .json file with all the retrieved boundaries. 

You may want to adjust the parameters in the config file according to your preferences. 



## References
Foote, J. (2000). *Automatic audio segmentation using a measure of audio novelty*. In Proceedings of the IEEE International Conference on Multimedia and Expo (ICME), New York, USA, pp. 452–455.

Müller, M. (2015). *Fundamentals of music processing: Audio, analysis, algorithms, applications* (Vol. 5, p. 62). Cham: Springer.


## License

[MIT](https://choosealicense.com/licenses/mit/)
