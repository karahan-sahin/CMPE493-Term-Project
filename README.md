# <span style="color:red">CMPE 493: Introduction to Information Retrieval Fall 2020</span>
# <span style="color:orange">README for Term Project</span>

## <span style="color:green">Introduction:</span>
This project has been given to us to understand and apply all possible information retrieval algorithms to create a CoViD-19 related search engine based on the TREC CoViD-19 dataset.

- Dataset Link: https://ir.nist.gov/covidSubmit/

## <span style="color:green">How To Run:</span>
To run the program, CLI is needed and many external library have been utilized.

Version & Platform: <span style="color:yellow">3.8.6 (64-bit)</span>

### - Put the unzipped dataset folder and metadata.csv separately in here.

### - After then, run the script to initialize and download necessary libraries:

```bash
./entry.sh
```

This should set your environment to be compatible with our application.

### - After that, type the below to run the application:

```bash
python3 evaluation.py
```

### - Randomization Test Tool: To run it, type the below:
### All results file should be in the tra/ folder within this current directory
### Example results file is in the tra/ folder.
### - P.S. : <prefix_of_the_result_file> ==> e.g. tf_idf_results.txt -> tf_idf

```bash
python3 randomization.py <prefix_of_the_first_result_file> <prefix_of_the_second_result_file>
```
