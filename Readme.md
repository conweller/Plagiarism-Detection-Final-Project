# Plagiarism Detection

Files:

-   `plagiarism-dataset/`: directory containing the dataset I used (note
    this needs to be downloaded from
    <https://dx.doi.org/10.21227/71fw-ss32>)
-   `dataset.py`: module to parse dataset to generate representations of
    ground truth and gather source code files
-   `fingerprint.py`: module with functions to generate fingerprints
    from source code
-   `fingerprint_eval.py`: module to evaluate fingerprint performance
-   `feature_extraction.py`: module to perform feature extraction to
    generate feature representations of student submissions
-   `ml-eval.py`: module to evaluate different machine learning
    classification schemes' performances
-   `ml-output.txt`: results for machine learning classifiers
-   `fingerprinting-output.txt`: results for fingerprinting classifier
