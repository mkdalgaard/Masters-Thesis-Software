# Masters-Thesis-Software
Publicly available Python scripts that has been used in the Master's thesis by Martin Kamp Dalgaard.

The repository contains the following:
<ul>
  <li>A library (library.py) with functions and classes that are used in the tests. The classes are e.g. used to compute the errors of the estimators and the CCDI of the autoregressive data and actual EEG data.</li>
  <li>A test script (test_script.py) with the tests that are performed and described in Chapters 3 and 4. The results from the tests are saved as pickle files in the folder "pickle".</li>
  <li>A figure script (figure_script.py) that produces the figures used in the thesis. The results from the test script are loaded through the pickle files in the folder "pickle".</li>
  <li>A script that performs data management on the data (data_management.py). Note however that the data are not available in this folder but it is available for download <a href=https://zenodo.org/record/2348892></a here> (see also the documentation <a href=https://hal.archives-ouvertes.fr/hal-02086581></a here>).</li>
</ul>
