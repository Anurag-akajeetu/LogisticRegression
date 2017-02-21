

There are three code files, lr.py DataSetCreation.py and polynest.py.

1) DataSetCreation file is to generate synthetic data. This file will create two kinds of data, linearly separable and non-linear separable data which would be placed under LinearSeparable and NonLinearSeparable folders, created as sub-directory.

To run the command just execute the DataSetCreation file. 
	
	Command to run is: python DataCreation.py
 
2) lr.py file is the file through which training and testing is done. 
Below is the dicription to execute the file.

python lr.py -mode ??? -folds True -cmp &&& -dataDir ####


1) There are two mode: loglh or hinge (Please replace ??? with any one of it before execution).

2) There are three cmp: l2reg, polyak, nesterov and none (Please replace &&& with any one of it before execution).

3) There are two dataDir: LinearSeparable and NonLinearSeparable (Please replace ### with any one of it before execution).


Loglh				: Log Likelihood function
Hinge				: Hinge Loss
L2reg				: L2 Regularization
Polyak			: Polyak Function
Nesterov			: Nesterov function
NonLinearSeparable	: NonlinearSeparable data directory
LinearSeparable		: linearSeparable data directory

