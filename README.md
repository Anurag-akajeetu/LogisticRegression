# LogisticRegression

Abstract—Two-class classification is a well-studied problem in the machine learning literature. In this work, we pick the basic binary
classifier flavor of the logistic regression machine and study the properties of its momentum. We implement the negative log likelihood
loss and hinge-loss methods into logistic regression to form the primary baselines. Subsequently, we learn the gradient descent learning
procedure using Polyak’s classical momentum [1] and Nesterov’s accelerated gradient [2] [3]. We compare all these variants of logistic
regression among themselves to understand the properties of the momentum and how they contribute to convergence by an iterative
observation of the parameters learnt such as weights and losses. We also study how Polyak and Nesterov compare to using L2-
regularization w.r.t. the speed and stability of learning. We demonstrate the effectiveness of each of the optimization techniques through
a case-by-case experimental analysis for the various samples drawn from the real-world datasets for Entity Resolution and the synthetic
datasets that we craft manually.

Index Terms—logistic regression, Polyak, Nesterov, momentum, regularization, classification, resolution
