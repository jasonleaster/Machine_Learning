## Machine Learning

My Practice about Machine Learning. The aim that I create this project is to understand algorithm in machine learning better. But not only play with mathmatic equations.

> Tell me and I forget, teach me and I may remember, involve me and I learn. -- Benjamin Franklin

Algorithms which I have implemented:

* Percetron
* K Nearest Neighbor
* Decision Tree
* Naive Bayesian
* AdaBoost (Adaptive Boosting, Real Number Version.)
* Boosting Tree
* SVM (Supported Vecter Machine. Base on SMO algorithm)

``` python
python ./tester.py
```

You can test these implementation with the corresponding test file which I name it with `tester.py` in each directory.

You could calculate the accuracy of this algorithm like what I have done.
The picture below there is the accuracy of AdaBoost with test file `tester6.py`

![images](https://github.com/jasonleaster/Machine_Learning/blob/master/accuracy.png)


If you find any thing wrong with my program, you are welcome to touch me by e-mail: jasonleaster@163.com

Thank you :)

Yours, EOF

----

## Stype Of Implementation:

    All training samples are intialized as `self._Mat` which is organized like a matrix. `self._Mat[i][j]` means that the i-th feature value of the j-th sample in the training set. In the same way. If this model is supervised, there will be a data member `self._Label` or `self._Tag` in that class. `self.[i]` is the label of training sample `self._Mat[:, i]`

    Object oriented programming is used to this implementation. It's convenient to help people to test these implementation and easy to understand.

    If my style is not well, tell me your suggestion. I will be glad to accept it and refactor my implementation.
