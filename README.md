# SKLearn Compatible Label Shift Detector

This is mostly a simple implementation of the paper:
**Detecting and Correcting for Label Shift with Black Box Predictors**.
This code was created as wrapper to a [scikit-learn](http://scikit-learn.org)
classifier. It's mostly used for teaching purposes.

## How to Use

Simply wrap [scikit-learn](http://scikit-learn.org) classifier.

```python
from sklearn import linear_model import LogisticRegressionCV
from label_shift.skwrapper import LabelShiftDetectorSKLearn

base = LogisticRegressionCV() # your base classifier, you can chage this.
classifier = LabelShiftDetectorSKLearn(base)
```

Train your model

```python
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
```

Now you can detect label-shifts using

```
p, pvals = classifier.label_shift_detector(X_test, return_bootstrap=True)
```

Here, `p` is the p-value for the 2-sided KS test. If it's below some threshold,
say `0.01`, you have detected a label shift.

`pvals` are bootstrapped p-values used to produce plots similar to the paper.

## Improving Classifiers

In case your base model supports the `class_weight` attribute, you can
improve predictions via the ERM (see paper). Note that the ERM is based on
a weighted loss-function. I'm not sure how every sklearn classifier explores
the `class_weight` attribute. Test before deploying.

```python
from sklearn import linear_model import LogisticRegressionCV
from label_shift.skwrapper import LabelShiftDetectorSKLearn

# Learn model
base = LogisticRegressionCV() # your base classifier, you can chage this.
classifier = LabelShiftDetectorSKLearn(base)
classifier.fit(X_train, y_train)


# Estimate weights
weights = classifier.wt_est_[:, 0].copy()
weights = weights / weights.sum()         # normalize to zero one
class_weights = {}
for k in range(len(weights)):
    class_weights[k] = weights[k]         # sklearn expects a dict

# This classifier should be better than the one with no weights.
new_classifier = linear_model.LogisticRegressionCV(class_weight=class_weights)
new_classifier.fit(X_train, y_train)
y_pred = new_classifier.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
```

## Using on tensorflow, mxnet, pytorch etc

Wrap your code in a class that supports the `fit/predict` methods from sklearn.
It's not ideal but doable for now.

Example below. Be warned that this is a hack with a lot of magic numbers!

```python

from mxnet import autograd
from mxnet import gluon
from mxnet import init
from mxnet import nd

from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from mxnet.gluon import utils


def load_array(features, labels, batch_size, is_train=True):

    transform = gdata.vision.transforms.Compose([
        gdata.vision.transforms.ToTensor(),
        gdata.vision.transforms.Cast('float32'),
        gdata.vision.transforms.Normalize(mean=0, std=1)])

    features = nd.array(features).reshape((len(features), 28, 28, 1))
    dataset = gluon.data.ArrayDataset(features, labels).transform_first(transform)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

class MXNetWrapper(object):

    def __init__(self):
        net = nn.Sequential()
        # network used in paper
        net.add(nn.Dense(512, activation='relu'),
                nn.Dense(512, activation='relu'),
                nn.Dense(10))
        cross_entropy = gloss.SoftmaxCrossEntropyLoss()
        lr = 0.01
        net.initialize(force_reinit=True, init=init.Xavier())

        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': lr})
        self.net = net
        self.trainer = trainer
        self.cross_entropy = cross_entropy

    def fit(self, X, y):
        net = self.net
        trainer = self.trainer
        cross_entropy = self.cross_entropy

        batch_size = 64
        train_iter = load_array(X, y, batch_size)

        num_iter = 20
        for i in range(num_iter):
            cumulative_loss = 0
            for data, y in train_iter:
                with autograd.record():
                    P = net(data)
                    loss = cross_entropy(P, y)
                loss.backward()
                trainer.step(data.shape[0])
                cumulative_loss += nd.sum(loss).asscalar()
            print('Iter {}. L {}'.format(i+1, cumulative_loss / len(data)))
        return self

    def predict(self, X):
        y = np.zeros(len(X))
        batch_size = 128
        test_iter = load_array(X, y, batch_size)

        net = self.net
        yres = []
        for data, _ in test_iter:
            yb = nd.softmax(net(data)).argmax(axis=1).asnumpy()
            yres.extend(yb)
        return np.array(yres, dtype='i')
```


## Links to other implementations

The notebooks execute several variations of the method. For some of them, you may need
the dataset from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/.

## Links to other implementations

1. Original https://github.com/zackchase/label_shift
1. Failing loudly https://github.com/steverab/failing-loudly
