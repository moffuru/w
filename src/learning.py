import glob
import pickle

import cv2
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

import wa_utils


def flatten_image(img):
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s) / 255
    return img_wide[0]


colors = ['r', 'g', 'b', 'y', 'p', 'o']

X_for_color = []
Y_for_color = []
X_for_empty = []
Y_for_empty = []

PATH = f'../resources/teacher_data/'

for i, c in enumerate(colors):
    d = glob.glob(f'{PATH}/{c}/*')
    for file in d:
        # Y.append(i)
        Y_for_color.append(c)
        Y_for_empty.append('C')
        t = wa_utils.transform_for_recognition(cv2.imread(file))
        X_for_color.append(t)
        X_for_empty.append(t)
    # print(d)

for file in glob.glob(f'{PATH}/e/*'):
    Y_for_empty.append('E')
    X_for_empty.append(wa_utils.transform_for_recognition(cv2.imread(file)))

X_for_color_train, X_for_color_test, Y_for_color_train, Y_for_color_test = train_test_split(X_for_color, Y_for_color, train_size=0.6, test_size=0.4, random_state=0, stratify=Y_for_color)
X_for_empty_train, X_for_empty_test, Y_for_empty_train, Y_for_empty_test = train_test_split(X_for_empty, Y_for_empty, train_size=0.6, test_size=0.4, random_state=0, stratify=Y_for_empty)


def test(model, tag, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    # print(model.predict(X_test)[10:20], Y_test[10:20])

    print(f"{tag} train score:", model.score(x_train, y_train))
    print(f"{tag} test score:", model.score(x_test, y_test))

    with open(f'../local_resources/model/{tag}.m', 'wb') as f:
        pickle.dump(model, f)


# test(LinearRegression(), 'linear')
test(KNeighborsClassifier(n_neighbors=3), '3n', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(KNeighborsClassifier(n_neighbors=2), '2n', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(DecisionTreeClassifier(max_depth=3), '3t', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(DecisionTreeClassifier(max_depth=4), '4t', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(DecisionTreeClassifier(max_depth=5), '5t', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(DecisionTreeClassifier(max_depth=10), '10t', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(DecisionTreeClassifier(max_depth=100), '100t', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(DecisionTreeClassifier(max_depth=400), '400t', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(LinearSVC(max_iter=60000), 'lsvm', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(SVC(max_iter=60000), 'svm', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000), 'logic', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
# test(LogisticRegressionCV(solver='lbfgs', multi_class='auto', max_iter=10000, cv=5), 'logicCV', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(RidgeClassifier(max_iter=10000), 'ridge', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)
test(RidgeClassifierCV(cv=5), 'ridgeCV', X_for_color_train, Y_for_color_train, X_for_color_test, Y_for_color_test)


test(KNeighborsClassifier(n_neighbors=3), '3n_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(KNeighborsClassifier(n_neighbors=2), '2n_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(DecisionTreeClassifier(max_depth=3), '3t_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(DecisionTreeClassifier(max_depth=4), '4t_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(DecisionTreeClassifier(max_depth=5), '5t_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(DecisionTreeClassifier(max_depth=10), '10t_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(DecisionTreeClassifier(max_depth=100), '100t_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(DecisionTreeClassifier(max_depth=400), '400t_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(LinearSVC(max_iter=60000), 'lsvm_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(SVC(max_iter=60000), 'svm_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000), 'logic_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
# test(LogisticRegressionCV(solver='lbfgs', multi_class='auto', max_iter=10000, cv=5), 'logicCV_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(RidgeClassifier(max_iter=10000), 'ridge_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
test(RidgeClassifierCV(cv=5), 'ridgeCV_empty', X_for_empty_train, Y_for_empty_train, X_for_empty_test, Y_for_empty_test)
