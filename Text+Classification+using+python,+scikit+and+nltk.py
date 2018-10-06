import json
from sklearn.datasets import load_files

from sklearn.cross_validation import train_test_split

twenty_train = load_files('./news/news-bydate-train',description=None, categories=None, load_content=True, shuffle=True, encoding='utf8',decode_error='ignore')
# twenty_train = bunch(subset='train', shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(twenty_train.data, twenty_train.target, test_size=.2)

#test
# a = 'President Trump moved on Friday to leave an even deeper mark on Republican primary season, boosting a personal ally who is running for governor of Florida and extending political clemency to a former critic, Representative Martha Roby of Alabama, who is in a difficult race for re-election.Ms. Roby has faced criticism from the right since withdrawing her endorsement of Mr. Trump in the closing weeks of the 2016 presidential election, after the release of the “Access Hollywood” recording that showed Mr. Trump boasting about groping women. Alabama Republicans declined to re-nominate her in a primary election earlier this month, forcing her instead into a July 17 runoff vote.'

# dataset want to predict
print(twenty_train.target_names)  # prints all the categories

# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Building a pipeline
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train, y_train)



# Performance of NB Classifier
import numpy as np
from sklearn.linear_model import SGDClassifier

predicted = text_clf.predict(X_test)
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='modified_huber',penalty = 'elasticnet', alpha=1e-3, n_iter=5, random_state=42))])

# modified_huber
text_clf_svm = text_clf_svm.fit(X_train, y_train)
predicted_svm_a = text_clf_svm.predict(X_test)
print("ML work done.")
print("Accuracy :" + str(np.mean(predicted_svm_a == y_test)))

def getCategories(newPredict):

    predicted_svm = text_clf_svm.predict(newPredict)
    print("this test values")
    print(newPredict)
    print("this predicted values")
    print((text_clf_svm.predict_proba(newPredict)[0])[predicted_svm[0]])
    print(twenty_train.target_names[predicted_svm[0]])

    np.mean(predicted_svm == y_test)
    aList = []
    for i in range(len(predicted_svm)):
        pob =(text_clf_svm.predict_proba(newPredict)[i])[predicted_svm[i]]
        print(str(pob))
        print(twenty_train.target_names[predicted_svm[i]])
        aList.append({ "category" : str(twenty_train.target_names[predicted_svm[i]]),"pob": str(pob)})

    print(aList)
    return aList



#server
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

from io import BytesIO

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        try:
            body = json.loads(self.rfile.read(content_length))
            print(body['data'])
            contents = []
            for i in range(len(body['data'])):
                print(i)
                contents.append(body['data'][i]['content'])
            cat = getCategories(contents)
            print(cat)
            for i in range(len(cat)):
                print(i)
                body['data'][i]['category'] = cat[i]

            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes(str(body['data']), "utf-8"))

        except Exception as e:
            print(str(e))
            self.send_response(401)
            self.end_headers()
            self.wfile.write({'message': str(e)})

httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
print(httpd.server_name+ httpd.server_port)
