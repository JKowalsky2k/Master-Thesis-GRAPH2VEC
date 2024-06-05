from google.colab import drive
drive.mount('/gdrive')

!cp -r "/gdrive/MyDrive/Colab Notebooks/Master/datasets/prepared/imdb.csv" "/content"
!cp -r "/gdrive/MyDrive/Colab Notebooks/Master/datasets/prepared/smses.csv" "/content"
!cp -r "/gdrive/MyDrive/Colab Notebooks/Master/datasets/prepared/speech.csv" "/content"

import glob

!rm -rf /content/sample_data/

# datasets_paths = glob.glob("/content/*")

!pip install torch torch-geometric networkx nltk --quiet

dataset = "speech.csv" # @param ["imdb.csv", "smses.csv", "speech.csv"]
datasets_paths = [f"/content/{dataset}"]

model_name = "ALL" # @param ["TFIDF", "WORD2VEC", "DOC2VEC", "GRAPH2VEC", "ALL"]
user_folds_number = 5 # @param [2, 3, 4, 5, 10] {type:"raw"}

smote = True # @param [True, False] {type: "raw"}

def save_to_csv(scores, folder, filename="default_filename"):
  partial_dataframe = prepare_partial_data(scores, display=False)
  summary_dataframe = prepare_summary_data(scores, display=False)
  tosave_dataframe = pandas.concat([partial_dataframe, summary_dataframe])
  print(f"Saved at: /gdrive/MyDrive/Colab Notebooks/Master/ALL/results/{folder}/{filename}.csv")
  tosave_dataframe.to_csv(f"/gdrive/MyDrive/Colab Notebooks/Master/ALL/results/{folder}/{filename}.csv", sep=',', index=False, encoding='utf-8')

def prepare_partial_data(scores, display=True):
  partial_data = []
  for idx in range(len(list(scores.values())[0][:user_folds_number])):
    tmp = list()
    for label in scores.keys():
      tmp.append(scores[label][idx])
    partial_data.append(tmp)

  partial_dataframe = pandas.DataFrame(partial_data, columns=[label for label in scores.keys()])
  if display:
    print(partial_dataframe)
  return partial_dataframe

def prepare_summary_data(scores, display=True):
  summary_data = [[]]
  for _, data in scores.items():
    summary_data[0].append(f"{numpy.mean(data):.3f} +- {numpy.std(data):.3f}")
  summary_dataframe = pandas.DataFrame(summary_data, columns=[label for label in scores.keys()])
  if display:
    print(summary_dataframe)
  return summary_dataframe

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pprint
import copy

import numpy

classifiers = [
    ("MLPClassifier", MLPClassifier()),
    ("Nearest_Neighbors", KNeighborsClassifier(3)),
    ("Linear_SVM", SVC(kernel="linear", C=0.025)),
    # ("RBF_SVM", SVC(gamma=2, C=1, random_state=42)),
    ("Decision_Tree", DecisionTreeClassifier(max_depth=5)),
    ("Random_Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("AdaBoost", AdaBoostClassifier()),
    ("Naive_Bayes", GaussianNB()),
    ("QDA", QuadraticDiscriminantAnalysis())
]

scores = {
            "fit_time_model": numpy.array([]),
            "fit_time": numpy.array([]),
            "score_time": numpy.array([]),
            "test_accuracy": numpy.array([]),
            "test_precision": numpy.array([]),
            "test_recall": numpy.array([]),
            "test_f1": numpy.array([]),
            "test_roc_auc": numpy.array([])
        }

models_names = ("TFIDF", "WORD2VEC", "DOC2VEC", "GRAPH2VEC")

results = {}
clasifier_metrics = {}
for model_name_t in models_names:
  for classifier_name, _ in classifiers:
    clasifier_metrics[classifier_name] = copy.deepcopy(scores)
  results[model_name_t] = copy.deepcopy(clasifier_metrics)

import pandas
pandas.set_option('display.width', 180)
import sklearn.feature_extraction.text
import time

def tfidf_dataframe_converter(train_dataframe, test_dataframe):
  start_time = time.time()
  vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_features=500, stop_words="english")
  end_time = time.time()
  for classifier_name, _ in classifiers:
    results["TFIDF"][classifier_name]["fit_time_model"] = numpy.append(results["TFIDF"][classifier_name]["fit_time_model"], end_time-start_time)

  tfidf_matrix = vectorizer.fit_transform(train_dataframe["Content"])
  tfidf_matrix = tfidf_matrix.toarray()
  train_vectors_dataframe = pandas.DataFrame(tfidf_matrix, columns=vectorizer.get_feature_names_out())
  train_vectors_with_class_dataframe = train_vectors_dataframe.assign(Class=train_dataframe["Class"].tolist())

  tfidf_matrix = vectorizer.transform(test_dataframe["Content"])
  tfidf_matrix = tfidf_matrix.toarray()
  test_vectors_dataframe = pandas.DataFrame(tfidf_matrix, columns=vectorizer.get_feature_names_out())
  test_vectors_with_class_dataframe = test_vectors_dataframe.assign(Class=test_dataframe["Class"].tolist())

  return train_vectors_with_class_dataframe, test_vectors_with_class_dataframe

import pandas
pandas.set_option('display.width', 180)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return filtered_words

def doc2vec_dataframe_converter(train_dataframe, test_dataframe):
  tagged_train_documents = [TaggedDocument(words=preprocess_text(row["Content"]), tags=[row["Class"]]) for _, row in train_dataframe.iterrows()]
  tagged_test_documents = [TaggedDocument(words=preprocess_text(row["Content"]), tags=[row["Class"]]) for _, row in test_dataframe.iterrows()]

  start_time = time.time()
  doc2vec_model = Doc2Vec(vector_size=100, window=10, min_count=1, workers=4, epochs=20)
  doc2vec_model.build_vocab(tagged_train_documents)
  doc2vec_model.train(tagged_train_documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
  end_time = time.time()
  for classifier_name, _ in classifiers:
    results["DOC2VEC"][classifier_name]["fit_time_model"] = numpy.append(results["DOC2VEC"][classifier_name]["fit_time_model"], end_time-start_time)

  columns_names = [f'V{column_idx}' for column_idx in range(doc2vec_model.vector_size)]

  train_vectors_dataframe, train_class_dataframe = zip(*[(doc2vec_model.infer_vector(document.words), document.tags[0]) for document in tagged_train_documents])
  train_vectors_dataframe = pandas.DataFrame(train_vectors_dataframe, columns=columns_names)
  train_vectors_with_class_dataframe = train_vectors_dataframe.assign(Class=train_dataframe["Class"].tolist())

  test_vectors_dataframe, test_class_dataframe = zip(*[(doc2vec_model.infer_vector(document.words), document.tags[0]) for document in tagged_test_documents])
  test_vectors_dataframe = pandas.DataFrame(test_vectors_dataframe, columns=columns_names)
  test_vectors_with_class_dataframe = test_vectors_dataframe.assign(Class=test_dataframe["Class"].tolist())

  # print(train_vectors_with_class_dataframe.head(n=1))
  # print(test_vectors_with_class_dataframe.head(n=1))

  return train_vectors_with_class_dataframe, test_vectors_with_class_dataframe

from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas
pandas.set_option('display.width', 180)
import numpy

def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return filtered_words

def create_document_vector(word2vec_model, document_words):
    document_words = [word for word in document_words if word in word2vec_model.wv.key_to_index]
    if len(document_words) == 0:
        return numpy.zeros(word2vec_model.vector_size)
    return numpy.mean(word2vec_model.wv[document_words], axis=0)

def word2vec_dataframe_converter(train_dataframe, test_dataframe):
  train_documents_words = [preprocess_text(document_content) for document_content in train_dataframe["Content"]]
  test_documents_words = [preprocess_text(document_content) for document_content in test_dataframe["Content"]]

  start_time = time.time()
  word2vec_model = Word2Vec(sentences=train_documents_words, vector_size=100, window=5, min_count=1, workers=4)
  end_time = time.time()
  for classifier_name, _ in classifiers:
    results["WORD2VEC"][classifier_name]["fit_time_model"] = numpy.append(results["WORD2VEC"][classifier_name]["fit_time_model"], end_time-start_time)

  columns_names = [f'V{column_idx}' for column_idx in range(word2vec_model.vector_size)]

  train_vectors_dataframe, train_class_dataframe = numpy.array([create_document_vector(word2vec_model, document) for document in train_documents_words]), train_dataframe["Class"]
  train_vectors_dataframe = pandas.DataFrame(train_vectors_dataframe, columns=columns_names)
  train_vectors_with_class_dataframe = train_vectors_dataframe.assign(Class=train_dataframe["Class"].tolist())

  test_vectors_dataframe, test_class_dataframe = numpy.array([create_document_vector(word2vec_model, document) for document in test_documents_words]), test_dataframe["Class"]
  test_vectors_dataframe = pandas.DataFrame(test_vectors_dataframe, columns=columns_names)
  test_vectors_with_class_dataframe = test_vectors_dataframe.assign(Class=test_dataframe["Class"].tolist())

  # print(train_vectors_with_class_dataframe.head(n=1))
  # print(test_vectors_with_class_dataframe.head(n=1))

  return train_vectors_with_class_dataframe, test_vectors_with_class_dataframe

import networkx as nx
import torch
import torch_geometric.utils
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE

import pandas
import numpy
import copy

import nltk.tokenize
import nltk.corpus
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
from gensim.models import Word2Vec

def preprocess_text(text):
    words = nltk.tokenize.word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return filtered_words

def create_document_vector_for_graph(tensor_graph_autoencoder_vector):
  numpy_graph_autoencoder_vector = tensor_graph_autoencoder_vector.detach().numpy()
  return numpy.mean(numpy_graph_autoencoder_vector, axis=0)

def build_graph_from_tokens(tokens, model_word2vec, debug=False, draw=False):
    G = nx.Graph()
    for token in tokens:
        if not G.has_node(token):
          try:
            vector = model_word2vec.wv[token]  # Próba uzyskania wektora dla słowa
          except KeyError:
            vector = numpy.zeros(model_word2vec.vector_size, dtype=numpy.float32)
          G.add_node(token, x=vector)
    for i, token in enumerate(tokens):
        for j in range(max(0, i - 2), min(i + 3, len(tokens))):
            if tokens[i] != tokens[j]:
                G.add_edge(tokens[i], tokens[j])
    return G

def convert_dataframe_to_graphs(dataframe, model_word2vec):
  converted_pyg_graph_dataset = list()
  dummy_pyg_graph = None
  for document_content, document_class in list(zip(dataframe["Content"], dataframe["Class"])):
    prepocessed_document_content_as_tokens = preprocess_text(document_content)
    nx_graph = build_graph_from_tokens(prepocessed_document_content_as_tokens, model_word2vec)
    pyg_graph = torch_geometric.utils.from_networkx(nx_graph)
    if pyg_graph.x != None:
      pyg_graph.neg_edge_index = negative_sampling(edge_index=pyg_graph.edge_index, num_nodes=pyg_graph.x.size(0), num_neg_samples=pyg_graph.edge_index.size(1))
      pyg_graph.y = torch.tensor([int(document_class)], dtype=torch.long)
      dummy_pyg_graph = copy.deepcopy(pyg_graph)
      converted_pyg_graph_dataset.append(pyg_graph)
    else:
      converted_pyg_graph_dataset.append(dummy_pyg_graph)
  return converted_pyg_graph_dataset

def convert_dataset_to_graphs(train_subset_dataframe, test_subset_dataframe):
  train_prepocesed_documents_content = [preprocess_text(document_content) for document_content in train_subset_dataframe["Content"]]
  model_word2vec = Word2Vec(train_prepocesed_documents_content, vector_size=20, window=5, min_count=1, workers=4)
  return convert_dataframe_to_graphs(train_subset_dataframe, model_word2vec), convert_dataframe_to_graphs(test_subset_dataframe, model_word2vec)

class GCNEncoder(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
      super(GCNEncoder, self).__init__()
      self.conv1 = GCNConv(in_channels, 2 * out_channels)
      self.conv2 = GCNConv(2 * out_channels, out_channels)

  def forward(self, x, edge_index):
      x = self.conv1(x, edge_index).relu()
      return self.conv2(x, edge_index)

def train_gae(arg_model_gae, data_loader, optimizer):
  arg_model_gae.train()
  total_loss = 0
  for data in data_loader:
      optimizer.zero_grad()
      z     = arg_model_gae.encode(data.x, data.edge_index)
      loss  = arg_model_gae.recon_loss(z, data.edge_index)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
  return total_loss / len(data_loader)

def test_gae(arg_model_gae, data_loader, optimizer):
  arg_model_gae.eval()
  total_auc, total_prec = 0, 0
  for data in data_loader:
      optimizer.zero_grad()
      z = arg_model_gae.encode(data.x, data.edge_index)
      total_auc_add, total_prec_add = arg_model_gae.test(z, data.edge_index, data.neg_edge_index)
      total_auc += total_auc_add
      total_prec += total_prec_add
  return (total_auc / len(data_loader), total_prec / len(data_loader))


def graph2vec_dataframe_converter(train_subset_dataframe, test_subset_dataframe):
  print("\tConverting dataset to graphs and splits to 'train' and 'test' subsets...", end=" ")
  train_dataset, test_dataset = convert_dataset_to_graphs(train_subset_dataframe, test_subset_dataframe)
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
  print("OK")

  print("\tTraining autoencoder...")
  model_autoencoder = GAE(GCNEncoder(20, 10))
  optimizer = torch.optim.Adam(model_autoencoder.parameters(), lr=0.01)
  start_time = time.time()
  for epoch in range(1, 101):
      loss    = train_gae(model_autoencoder, train_loader, optimizer)
      auc, ap = test_gae(model_autoencoder, test_loader, optimizer)
      print('\tEpoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
  end_time = time.time()
  for classifier_name, _ in classifiers:
    results["GRAPH2VEC"][classifier_name]["fit_time_model"] = numpy.append(results["GRAPH2VEC"][classifier_name]["fit_time_model"], end_time-start_time)
  print("\tOK")

  print("\tCreating new 'train' dataset...", end=" ")
  columns_names = [f'V{column_index}' for column_index in range(10)]
  train_vectors_dataframe, train_class_dataframe = numpy.array([create_document_vector_for_graph(model_autoencoder(graph.x, graph.edge_index)) for graph in train_dataset]), train_subset_dataframe["Class"]
  train_vectors_dataframe = pandas.DataFrame(train_vectors_dataframe, columns=columns_names)
  train_vectors_with_class_dataframe = train_vectors_dataframe.assign(Class=train_subset_dataframe["Class"].tolist())
  print("OK")

  print("\tCreating new 'test' dataset...", end=" ")
  test_vectors_dataframe, test_class_dataframe = numpy.array([create_document_vector_for_graph(model_autoencoder(graph.x, graph.edge_index)) for graph in test_dataset]), test_subset_dataframe["Class"]
  test_vectors_dataframe = pandas.DataFrame(test_vectors_dataframe, columns=columns_names)
  test_vectors_with_class_dataframe = test_vectors_dataframe.assign(Class=test_subset_dataframe["Class"].tolist())
  print("OK")

  return train_vectors_with_class_dataframe, test_vectors_with_class_dataframe

import pandas
pandas.set_option('display.width', 180)

import sklearn.metrics
import sklearn.model_selection
import time

from imblearn.over_sampling import SMOTE

folds = user_folds_number

for dataset_path in datasets_paths:
  print(f"Dataset: {dataset_path.split('/')[-1]}")
  try:
    dataset_dataframe = pandas.read_csv(dataset_path)
  except:
    dataset_dataframe = pandas.read_csv(dataset_path, delimiter=";")
  stratified_kfold = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True)
  for iteration, (train_subset_indexes, test_subset_indexes) in enumerate(stratified_kfold.split(dataset_dataframe["Content"], dataset_dataframe["Class"])):
    print(f"Fold: {iteration+1}/{user_folds_number}")

    train_subset_dataframe = dataset_dataframe.iloc[train_subset_indexes]
    test_subset_dataframe = dataset_dataframe.iloc[test_subset_indexes]

    match model_name:
      case "TFIDF":
        print("Model: 'TFIDF'")
        converted_train_subset_dataframe, converted_test_subset_dataframe = tfidf_dataframe_converter(train_subset_dataframe, test_subset_dataframe)
        models_datasets = {"TFIDF": (converted_train_subset_dataframe, converted_test_subset_dataframe)}
      case "WORD2VEC":
        print("Model: 'WORD2VEC'")
        converted_train_subset_dataframe, converted_test_subset_dataframe = word2vec_dataframe_converter(train_subset_dataframe, test_subset_dataframe)
        models_datasets = {"WORD2VEC": (converted_train_subset_dataframe, converted_test_subset_dataframe)}
      case "DOC2VEC":
        print("Model: 'DOC2VEC'")
        converted_train_subset_dataframe, converted_test_subset_dataframe = doc2vec_dataframe_converter(train_subset_dataframe, test_subset_dataframe)
        models_datasets = {"DOC2VEC": (converted_train_subset_dataframe, converted_test_subset_dataframe)}
      case "GRAPH2VEC":
        print("Model: 'GRAPH2VEC'")
        converted_train_subset_dataframe, converted_test_subset_dataframe = graph2vec_dataframe_converter(train_subset_dataframe, test_subset_dataframe)
        models_datasets = {"GRAPH2VEC": (converted_train_subset_dataframe, converted_test_subset_dataframe)}
      case "ALL":
        print("Model: 'ALL Models'")
        print("\tConverting dataset (TF-IDF) ...")
        converted_train_subset_dataframe_tfidf, converted_test_subset_dataframe_tfidf         = tfidf_dataframe_converter(train_subset_dataframe.copy(), test_subset_dataframe.copy())
        print("\tConverting dataset (Word2Vec) ...")
        converted_train_subset_dataframe_word2vec, converted_test_subset_dataframe_word2vec   = word2vec_dataframe_converter(train_subset_dataframe.copy(), test_subset_dataframe.copy())
        print("\tConverting dataset (Doc2Vec) ...")
        converted_train_subset_dataframe_doc2vec, converted_test_subset_dataframe_doc2vec     = doc2vec_dataframe_converter(train_subset_dataframe.copy(), test_subset_dataframe.copy())
        print("\tConverting dataset (Graph2Vec) ...")
        converted_train_subset_dataframe_graph2vec, converted_test_subset_dataframe_graph2vec = graph2vec_dataframe_converter(train_subset_dataframe.copy(), test_subset_dataframe.copy())
        models_datasets = { "TFIDF": (converted_train_subset_dataframe_tfidf, converted_test_subset_dataframe_tfidf),
                            "WORD2VEC": (converted_train_subset_dataframe_word2vec, converted_test_subset_dataframe_word2vec),
                            "DOC2VEC": (converted_train_subset_dataframe_doc2vec, converted_test_subset_dataframe_doc2vec),
                            "GRAPH2VEC": (converted_train_subset_dataframe_graph2vec, converted_test_subset_dataframe_graph2vec)
                          }

    for current_model_name, (converted_train_subset_dataframe, converted_test_subset_dataframe) in models_datasets.items():
      print(f"\tModel: '{current_model_name}'")

      X_train, y_train  = converted_train_subset_dataframe.drop("Class", axis='columns'), converted_train_subset_dataframe["Class"]

      if smote:
        print(f"\t\tSMOTE enabled")
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

      X_test, y_test    = converted_test_subset_dataframe.drop("Class", axis='columns'), converted_test_subset_dataframe["Class"]
      for classifier_name, classifier_model in classifiers:
        print(f"\t\tTraining ({classifier_name}) ...")
        time_model_train_start = time.time()
        classifier_model.fit(X_train, y_train)
        time_model_train_end = time.time()
        time_model_test_start = time.time()
        predictions = classifier_model.predict(X_test)
        time_model_test_end = time.time()

        results[current_model_name][classifier_name]["fit_time"]        = numpy.append(results[current_model_name][classifier_name]["fit_time"], time_model_train_end-time_model_train_start)
        results[current_model_name][classifier_name]["score_time"]      = numpy.append(results[current_model_name][classifier_name]["score_time"], time_model_test_end-time_model_test_start)
        results[current_model_name][classifier_name]["test_accuracy"]   = numpy.append(results[current_model_name][classifier_name]["test_accuracy"], sklearn.metrics.accuracy_score(y_test, predictions))
        results[current_model_name][classifier_name]["test_precision"]  = numpy.append(results[current_model_name][classifier_name]["test_precision"], sklearn.metrics.precision_score(y_test, predictions))
        results[current_model_name][classifier_name]["test_recall"]     = numpy.append(results[current_model_name][classifier_name]["test_recall"], sklearn.metrics.recall_score(y_test, predictions))
        results[current_model_name][classifier_name]["test_f1"]         = numpy.append(results[current_model_name][classifier_name]["test_f1"], sklearn.metrics.f1_score(y_test, predictions))
        results[current_model_name][classifier_name]["test_roc_auc"]    = numpy.append(results[current_model_name][classifier_name]["test_roc_auc"], sklearn.metrics.roc_auc_score(y_test, predictions))

  for current_model_name in results.keys():
    print(f"{current_model_name}")
    for classifier_name in results[current_model_name].keys():
      print(classifier_name)
      prepare_partial_data(results[current_model_name][classifier_name])
      prepare_summary_data(results[current_model_name][classifier_name])
      save_to_csv(results[current_model_name][classifier_name], folder=f"{current_model_name}", filename=f"{classifier_name}")

for current_model_name in results.keys():
    print(f"{current_model_name}")
    for classifier_name in results[current_model_name].keys():
      print(classifier_name)
      prepare_partial_data(results[current_model_name][classifier_name])
      prepare_summary_data(results[current_model_name][classifier_name])
      save_to_csv(results[current_model_name][classifier_name], folder=f"{current_model_name}", filename=f"{classifier_name}")
