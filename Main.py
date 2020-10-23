from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd

malicious_urls = pd.read_csv("malicious_urls.csv")  # from https://urlhaus.abuse.ch/api/ and only used for providing a list of known malicious URLs
benign_urls = pd.read_csv("benign_urls.csv")  # from https://github.com/Anmol-Sharma/URL_CLASSIFICATION_SYSTEM/blob/master/benign_url.csv and only used for providing a list of known benign URLs

# setting viewing preferences
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


benign_urls = benign_urls.drop(['Lable'], axis=1).rename(columns={"URL": "url"})
malicious_urls = malicious_urls.drop(['id', 'threat', 'tags', 'urlhaus_link', 'reporter', 'url_status', 'dateadded'], axis=1)[:3494]  # 1747 is half the number of benign urls we have

url_lengths = []
protocols = []
domains = []
tld = []
ip_addresses = []
domain_dots = []
domain_lengths = []
is_malicious = []
paths = []
path_lengths = []

# go through each item in the malicious url dataset and retrieve info from them
for item in malicious_urls['url']:
    url_lengths.append(len(item))  # length of url
    protocols.append(item.split(':')[0])  # protocol of url
    current_domain = item.split('/')[2]
    domains.append(current_domain)  # domain of url
    current_path = '/'.join(item.split('/')[3:])
    paths.append(current_path)
    path_lengths.append(len(current_path))
    # if the the first two items of the domain are integers, append true to the array
    try:
        int(current_domain.split('.')[0])
        int(current_domain.split('.')[1])
        ip_addresses.append(1)
        tld.append("unknown")
    except ValueError:
        ip_addresses.append(0)
        tld.append('.' + current_domain.split('.')[-1])
    domain_dots.append(len(current_domain.split('.')) - 1)  # number of dots in the domain name
    domain_lengths.append(len(current_domain))  # length of domain name

    is_malicious.append(1)

# add column with URL length
malicious_urls['url_length'] = url_lengths

# add column with protocol
malicious_urls['protocol'] = protocols

# add column with domains
malicious_urls['domain'] = domains

# add column with the top level domain
malicious_urls['top-level domain'] = tld

# add column with domain lengths
malicious_urls['domain length'] = domain_lengths

# add column with whether or not the domain contains an IP address
malicious_urls['contains IP'] = ip_addresses

# add column with number of dots in the domain
malicious_urls['number of dots in domain'] = domain_dots

# add column with path
malicious_urls['path'] = paths

# add column with path length
malicious_urls['path length'] = path_lengths

# add column with whether or not the url is malicious
malicious_urls['is malicious'] = is_malicious

# remove url column because all features have been extracted
malicious_urls = malicious_urls.drop(['url'], axis=1)

# clear all previous lists to prepare for info retrieval from benign urls
url_lengths.clear()
protocols.clear()
domains.clear()
tld.clear()
ip_addresses.clear()
domain_dots.clear()
domain_lengths.clear()
is_malicious.clear()
paths.clear()
path_lengths.clear()

# go through each item in the benign url dataset and retrieve info from them
for item in benign_urls['url']:
    url_lengths.append(len(item))  # length of url
    protocols.append(item.split(':')[0])  # protocol of url
    current_domain = item.split('/')[2]
    domains.append(current_domain)  # domain of url
    current_path = '/'.join(item.split('/')[3:])
    paths.append(current_path)
    path_lengths.append(len(current_path))
    # if the the first two items of the domain are integers, append true to the array
    try:
        int(current_domain.split('.')[0])
        int(current_domain.split('.')[1])
        ip_addresses.append(1)
        tld.append("unknown")
    except ValueError:
        ip_addresses.append(0)
        tld.append('.' + current_domain.split('.')[-1])
    domain_dots.append(len(current_domain.split('.')) - 1)  # number of dots in the domain name
    domain_lengths.append(len(current_domain))  # length of domain name

    is_malicious.append(0)

# add column with URL length
benign_urls['url_length'] = url_lengths

# add column with protocol
benign_urls['protocol'] = protocols

# add column with domains
benign_urls['domain'] = domains

# add column with top-level domain
benign_urls['top-level domain'] = tld

# add column with domain lengths
benign_urls['domain length'] = domain_lengths

# add column with whether or not the domain contains an IP address
benign_urls['contains IP'] = ip_addresses

# add column with number of dots in the domain
benign_urls['number of dots in domain'] = domain_dots

# add column with path
benign_urls['path'] = paths

# add column with path length
benign_urls['path length'] = path_lengths

# add column with whether or not the url is malicious
benign_urls['is malicious'] = is_malicious

# remove url column because all features have been extracted
benign_urls = benign_urls.drop(['url'], axis=1)

# split overall dataset 80% for training and 20% for validation
benign_training = benign_urls[:2795]  # 2795 is 80% of benign urls
benign_testing = benign_urls[2795:]

malicious_training = malicious_urls[:2795]  # 2795 is 80% of malicious urls
malicious_testing = malicious_urls[2795:]

# finalise training data and testing data
training_data = pd.concat([benign_training, malicious_training])
testing_data = pd.concat([benign_testing, malicious_testing])
shuffled_testing_data = testing_data.sample(frac=1)
mlp_ground_truth = shuffled_testing_data['is malicious'].tolist()
shuffled_testing_data = shuffled_testing_data.drop(['is malicious'], axis=1)
shuffled_training_data = training_data.sample(frac=1)


###################################################
#       PREPARE DATA FOR PREDICTION PROCESS       #
###################################################

# encode textual data into numerical data
le = LabelEncoder()
encoded_training_data = shuffled_training_data.copy()
encoded_training_data['protocol'] = le.fit_transform(encoded_training_data['protocol'])
encoded_training_data['domain'] = le.fit_transform(encoded_training_data['domain'])
encoded_training_data['top-level domain'] = le.fit_transform(encoded_training_data['top-level domain'])
encoded_training_data['path'] = le.fit_transform(encoded_training_data['path'])

# scale numerical data
sc = StandardScaler()
encoded_training_data[['url_length', 'protocol', 'domain', 'top-level domain', 'domain length', 'number of dots in domain', 'path', 'path length']] = sc.fit_transform(encoded_training_data[['url_length', 'protocol', 'domain', 'top-level domain', 'domain length', 'number of dots in domain', 'path', 'path length']])

# create new mlp classifier with maximum 1000 iterations
mlp_classifier = MLPClassifier(max_iter=1000, hidden_layer_sizes=(5,))
print("Training MLP...")
mlp_classifier.fit(encoded_training_data.drop(['is malicious'], axis=1), encoded_training_data['is malicious'])
print("Done.")

# create new random forest classifier with 500 decision trees
srf = RF(n_estimators=500, n_jobs=-1, random_state=666)
print("Training RF...")
srf.fit(encoded_training_data.drop(['is malicious'], axis=1), encoded_training_data['is malicious'])
print("Done.")

###################################################
#                   USER INPUT                    #
###################################################
next_url = 'y'
while next_url.lower() != 'n' and next_url.lower() != 'no':

    input_url = input("Enter a URL:")
    # if no URL is entered, ask again
    while input_url == '' or input_url == ' ':
        print("No URL entered.")
        input_url = input("Enter a URL:")

    input_classifier = input("Enter a classifier:").lower()
    # if the classifier is not valid, ask again
    while input_classifier not in ['mlp', 'rf']:
        print("Invalid classifier")
        input_classifier = input("Enter a classifier:").lower()

    input_data = pd.DataFrame({'url': [input_url]})
    input_domain = input_url.split('/')[2]
    result = ''
    contains_ip = 0
    tld = 'unknown'

    # extract features from input url
    try:
        int(input_domain.split('.')[0])
        int(input_domain.split('.')[1])
        contains_ip = 1
    except ValueError:
        tld = '.' + input_domain.split('.')[-1]

    input_data['url_length'] = [len(input_url)]
    input_data['protocol'] = [input_url.split(':')[0]]
    input_data['domain'] = [input_domain]
    input_data['top-level domain'] = [tld]
    input_data['domain length'] = [len(input_domain)]
    input_data['contains IP'] = [contains_ip]
    input_data['number of dots in domain'] = [len(input_domain.split('.')) - 1]
    input_path = '/'.join(input_url.split('/')[3:])
    input_data['path'] = input_path
    input_data['path length'] = len(input_path)
    input_data = input_data.drop(['url'], axis=1)
    print(input_data, '\n\n')
    # merge input data with current testing data
    merged_data = pd.concat([shuffled_testing_data, input_data])

    # encode and scale merged data
    encoded_merged_data = merged_data.copy()
    encoded_merged_data['protocol'] = le.fit_transform(encoded_merged_data['protocol'])
    encoded_merged_data['domain'] = le.fit_transform(encoded_merged_data['domain'])
    encoded_merged_data['top-level domain'] = le.fit_transform(encoded_merged_data['top-level domain'])
    encoded_merged_data['path'] = le.fit_transform(encoded_merged_data['path'])
    encoded_merged_data[['url_length', 'protocol', 'domain', 'top-level domain', 'domain length',
                        'number of dots in domain', 'path', 'path length']] = sc.fit_transform(encoded_merged_data[
                                                                            ['url_length', 'protocol',
                                                                             'domain', 'top-level domain',
                                                                             'domain length',
                                                                             'number of dots in domain', 'path',
                                                                             'path length']])

    if input_classifier == 'mlp' or input_classifier == 'MLP':
        prediction = mlp_classifier.predict(encoded_merged_data)
        output = pd.DataFrame({'is_malicious_pred': prediction[:-1], 'is_malicious_truth': mlp_ground_truth})

        # calculate accuracy, precision, recall, and f1 of using mlp
        mlp_accuracy = accuracy_score(output['is_malicious_truth'], output['is_malicious_pred'])
        mlp_precision = precision_score(output['is_malicious_truth'], output['is_malicious_pred'])
        mlp_recall = recall_score(output['is_malicious_truth'], output['is_malicious_pred'])
        mlp_f1 = f1_score(output['is_malicious_truth'], output['is_malicious_pred'])

        print("\nMLP Accuracy:", mlp_accuracy)
        print("MLP Precision:", mlp_precision)
        print("MLP Recall:", mlp_recall)
        print("MLP F1 Score:", mlp_f1)

        result = 'MALICIOUS' if prediction[-1] == 1 else 'BENIGN'
        print("\nPrediction:", result)

    elif input_classifier == 'rf' or input_classifier == 'RF':
        prediction_rf = srf.predict(encoded_merged_data)
        output_rf = pd.DataFrame({'is_malicious_pred': prediction_rf[:-1], 'is_malicious_truth': mlp_ground_truth})

        # calculate accuracy, precision, recall, and f1 of using mlp
        srf_accuracy = accuracy_score(output_rf['is_malicious_truth'], output_rf['is_malicious_pred'])
        srf_precision = precision_score(output_rf['is_malicious_truth'], output_rf['is_malicious_pred'])
        srf_recall = recall_score(output_rf['is_malicious_truth'], output_rf['is_malicious_pred'])
        srf_f1 = f1_score(output_rf['is_malicious_truth'], output_rf['is_malicious_pred'])

        print("\nRF Accuracy:", srf_accuracy)
        print("RF Precision:", srf_precision)
        print("RF Recall:", srf_recall)
        print("RF F1 Score:", srf_f1)

        result = 'MALICIOUS' if prediction_rf[-1] == 1 else 'BENIGN'
        print("\nPrediction:", result)

    next_url = input("Would you like to enter another URL? (y/n)").lower()
    while next_url.lower() not in ['y', 'n', 'yes', 'no']:
        print("Invalid response")
        next_url = input("Would you like to enter another URL? (y/n)").lower()
