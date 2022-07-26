import os
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
import random
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

rng_seed = 20

def reset_rng(rng_seed):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    np.set_printoptions(precision=4, suppress=True)

reset_rng(rng_seed)


def str_to_list(input):
    input = str.replace(input, '[', '')
    input = str.replace(input, ']', '')
    input = str.replace(input, ' ', '')
    input = str.replace(input, '\'', '')
    output = input.split(',')
    if output == ['']:
        output = []
    return output


def column_to_cat(data, col_name):
    data[col_name] = data[col_name].astype('category')
    data[col_name] = data[col_name].cat.codes
    data[col_name] = data[col_name].astype('int')

    return data


def mlb_on_column(data, col_name, amount):
    mlb = MultiLabelBinarizer()

    # fix statistics leak from test
    mlb.fit_transform(data[col_name].iloc[0:7000])
    cols = mlb.transform(data[col_name])
    cols_df = pd.DataFrame(cols, columns=mlb.classes_)

    if amount > 0:
        sum_of_cols = np.sum(cols, axis=0)
        keep_cols_index = np.argsort(-sum_of_cols)[:amount]
        cols_df = cols_df.iloc[:, keep_cols_index]

    data = pd.concat([data, cols_df], axis=1)
    column_names = cols_df.columns.tolist()

    return data, column_names


def read_multi_seeds(data, fname, col_name):
    data[col_name] = np.zeros(data.shape[0])
    num_seeds = 0
    flist = os.listdir()
    for f in flist:
        if fname in f:
            bert_depth = pd.read_csv(f)
            data[col_name] = data[col_name] + bert_depth[col_name]
            num_seeds = num_seeds + 1
    if num_seeds > 0:
        data[col_name] = data[col_name] / num_seeds


def prepare_dataset(train, test, settings):
    train_data = train.copy()
    test_data = test.copy()

    article_parse_train = pd.read_csv('article_parsing_data_train.csv')
    article_parse_test = pd.read_csv('article_parsing_data_test.csv')

    input_cols = []
    cat_feature_cols = []

    # merging various data sources
    train_data['left_id_24'] = train_data['document_id'].apply(lambda x: x[0:24])
    test_data['left_id_24'] = test_data['document_id'].apply(lambda x: x[0:24])
    train_data = pd.merge(train_data, article_parse_train, how='left', on='left_id_24')
    test_data = pd.merge(test_data, article_parse_test, how='left', on='left_id_24')

    read_multi_seeds(train_data, 'bert_depth_predict_seed', 'depth_predict')
    train_data.loc[train_data['depth_predict'] == 0, 'depth_predict'] = 1
    read_multi_seeds(test_data, 'bert_depth_predict_test_seed', 'depth_predict')
    test_data.loc[test_data['depth_predict'] == 0, 'depth_predict'] = 1

    if settings['transformers_depth']:
        input_cols = input_cols + ['depth_predict']

    read_multi_seeds(train_data, 'bert_full_reads_percent_predict_seed', 'full_reads_percent_predict')
    train_data.loc[train_data['full_reads_percent_predict'] == 0, 'full_reads_percent_predict'] = 30
    read_multi_seeds(test_data, 'bert_full_reads_percent_predict_test_seed', 'full_reads_percent_predict')
    test_data.loc[test_data['full_reads_percent_predict'] == 0, 'full_reads_percent_predict'] = 30

    if settings['transformers_full_reads_percent']:
        input_cols = input_cols + ['full_reads_percent_predict']

    # working on combined data for consistency in train-test feature labels
    combined_data = pd.concat([train_data, test_data], axis=0)
    combined_data = combined_data.reset_index()

    combined_data['publish_date'] = pd.to_datetime(combined_data['publish_date'])

    combined_data['day'] = combined_data['publish_date'].dt.day
    combined_data['month'] = combined_data['publish_date'].dt.month
    combined_data['dayofyear'] = combined_data['publish_date'].dt.dayofyear
    combined_data['dayofweek'] = combined_data['publish_date'].dt.dayofweek
    combined_data['hour'] = combined_data['publish_date'].dt.hour
    combined_data['minute'] = combined_data['publish_date'].dt.minute
    combined_data['second'] = combined_data['publish_date'].dt.second
    combined_data['minuteofday'] = combined_data['publish_date'].dt.hour * 60 + combined_data['publish_date'].dt.minute
    combined_data['hourofyear'] = combined_data['publish_date'].dt.dayofyear * 24 + combined_data['publish_date'].dt.hour

    if settings['day']:
        input_cols = input_cols + ['day']
    if settings['month']:
        input_cols = input_cols + ['month']
    if settings['dayofyear']:
        input_cols = input_cols + ['dayofyear']
    if settings['dayofweek']:
        input_cols = input_cols + ['dayofweek']
    if settings['hour']:
        input_cols = input_cols + ['hour']
    if settings['minute']:
        input_cols = input_cols + ['minute']
    if settings['minuteofday']:
        input_cols = input_cols + ['minuteofday']
    if settings['second']:
        input_cols = input_cols + ['second']
    if settings['hourofyear']:
        input_cols = input_cols + ['hourofyear']

    if settings['ctr']:
        input_cols = input_cols + ['ctr']

    if settings['article_len']:
        input_cols = input_cols + ['article_len']
    if settings['article_word_count']:
        input_cols = input_cols + ['article_word_count']
    if settings['og_url_type']:
        combined_data = column_to_cat(combined_data, 'og_url_type')

        input_cols = input_cols + ['og_url_type']
        cat_feature_cols = cat_feature_cols + ['og_url_type']
    if settings['author_org']:
        combined_data['author_org'] = combined_data['author_org'].fillna(0)
        combined_data = column_to_cat(combined_data, 'author_org')

        input_cols = input_cols + ['author_org']
        cat_feature_cols = cat_feature_cols + ['author_org']
    if settings['data_category']:
        combined_data = column_to_cat(combined_data, 'data_category')

        input_cols = input_cols + ['data_category']
        cat_feature_cols = cat_feature_cols + ['data_category']
    if settings['data_5_type']:
        def check_type(x):
            if 'photoreport' in x: return 0
            if 'article' in x: return 1
            if 'short_news' in x: return 2
            if 'interview' in x: return 3
            if 'opinion' in x: return 4
            return 5
        combined_data['data_5_type'] = combined_data['data_type']
        combined_data['data_5_type'] = combined_data['data_5_type'].apply(check_type)

        input_cols = input_cols + ['data_5_type']
        cat_feature_cols = cat_feature_cols + ['data_5_type']
    if settings['data_type']:
        combined_data = column_to_cat(combined_data, 'data_type')

        input_cols = input_cols + ['data_type']
        cat_feature_cols = cat_feature_cols + ['data_type']
    if settings['inline_articles']:
        input_cols = input_cols + ['inline_articles']
    if settings['hrefs_rbc']:
        input_cols = input_cols + ['hrefs_rbc']
    if settings['hrefs_all']:
        input_cols = input_cols + ['hrefs_all']
    if settings['sentence_count']:
        input_cols = input_cols + ['sentence_count']
    if settings['picture_count']:
        input_cols = input_cols + ['picture_count']

    if settings['left_id']:
        combined_data['left_id'] = combined_data['document_id'].apply(lambda x: x[0:8])
        combined_data['left_id'] = combined_data['left_id'].apply(str_to_list)

        combined_data, column_names = mlb_on_column(combined_data, 'left_id', settings['keep_left_id'])

        input_cols = input_cols + column_names

    if settings['tags'] == 'onehot':
        combined_data['tags'] = combined_data['tags'].apply(str_to_list)

        combined_data, column_names = mlb_on_column(combined_data, 'tags', settings['keep_tags'])

        input_cols = input_cols + column_names
    elif settings['tags'] == 'int':
        combined_data = column_to_cat(combined_data, 'tags')

        input_cols = input_cols + ['tags']
        cat_feature_cols = cat_feature_cols + ['tags']

    if settings['category'] == 'onehot':
        combined_data['category'] = combined_data['category'].apply(str_to_list)

        combined_data, column_names = mlb_on_column(combined_data, 'category', settings['keep_category'])

        input_cols = input_cols + column_names
    elif settings['category'] == 'int':
        combined_data = column_to_cat(combined_data, 'category')

        input_cols = input_cols + ['category']
        cat_feature_cols = cat_feature_cols + ['category']

    if settings['authors'] == 'onehot':
        combined_data['authors'] = combined_data['authors'].apply(str_to_list)

        combined_data, column_names = mlb_on_column(combined_data, 'authors', settings['keep_authors'])

        input_cols = input_cols + column_names
    elif settings['authors'] == 'int':
        combined_data = column_to_cat(combined_data, 'authors')

        input_cols = input_cols + ['authors']
        cat_feature_cols = cat_feature_cols + ['authors']

    if settings['keywords'] == 'onehot':
        combined_data['meta_keywords'] = combined_data['meta_keywords'].apply(str_to_list)

        combined_data, column_names = mlb_on_column(combined_data, 'meta_keywords', settings['keep_keywords'])

        input_cols = input_cols + column_names

    train_data = combined_data.iloc[0:7000].copy().reset_index()
    test_data = combined_data.iloc[7000:10000].copy().reset_index()

    return train_data, test_data, input_cols, cat_feature_cols


def get_binned_folds(train_data, bin_column, num_folds, rng_seed):
    # 1 - binning the labels by value
    label_group = pd.cut(train_data[bin_column], 20, labels=False)

    # 2 - R2 metric = predict outliers, especially in views case; all outliers should be in train as much as possible
    low_pop_groups_index = []
    for i in range(0, 20):
        if len(label_group[label_group == i]) < num_folds:
            low_pop_groups_index = low_pop_groups_index + list(np.where(label_group == i)[0])

    # 3 - generate fold ids
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=rng_seed)
    train_ids = []
    val_ids = []
    for fold_id, (train_index, val_index) in enumerate(skf.split(np.zeros(train_data.shape[0]), label_group)):
        train_index = list(train_index)
        val_index = list(val_index)
        # 2 - add outliers back to training data
        for inx in low_pop_groups_index:
            if inx not in train_index:
                train_index.append(inx)
            if inx in val_index:
                val_index.remove(inx)
        random.shuffle(train_index)

        train_ids.append(train_index)
        val_ids.append(val_index)

    return train_ids, val_ids


train = pd.read_csv('train_dataset_train.csv')
test = pd.read_csv('test_dataset_test.csv')

train_targets = {
    'views': 'cat_settings_views.json',
    'depth': 'cat_settings_depth.json',
    'full_reads_percent': 'cat_settings_full_reads_percent.json',
}
rng_seeds = [0, 1, 2, 3, 4]

sub = {}
cv_scores = {}
for target in train_targets:

    cv_results = []
    test_results = []

    for add_seed in rng_seeds:
        with open(f'cat_settings_{target}.json', 'r') as jsf:
            settings = json.load(jsf)

        settings['train_settings']['random_seed'] = settings['train_settings']['random_seed'] + add_seed
        reset_rng(settings['train_settings']['random_seed'])

        if not os.path.exists(f'models/catboost/{settings["train_settings"]["random_seed"]}/{target}/'):
            os.makedirs(f'models/catboost/{settings["train_settings"]["random_seed"]}/{target}/')

        train_data, test_data, input_cols, cat_feature_cols = prepare_dataset(train, test, settings)

        train_ids, val_ids = get_binned_folds(train_data, target, settings['num_folds'], settings['train_settings']['random_seed'])

        cv_result = np.zeros(7000) - 1
        test_result = np.zeros(3000)

        for i in range(0, settings['num_folds']):
            fold_train_data = train_data.loc[train_ids[i]]
            fold_val_data = train_data.loc[val_ids[i]]

            print(f'fold: {i}, input columns total: {len(input_cols)}, categorical columns total: {len(cat_feature_cols)}')

            fold_train_history = fold_train_data[input_cols]
            fold_train_future = fold_train_data[target]
            fold_val_history = fold_val_data[input_cols]
            fold_val_future = fold_val_data[target]

            model = CatBoostRegressor(**settings['train_settings'])
            train_pool = Pool(fold_train_history, label=fold_train_future, cat_features=cat_feature_cols)
            val_pool = Pool(fold_val_history, label=fold_val_future, cat_features=cat_feature_cols)
            model.fit(X=train_pool, verbose=int(np.ceil(settings['train_settings']['iterations'] / 50)),
                      eval_set=val_pool, use_best_model=True)

            fold_result = model.predict(fold_val_history)

            model.save_model(f'models/catboost/{settings["train_settings"]["random_seed"]}/{target}/fold_{i}.cbm')

            if target == 'views':
                fold_result[fold_result < 0] = 0

            cv_result[val_ids[i]] = fold_result

            fold_score = np.round(r2_score(fold_val_future, fold_result), 4)
            print(f'target: {target}, seed: {settings["train_settings"]["random_seed"]}, fold: {i}, fold score: {fold_score}')
            print('-------------------')

            test_history = test_data[input_cols]
            test_pool = Pool(test_history, cat_features=cat_feature_cols)
            fold_test_result = model.predict(test_history)
            if target == 'views':
                fold_test_result[fold_test_result < 0] = 0
            test_result = test_result + fold_test_result

        test_result = test_result / settings['num_folds']
        has_val_score = np.where(cv_result != -1)[0]

        if settings['transformers_raw_mix']:
            for mix in range(1, 30):
                bert_result = train_data[(target + '_predict')].iloc[has_val_score]
                cv_mix = cv_result[has_val_score] * (100 - mix) * 0.01 + bert_result[has_val_score] * mix * 0.01
                cv_mix_score = np.round(r2_score(train_data[target].iloc[has_val_score], cv_mix), 4)
                print(f'target: {target}, cv mix ({100 - mix}/{mix}): {cv_mix_score}')

            transformers_data = train_data[(target + '_predict')].iloc[has_val_score]
            cv_result[has_val_score] = cv_result[has_val_score] * (1 - settings['transformers_raw_mix']) + transformers_data * settings['transformers_raw_mix']

            transformers_test_data = test_data[(target + '_predict')]
            test_result = test_result * (1 - settings['transformers_raw_mix']) + transformers_test_data * settings['transformers_raw_mix']

        cv_results.append(cv_result)
        test_results.append(test_result)

        cv_score = np.round(r2_score(train_data[target].iloc[has_val_score], cv_result[cv_result != -1]), 4)

        print(f'target: {target}, seed: {settings["train_settings"]["random_seed"]}, cv score: {cv_score}')
        print('-------------------')

    cv_results = np.mean(cv_results, axis=0)
    test_results = np.mean(test_results, axis=0)

    has_val_score = np.where(cv_results != -1)[0]
    cv_score = np.round(r2_score(train_data[target].iloc[has_val_score], cv_results[cv_results != -1]), 4)
    cv_scores[target] = cv_score

    sub[target] = test_results

score_final = 0
for t in train_targets:
    if t == 'views':
        score_final = score_final + cv_scores[t] * 0.4
    if t == 'depth' or t == 'full_reads_percent':
        score_final = score_final + cv_scores[t] * 0.3
    print(f'score for {t}: {cv_scores[t]}')
score_final = np.round(score_final, 4)
print(f'final score: {score_final}')

sub_df = pd.DataFrame({'document_id': test_data['document_id'], 'views': sub['views'], 'depth': sub['depth'], 'full_reads_percent': sub['full_reads_percent']})
sub_df.to_csv(f'submission_{score_final}.csv', index=False)
