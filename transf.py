import numpy as np
import random
import os
import torch
import pandas as pd
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

rng_seed = 20

def reset_rng():
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    os.environ["PYTHONHASHSEED"] = str(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)

reset_rng()


def preprocess_function(examples):
    if objective in examples:
        label = examples[objective]
    else:
        label = 0

    examples['title'] = str.split(examples['title'], '\n')[0]
    examples['meta_description'] = str.replace(examples['meta_description'], '\n', '')
    examples['meta_description'] = str.replace(examples['meta_description'], '"/>', '')
    examples['meta_keywords'] = str(examples['meta_keywords'])

    train_string = examples['title'] + ". " + examples['meta_description'] + ". " + examples['meta_keywords']

    token_title = tokenizer(train_string, truncation=True, padding='max_length', max_length=max_length)
    token_title['label'] = float(label) / train_scale_label

    return token_title


def get_binned_folds(train_data, num_folds, rng_seed):
    # 1 - binning the labels by value
    label_list = []
    for t in train_data:
        label_list.append(t['label'])
    label_group = pd.cut(label_list, 20, labels=False)

    # 2 - R2 metric = predict outliers, especially in views case; all outliers should be in train as much as possible
    low_pop_groups_index = []
    for i in range(0, 20):
        if len(label_group[label_group == i]) < num_folds:
            low_pop_groups_index = low_pop_groups_index + list(np.where(label_group == i)[0])

    # 3 - generate fold ids
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=rng_seed)
    train_ids = []
    val_ids = []
    for fold_id, (train_index, val_index) in enumerate(skf.split(np.zeros(train.shape[0]), label_group)):
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


def compute_r2(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    r2 = r2_score(labels, logits)

    return {"r2": r2}


if not os.path.exists('models/transformers/'):
    os.makedirs('models/transformers/')

train = pd.read_csv('train_dataset_train.csv')
test = pd.read_csv('test_dataset_test.csv')

# scale depth for training - two distinct regions in data, probably algo change
train['publish_date'] = pd.to_datetime(train['publish_date'])
train['hourofyear'] = train['publish_date'].dt.dayofyear * 24 + train['publish_date'].dt.hour

test['publish_date'] = pd.to_datetime(test['publish_date'])
test['hourofyear'] = test['publish_date'].dt.dayofyear * 24 + test['publish_date'].dt.hour

early_days_depth = train.loc[train['hourofyear'] < 2365, 'depth'].to_numpy()
late_days_depth = train.loc[train['hourofyear'] >= 2365, 'depth'].to_numpy()
train.loc[train['hourofyear'] >= 2365, 'depth'] -= np.mean(late_days_depth)
train.loc[train['hourofyear'] >= 2365, 'depth'] /= np.std(late_days_depth)
train.loc[train['hourofyear'] >= 2365, 'depth'] *= np.std(early_days_depth)
train.loc[train['hourofyear'] >= 2365, 'depth'] += np.mean(early_days_depth)
train['depth'] = (train['depth'] - 1) * 4

# combine with parsed external data
article_parse_train = pd.read_csv('article_parsing_data_train.csv')
article_parse_test = pd.read_csv('article_parsing_data_test.csv')

train['left_id_24'] = train['document_id'].apply(lambda x: x[0:24])
test['left_id_24'] = test['document_id'].apply(lambda x: x[0:24])

train = pd.merge(train, article_parse_train, how='left', on='left_id_24')
test = pd.merge(test, article_parse_test, how='left', on='left_id_24')

do_train = True
do_predict = True

train_objective = ['depth', 'full_reads_percent']
max_length = 256
batch_size = 16
epochs = 4
num_folds = 6

for objective in train_objective:
    # consistent results for training for single/multi target
    reset_rng()

    if objective == 'views':
        learning_rate = 1e-5
        train_scale_label = 100000
        scale_depth = False
    elif objective == 'depth':
        learning_rate = 2e-5
        train_scale_label = 1
        scale_depth = True
    elif objective == 'full_reads_percent':
        learning_rate = 1e-5
        train_scale_label = 50
        scale_depth = False

    train_args = TrainingArguments(
        output_dir="models/",
        learning_rate=learning_rate,
        lr_scheduler_type='cosine',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        gradient_accumulation_steps=1,
        logging_steps=100,
        save_steps=200,
        save_total_limit=1,
        seed=rng_seed,
        data_seed=rng_seed,
        metric_for_best_model="r2",
        load_best_model_at_end=True,
        weight_decay=0.01,
        fp16=True,
        disable_tqdm=True,
        warmup_steps=400,
    )

    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
    tokenizer.model_max_length = max_length

    # train data into transformer
    train_data = train.copy()
    train_data = train_data.apply(preprocess_function, axis=1)

    train_ids, val_ids = get_binned_folds(train_data, num_folds, rng_seed)

    if do_train:
        for fold in range(num_folds):
            train_fold = train_data.iloc[train_ids[fold]].to_list()
            val_fold = train_data.iloc[val_ids[fold]].to_list()

            model = AutoModelForSequenceClassification.from_pretrained("sberbank-ai/ruBert-base",
                                                                       num_labels=1, problem_type='regression')
            model.cuda()
            model.train()

            model_trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=train_fold,
                eval_dataset=val_fold,
                compute_metrics=compute_r2,
            )
            model_trainer.train()
            model_trainer.save_model(f'models/transformers/{objective}_fold_{fold}/')

    if do_predict:
        with torch.no_grad():
            # empty columns for results
            train[(objective + '_predict')] = np.zeros(train.shape[0])
            test[(objective + '_predict')] = np.zeros(test.shape[0])

            for fold in range(num_folds):
                val_fold = train_data.iloc[val_ids[fold]].to_list()

                model = AutoModelForSequenceClassification.from_pretrained(f'models/transformers/{objective}_fold_{fold}')
                model.cuda()
                model.eval()

                # predicting train values on out-of-fold examples (1 model per example, sadly)
                for i in range(0, len(val_fold)):
                    v = val_fold[i].copy()
                    del v['label']

                    v['input_ids'] = torch.tensor([v['input_ids']]).cuda()
                    v['token_type_ids'] = torch.tensor([v['token_type_ids']]).cuda()
                    v['attention_mask'] = torch.tensor([v['attention_mask']]).cuda()

                    logits = model(**v).logits.cpu().detach().numpy()[0]
                    predict = logits * train_scale_label

                    if i % 100 == 0:
                        print(f'id: {val_ids[fold][i]}, text: {train.loc[val_ids[fold][i], "title"]},'
                              f' truth: {train.loc[val_ids[fold][i], objective]}, predict: {predict}')

                    # reverse scaling for depth
                    if objective == 'depth':
                        predict = predict / 4 + 1
                        if train.loc[val_ids[fold][i], 'hourofyear'] >= 2365:
                            predict -= np.mean(early_days_depth)
                            predict /= np.std(early_days_depth)
                            predict *= np.std(late_days_depth)
                            predict += np.mean(late_days_depth)

                    # saving results
                    train.loc[val_ids[fold][i], (objective + '_predict')] = predict

                # predicting test values, average of all models (6 models per example)
                test_data = test.copy()
                test_data = test_data.apply(preprocess_function, axis=1)

                for i in range(0, len(test_data)):
                    v = test_data[i].copy()
                    del v['label']

                    v['input_ids'] = torch.tensor([v['input_ids']]).cuda()
                    v['token_type_ids'] = torch.tensor([v['token_type_ids']]).cuda()
                    v['attention_mask'] = torch.tensor([v['attention_mask']]).cuda()

                    logits = model(**v).logits.cpu().detach().numpy()[0]
                    predict = logits * train_scale_label

                    # reverse scaling for depth
                    if objective == 'depth':
                        predict = predict / 4 + 1
                        if test.loc[i, 'hourofyear'] >= 2365:
                            predict -= np.mean(early_days_depth)
                            predict /= np.std(early_days_depth)
                            predict *= np.std(late_days_depth)
                            predict += np.mean(late_days_depth)

                    if i % 100 == 0:
                        print(f'id: {i}, text: {test.loc[i, "title"]}, predict: {predict}')

                    # saving results in sum
                    test.loc[i, (objective + '_predict')] = test.loc[i, (objective + '_predict')] + predict

            # end of predict, saving to files
            train_save = train[['document_id', (objective + '_predict')]]
            train_save.to_csv(f'bert_{(objective + "_predict")}.csv', index=False)

            test_save = test[['document_id', (objective + '_predict')]]
            test_save[(objective + "_predict")] = test_save[(objective + "_predict")] / num_folds
            test_save.to_csv(f'bert_{(objective + "_predict")}_test.csv', index=False)





