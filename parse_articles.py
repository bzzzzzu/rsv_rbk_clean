import regex
import os
import pandas as pd

regex_len = regex.compile('data-chars-length=".*?"')
regex_id = regex.compile('data-id="[0-9a-f]{24}"')
regex_word_count = regex.compile('article_word_count.*"')
regex_data_index = regex.compile('data-index=".*?"')
regex_data_type = regex.compile('data-type=".*?"')
regex_data_category = regex.compile('data-category=".*?"')
regex_data_agg = regex.compile('data-aggregator=".*?"')
regex_content = regex.compile('content=".*?"')

def scan_dataset(dir):
    files = os.listdir(dir)
    data = []

    for i, article in enumerate(files):
        with open(f'{dir}{article}', 'r', encoding='utf-8') as f:
            article_text = f.readlines()

        final_len = 0
        final_id = ''
        final_word_count = 0
        data_type = []
        final_data_category = ''
        final_data_agg = ''
        author_org = ''
        date_published = ''
        date_modified = ''
        meta_title = ''
        meta_description = ''
        meta_keywords = ''
        meta_copyright = ''
        og_type = ''
        og_url_type = ''
        ad_cat_id = 0

        ignore_next_video = 0
        get_next_author_org = 0
        active_description = 0

        # line-by-line manual parsing
        for line in article_text:
            # conditional searches
            if active_description == 1:
                meta_description = meta_description + line
            if get_next_author_org == 1:
                author_org = regex.findall(regex_content, line)
                author_org = str.split(author_org[0], '"')[1]
                get_next_author_org = 0

            # regex searches
            article_len = regex.findall(regex_len, line)
            article_id = regex.findall(regex_id, line)
            article_word_count = regex.findall(regex_word_count, line)
            article_data_index = regex.findall(regex_data_index, line)
            article_data_type = regex.findall(regex_data_type, line)
            article_data_category = regex.findall(regex_data_category, line)
            article_data_agg = regex.findall(regex_data_agg, line)

            # and their parsing if matched
            if len(article_len) > 0:
                final_len = int(str.split(article_len[0], '"')[1])
            if len(article_id) > 0:
                final_id = str.split(article_id[0], '"')[1]
            if len(article_word_count) > 0:
                final_word_count = int(str.split(article_word_count[0], '"')[1])
            if len(article_data_index) > 0:
                final_index = str.split(article_data_index[0], '"')[1]
            if len(article_data_type) > 0:
                if ignore_next_video == 0:
                    if 'topline' not in line:
                        data_type.append(str.split(article_data_type[0], '"')[1])
            if len(article_data_category) > 0:
                final_data_category = str.split(article_data_category[0], '"')[1]
            if len(article_data_agg) > 0:
                final_data_agg = str.split(article_data_agg[0], '"')[1]

            # simple searches for "content=x" or similar
            if 'itemprop="datePublished"' in line:
                date_published = regex.findall(regex_content, line)
                date_published = str.split(date_published[0], '"')[1]
            if 'itemprop="dateModified"' in line:
                date_modified = regex.findall(regex_content, line)
                date_modified = str.split(date_modified[0], '"')[1]
            if '<meta name="title" content="' in line:
                meta_title = regex.findall(regex_content, line)
                meta_title = str.split(meta_title[0], '"')[1]
            if '<meta name="description" content="' in line:
                meta_description = str.split(line, "content=")[1]
                active_description = 1
            if '<meta name="keywords" content="' in line:
                meta_keywords = regex.findall(regex_content, line)
                meta_keywords = str.split(meta_keywords[0], '"')[1]
            if '<meta name="copyright" content="' in line:
                meta_copyright = regex.findall(regex_content, line)
                meta_copyright = str.split(meta_copyright[0], '"')[1]
            if 'meta property="og:type"' in line:
                og_type = regex.findall(regex_content, line)
                og_type = str.split(og_type[0], '"')[1]
            if 'meta property="og:url"' in line:
                og_url_type = regex.findall(regex_content, line)
                og_url_type = str.split(og_url_type[0], '"')[1]
                og_url_type = str.split(og_url_type, '/')[3]
            if "'cat_id':" in line:
                ad_cat_id = str.split(line, ": ")[1]

            # setting condition for later lines
            if '"/>' in line:
                active_description = 0
            ignore_next_video = 0
            if 'data-role="fox-tail"' in line:
                ignore_next_video = 1
            if '<div itemprop="author" itemscope itemtype="https://schema.org/Organization">' in line:
                get_next_author_org = 1

        not_modified = (date_published == date_modified)

        data.append([final_id, final_len, final_word_count, final_index, data_type,
                     final_data_category, final_data_agg, author_org, not_modified,
                     meta_title, meta_description, meta_keywords, meta_copyright,
                     og_type, og_url_type, ad_cat_id,])

        if i % 100 == 0:
            print(f'i: {i}, article id: {final_id}, article len: {final_len}')

    return data

data_train = scan_dataset('articles/train/')
data_train = pd.DataFrame(data_train, columns=['left_id_24', 'article_len', 'article_word_count', 'data_index',
                                               'data_type', 'data_category', 'data_agg', 'author_org', 'not_modified',
                                               'meta_title', 'meta_description', 'meta_keywords', 'meta_copyright',
                                               'og_type', 'og_url_type', 'ad_cat_id',])
data_train.to_csv('article_parsing_data_train.csv', index=False)

data_test = scan_dataset('articles/test/')
data_test = pd.DataFrame(data_test, columns=['left_id_24', 'article_len', 'article_word_count', 'data_index',
                                             'data_type', 'data_category', 'data_agg', 'author_org', 'not_modified',
                                             'meta_title', 'meta_description', 'meta_keywords', 'meta_copyright',
                                             'og_type', 'og_url_type', 'ad_cat_id',])
data_test.to_csv('article_parsing_data_test.csv', index=False)