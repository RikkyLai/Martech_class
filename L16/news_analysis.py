import requests
from bs4 import BeautifulSoup
import jieba
import jieba.analyse
import jieba.posseg as pseg
import re
from textrank4zh import TextRank4Keyword, TextRank4Sentence

# url = 'https://3w.huanqiu.com/a/c36dc8/3xqGPRBcUE6?agt=8'
url = 'https://opinion.huanqiu.com/article/3ytQxAIS1rU'
# url  = 'https://movie.douban.com/subject/1291999/'
html = requests.get(url, timeout=10)
content = html.content
# print(content)

soup = BeautifulSoup(content, 'html.parser', from_encoding='utf-8')
text = soup.get_text()
# print(text)

words = pseg.lcut(text)
news_person = {word for word, flag in words if flag=='nr'}
news_place = {word for word, flag in words if flag == 'ns'}
# print(" news person: ", news_person)
# print("news place: ", news_place)

text = re.sub('[^\u4e00-\u9fa5。，！：、；]{3,}', '', text)
# print(text)

tr4w = TextRank4Keyword()
tr4w.analyze(text=text, lower=True, window=2)
print('关键词: ')
for item in tr4w.get_keywords(20, word_min_len=2):
    print(item.word, item.weight)

tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source='all_filters')
print('摘要： ')
for item in tr4s.get_key_sentences(num=2):
    print(item.index, item.weight, item.sentence)

from wordcloud import WordCloud
import numpy as np
from lxml import etree
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# 去掉停用词
def remove_stop_words(f):
	stop_words = ['社会', '体育', '财经']
	for stop_word in stop_words:
		f = f.replace(stop_word, '')
	return f

# 生成词云
def create_word_cloud(f):
	print('根据词频，开始生成词云!')
	f = remove_stop_words(f)
	cut_text = word_tokenize(f)
	#print(cut_text)
	cut_text = " ".join(cut_text)
	wc = WordCloud(
		max_words=100,
		width=2000,
		height=1200,
        font_path = 'C:\STHeiti-Light.ttc'
    )
	wordcloud = wc.generate(cut_text)
	# 写词云图片
	wordcloud.to_file("wordcloud.jpg")
	# 显示词云文件
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()

create_word_cloud(text)