import re
import json
import codecs
import operator
import argparse
import logging

import wikipedia
import stop_words
import pymorphy2

from math import log
from collections import defaultdict, namedtuple, Counter, OrderedDict

from tabulate import tabulate


# from https://stackoverflow.com/a/32782927/4676150
def namedtuple_asdict(obj):
  if hasattr(obj, "_asdict"): # detect namedtuple
    return OrderedDict(zip(obj._fields, (namedtuple_asdict(item) for item in obj)))
  elif isinstance(obj, str): # iterables - strings
     return obj
  elif hasattr(obj, "keys"): # iterables - mapping
     return OrderedDict(zip(obj.keys(), (namedtuple_asdict(item) for item in obj.values())))
  elif hasattr(obj, "__iter__"): # iterables - sequence
     return type(obj)((namedtuple_asdict(item) for item in obj))
  else: # non-iterable cannot contain namedtuples
    return obj


class Index:

    Doc = namedtuple('Doc', ['sz', 'title', 'url'])
    Entry = namedtuple('Entry', ['docID', 'cnt'])

    def __init__(self):
        self._inv_idx = defaultdict(list)
        self._docs = []
        self._stop_words = set(stop_words.get_stop_words('ru')).union(['.', ',', '!', '?', ':', ';'])
        self._morph = pymorphy2.MorphAnalyzer()

    def dump(self, fn):
        with codecs.open(fn, 'w', 'utf8') as fd:
            serialized = namedtuple_asdict({'docs': self._docs, 'inv_idx': self._inv_idx})
            json.dump(serialized, fd, indent=2, ensure_ascii=False)

    def load(self, fn):
        with codecs.open(fn, 'r', 'utf8') as fd:
            serialized = json.load(fd)
            self._docs = list(map(lambda doc: self.Doc(**doc), serialized['docs']))
            for term, es in serialized['inv_idx'].items():
                self._inv_idx[term] = list(map(lambda e: self.Entry(**e), es))

    def index(self, title, url, doc):
        counter = Counter()
        docID = len(self._docs)

        for term in self._tokenize(doc):
            counter[term] += 1

        totalCnt = 0
        for term, count in counter.items():
            totalCnt += count
            self._inv_idx[term].append(self.Entry(docID=docID, cnt=count))

        self._docs.append(self.Doc(sz=totalCnt, title=title, url=url))

    def query(self, qs):
        avgdl = sum(d.sz for d in self._docs) / len(self._docs)
        scores = defaultdict(int)
        for q in self._tokenize(qs):
            for e in self._inv_idx[q]:
                scores[e.docID] += self._bm25(q, e.docID, e.cnt, avgdl)
        scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

        def make_result(x):
            doc = self._docs[x[0]]
            return {'title': doc.title, 'url': doc.url, 'score': x[1]}

        return list(map(make_result, scores))

    def _bm25(self, termID, docID, cnt, avgdl):

        N = len(self._docs)
        n = len(self._inv_idx[termID])
        idf = log(N - n + 0.5 / (n + 0.5))

        k = 1.5
        b = 0.75
        D = self._docs[docID].sz
        freq = cnt / float(D)
        Dq = freq * (k + 1) / (freq + k * (1 - b + b * D / avgdl))

        return idf * Dq

    def _tokenize(self, text):
        def check(word):
            return word not in self._stop_words

        def normalize(word):
            return self._morph.parse(word.lower())[0].normal_form

        return map(normalize, filter(check, re.findall(r"[\w']+|[.,!?:;]", text)))


def crawl(start, n, callback):
    i = 0
    queue = [start]
    visited = set(start)
    wikipedia.set_lang('ru')

    def get_page(title):
        return wikipedia.page(title, redirect=False)

    def process_page(title):
        try:
            page = get_page(title)
        except:
            logging.exception("Exception during request to page {}".format(title))
            return
        nonlocal i
        i += 1
        print("Processing page #{} {} ...".format(i, title))
        for link in page.links:
            if link not in visited:
                visited.add(link)
                queue.append(link)
        callback(page.title, page.url, page.content)

    while (i < n) and queue:
        try:
            process_page(queue.pop())
        except wikipedia.DisambiguationError as exc:
            for title in exc.options:
                if i >= n:
                    return
                if title in visited:
                    continue
                try:
                    process_page(title)
                except wikipedia.DisambiguationError:
                    continue


def index(args):
    idx = Index()
    crawl(args.start, args.n, idx.index)
    idx.dump(args.dump + '.json')


def query_repl(args):
    idx = Index()
    idx.load(args.load + '.json')
    n = args.n

    while True:
        query = input("Enter search query (or q for exit):")
        if query == "q":
            print("Exit")
            return
        results = idx.query(query)
        results = [(i+1, x['score'], x['title'], x['url']) for i, x in enumerate(results)]
        print(tabulate(results[:n], headers=['#n', 'score', 'title', 'url']))


def main():
    parser = argparse.ArgumentParser(description='Simpe Wikipedia Search Engine')

    subparsers = parser.add_subparsers()
    index_parser = subparsers.add_parser('index')
    query_parser = subparsers.add_parser('query')

    index_parser.add_argument('-n', type=int, default=1000, help='total number of pages to be indexed')
    index_parser.add_argument('--start', type=str, default='Main_Page', help='starting page')
    index_parser.add_argument('--dump', type=str, default='index', help='dump index to filesystem')
    index_parser.set_defaults(func=index)

    query_parser.add_argument('-n', type=int, default=3, help='number of displayed results')
    query_parser.add_argument('--load', type=str, default='index', help='file containing index')
    query_parser.set_defaults(func=query_repl)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()