from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.stem.snowball import FrenchStemmer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import pelote
from ipysigma import Sigma
from ural import get_domain_name, is_shortened_url, is_url, should_follow_href

"""# Extrait le contenu de pages web."""

def parse(result):
  if result.status_code == 200:
    soup = BeautifulSoup(result.content, "html.parser")
    page_content = {
        "url": result.url,
        "title": soup.title.text if soup.title else None,
        "meta-description": soup.find("meta", attrs={'name':'description'}).get("content") if soup.find("meta", attrs={'name':'description'}) is not None else None,
        "a": [a.get("href") for a in soup.find_all("a")],
        "brut": soup.get_text()
        }
    for balise in ["h1", "h2", "h3"]:
      list_balises = soup.find_all(balise)
      page_content[balise] = [balise.text for balise in list_balises if balise is not None]

  return page_content


def scrap(liste_urls):
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0'}

  list_page_content = []

  for url in tqdm(liste_urls):
    try:
      result = requests.get(url, headers=headers, timeout=10)
      list_page_content.append(parse(result))
    except Exception as e:
      print(e)

  df = pd.DataFrame(list_page_content)

  df["backlinks"] = df["a"].map(lambda x: [href for href in x if href is not None and is_url(href)])

  return df

def crawl(corpus, depht=2):
  list_df = []
  crawled = set()

  for i in range(depht+1):
    print(f"Depht {i}")
    df_depht = scrap(corpus)
    list_df.append(df_depht)
    crawled.update(corpus)
    corpus = {link for link in set(df_depht.explode("backlinks")["backlinks"]) if link not in crawled}

  df = pd.concat(list_df)

  df["domains"] = df["backlinks"].map(lambda x: [get_domain_name(url) for url in x])

  return df

def crawl_to_domain_graph(df, filter_rs=False):
  df_exploded = df.explode("domains")

  df_links = df_exploded[["url", "domains"]].copy()

  df_links.rename(columns={"url":"source", "domains":"target"}, inplace=True)

  df_links["source"] = df_links["source"].map(get_domain_name)

  df_links["weight"] = 1

  df_links = df_links.groupby(["source", "target"]).sum().reset_index()

  if filter_rs:
    rs = ["twitter.com", "facebook.com", "youtube.com", "youtu.be", "instagram.com", "linkedin.com", "google.com", "apple.com", "dailymotion.com"]
    df_links = df_links.query("target not in @rs and source not in @rs")

  return pelote.edges_table_to_graph(df_links, directed=True)

def crawl_to_sem_graph(df):
  return

with open("urls.txt", "r") as f:
  corpus = f.readlines()

corpus = ["https://www.data.gouv.fr/fr/", "https://data.fr/", "https://www.cnil.fr/fr/definition/big-data"]

df = crawl(corpus, depht=1)

G = crawl_to_domain_graph(df, True)

sigma = Sigma(
    G,
    node_size=G.in_degree,
    node_size_range=[1, 25],

    node_metrics={'category': 'louvain'},
    node_color="category",
    node_color_gradient = "Sinebow",
    # max_categorical_colors = 100,

    default_node_border_ratio=0.2,
    node_border_color="in_playlist",
    node_border_color_palette = {True:"black", False:"white"},

    node_label_size = G.in_degree,
    node_label_size_range = (8, 16),
    node_label_color = "white",

    default_edge_type="curve",
    background_color ="black",
    start_layout=True,
    height=720
)

sigma

"""---

# Data processing.
"""

with open("stopwords.txt", "r") as f:
  stopwords_fr = f.readlines()

stopwords_fr = [stopword.replace("\n", "") for stopword in stopwords_fr]

tokener = nltk.RegexpTokenizer(r"\w+")
lemmatizer = FrenchLefffLemmatizer()
stemmer = FrenchStemmer()

def processing(txt):
  if type(txt) is str:
    tokens = [token.lower() for token in tokener.tokenize(txt) if token.lower() not in stopwords_fr]

    # lemmes = [lemmatizer.lemmatize(token, "n") for token in tokens]

    stems = [stemmer.stem(lemme) for lemme in tokens]

    return stems

def graphing(tokens):
  links = list()

  for i in range(len(tokens)-1):
    links.append(
        {
            "source":tokens[i],
            "target":tokens[i+1:]
        }
    )

  nodes = {"id":tokens}

  return nodes, links

def crawl_to_sem_graph(df):
  links = []
  nodes = []

  for balise in tqdm(["title", "meta-description", "h1", "h2", "h3"]):
    df[f"{balise}_stems"] = df[balise].map(processing)
    for stem in df[f"{balise}_stems"]:
      if stem is not None:
        result = graphing(stem)
        nodes.append(result[0])
        links += result[1]

  df_nodes = pd.DataFrame(nodes)
  df_nodes = df_nodes.explode("id")
  df_nodes["count"] = 1
  df_nodes = df_nodes.groupby("id").sum().reset_index()

  df_links = pd.DataFrame(links)
  df_links = df_links.explode("target")
  df_links["weight"] = 1
  df_links = df_links.groupby(["source", "target"]).sum().reset_index()

  return df_nodes, df_links

df_nodes, df_links = crawl_to_sem_graph(df)

df_links

G = pelote.tables_to_graph(
    df_nodes, df_links, node_col="id", node_data=["count"], edge_data=["weight"]
)

sigma = Sigma(
    G,
    node_size="count",
    node_size_range=[3, 30],

    node_metrics={'category': 'louvain'},
    node_color="category",
    node_color_gradient = "Sinebow",

    node_label_size = G.degree,
    node_label_size_range = (8, 16),
    node_label_color = "white",

    default_edge_type="curve",
    # background_color ="black",
    start_layout=True,
    height=720
)

sigma