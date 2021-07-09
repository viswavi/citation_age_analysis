from collections import Counter
import json
import jsonlines
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import pickle
from tqdm import tqdm



CACHES_FOLDER = "PATH_REDACTED"
HOME_FOLDER = "PATH_REDACTED"

# This contains the citation graph of papers within 3 citation hops of SciREX
# (https://github.com/allenai/SciREX)
CITATION_GRAPH_PICKLE = os.path.join(CACHES_FOLDER, "citation_graph_radius_3.pkl")
S2ORC_METADATA_DIRECTORY = os.path.join(CACHES_FOLDER, "s2orc_metadata")
PAPER_YEAR_PICKLE_FILE = os.path.join(CACHES_FOLDER, "paper_year_mapping_keyword_filtered.json")
AI_PAPER_LIST = os.path.join(CACHES_FOLDER, "ai_paper_list.json")
PAPER_AGE_DIRECTORY = os.path.join(HOME_FOLDER, "relative_paper_age_plots")

def load_citation_graph():
    out_edges, _ = pickle.load(open(CITATION_GRAPH_PICKLE, 'rb'))
    return out_edges

def keyword_match(abstract):
    UNCASED_KEYPHRASES = ['classification',
                'natural language processing',
                'speech recognition',
                'speech synthesis',
                'deep learning',
                'machine learning',
                'statistical learning',
                'learning theory',
                'bandits',
                'neural ',
                'regularization',
                'machine translation',
                'reinforcement learning',
                'support vector machine',
                'question answering',
                'information extraction',
                'relation extraction',
                'named entity recognition',
                'sentiment analysis',
                'image generation',
                'artifical intelligence',
                'robotic',
                'language generation',
                'self-attention',
                'perceptron',
                'interpretability',
                'backpropagation',
                'convolutional',
                'recurrent neural',
                'deformable parts',
                'gradient',
                'nonparametric',
                'multitask',
                'dialogue generation'
                ]
    UPPERCASE_KEYWORDS = ["MT", "RL", "AI", "ML", "NLP", "CNN", "RNN", "LSTM", "SVM", "ASR"]

    abstract_lower = abstract.lower()
    matched = False
    for phrase in UNCASED_KEYPHRASES:
        if phrase in abstract_lower:
            matched = True
    for phrase in UPPERCASE_KEYWORDS:
        if phrase in abstract.split():
            matched = True
    return matched

def calculate_ages(ids):
    if os.path.exists(PAPER_YEAR_PICKLE_FILE):
        print("Loading age mapping from file.")
        paper_year_mapping = json.load(open(PAPER_YEAR_PICKLE_FILE))
        ai_paper_list = json.load(open(AI_PAPER_LIST))
    else:
        ai_paper_list = set()
        print("Pickle not found.")
        paper_year_mapping = {}
        matched_documents = 0
        # Assumes you have the entire S2ORC metadata downloaded to disk already.
        for i, metadata_file in enumerate(os.listdir(S2ORC_METADATA_DIRECTORY)):
            print(f"Opening file #{i+1}")
            for row in tqdm(jsonlines.open(os.path.join(S2ORC_METADATA_DIRECTORY, metadata_file))):
                if "paper_id" in row:
                    paper_id = row["paper_id"]
                    if paper_id in ids:
                        ids.remove(paper_id)
                        paper_year_mapping[paper_id] = row["year"]
                        matched_documents += 1
                        if row.get('mag_field_of_study', None) is None or "Computer Science" not in row['mag_field_of_study'] or len(row['mag_field_of_study']) > 1:
                            continue
                        if not isinstance(row.get('abstract', None), str) or not keyword_match(row['abstract']):
                            continue
                        ai_paper_list.add(paper_id)
            print(f"Finished processing file. {matched_documents} documents matched so far.")
        json.dump(paper_year_mapping, open(PAPER_YEAR_PICKLE_FILE, "w"))
        json.dump(list(ai_paper_list), open(AI_PAPER_LIST, "w"))

    filtered_paper_year_mapping = {}
    for paper, year in paper_year_mapping.items():
        # Filter out bad paper years with a little sanity check.
        if isinstance(year, int) and year > 1800 and year <= 2020:
            filtered_paper_year_mapping[paper] = year

    return filtered_paper_year_mapping, ai_paper_list

def all_papers_published_in(paper_ids, year_X, age_mapping):
    papers_in_year = []
    for paper_id in paper_ids:
        if paper_id not in age_mapping:
            # Paper age not included
            continue
        if age_mapping[paper_id] == year_X:
            papers_in_year.append(paper_id)
    print(f"{len(papers_in_year)} papers found published in year {year_X}")
    return papers_in_year

def draw_bar_chart(ax, x_ticks, y_ticks, ylabel=None, xlabel=None, title=None, fname="/tmp/scratch.png", num_buckets=50, xlimit=None, ylimit=None, include_x_axis=True, include_y_axis=True):
    if xlabel:
        ax.set_xlabel(xlabel)
    if include_y_axis:
        ax.set_ylabel(ylabel, rotation=90)
    else:
        ax.set_ylabel('_', rotation=90)
    if not include_y_axis:
        ax.yaxis.label.set_color('white')
    if xlimit:
        ax.set_xlim(0, xlimit)
    if ylimit:
        ax.set_ylim(0, ylimit)
    if title:
        ax.set_title(title, fontdict={'fontsize': 10})

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        if not include_x_axis:
            tick.label.set_color("white") 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
        if not include_y_axis:
            tick.label.set_color("white") 

    ax.bar(x_ticks, y_ticks, color="lightblue", width=0.8)
    # ax1.set_aspect('equal')

def make_histogram_for(ax, year_X, citation_graph, age_mapping, ai_paper_list, max_age = 50, include_x_axis=True, include_y_axis=True):
    citing_papers = list(set(citation_graph.keys()).intersection(ai_paper_list))
    paper_ages = []
    for paper_id in all_papers_published_in(citing_papers, year_X, age_mapping):
        for cited_paper in citation_graph[paper_id]:
            if cited_paper not in age_mapping:
                # Paper age not included
                continue
            age_difference = year_X - age_mapping[cited_paper]
            if age_difference < 0 or age_difference > max_age:
                continue
            paper_ages.append(age_difference)

    age_counter = Counter(paper_ages)
    ages = sorted(age_counter.keys())
    counts = [age_counter[age] for age in ages]
    normalized_counts = [float(count) / sum(counts) for count in counts]
    draw_bar_chart(ax, ages, normalized_counts,
                   title=str(year_X),
                   fname=os.path.join(PAPER_AGE_DIRECTORY, f"{year_X}.png"),
                   ylabel = "Fraction of citations",
                   ylimit=0.2,
                   include_x_axis=include_x_axis,
                   include_y_axis=include_y_axis,
                   xlimit=30)
    mean_age = "%.3f" % round(np.mean   (paper_ages), 3)
    std_age = "%.3f" % round(np.std(paper_ages), 3)
    ax.text(0.6, 0.8, f"μ: {mean_age}", fontsize=11, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.6, 0.65, f"σ: {std_age}", fontsize=11, ha='center', va='center', transform=ax.transAxes)

if __name__ == "__main__":
    out_edges = load_citation_graph()
    document_ids = set()
    for from_edge, to_edges in out_edges.items():
        document_ids.add(from_edge)
        document_ids.update(to_edges)
    print(f"{len(document_ids)} documents loaded from the citation graph.\n")
    age_mapping, ai_paper_list = calculate_ages(document_ids)
    ai_paper_list = set(ai_paper_list)

    fig = plt.figure(figsize = (12, 6))
    gs1 = gridspec.GridSpec(2, 5)
    gs1.update(wspace=0.3, hspace=0.2) # set the spacing between axes. 

    years = list(range(2002, 2021, 2))
    assert len(list(gs1)) == len(years), breakpoint()

    for i, year in enumerate(years):
        include_x_axis = i >= len(years) - 5
        include_y_axis = i % 5 == 0
        make_histogram_for(plt.subplot(gs1[i]), year, out_edges, age_mapping, ai_paper_list, include_x_axis=include_x_axis, include_y_axis=include_y_axis)

    fig.suptitle('Relative year of publication of cited works in AI over the years', fontsize=18)

    os.makedirs(PAPER_AGE_DIRECTORY, exist_ok=True)
    fname = os.path.join(PAPER_AGE_DIRECTORY, "age_of_citation_trends.png")
    print(f"Wrote figure to {fname}")
    fig.savefig(fname, dpi=400, bbox_inches='tight')
