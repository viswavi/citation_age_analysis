from collections import Counter
import json
import jsonlines
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

CITATION_GRAPH_PICKLE = "/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/citation_graph_radius_3.pkl"
S2ORC_METADATA_DIRECTORY = "/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/s2orc_metadata"
PAPER_YEAR_PICKLE_FILE = "/home/vijayv/pickle_backups/paper_year_mapping.json"
PAPER_AGE_DIRECTORY = "/home/vijayv/relative_paper_age_plots"

def load_citation_graph():
    out_edges, _ = pickle.load(open(CITATION_GRAPH_PICKLE, 'rb'))
    return out_edges

def calculate_ages(ids):
    if os.path.exists(PAPER_YEAR_PICKLE_FILE):
        print("Loading age mapping from file.")
        paper_year_mapping = json.load(open(PAPER_YEAR_PICKLE_FILE))
    else:
        print("Pickle not found.")
        paper_year_mapping = {}
        matched_documents = 0
        for i, metadata_file in enumerate(os.listdir(S2ORC_METADATA_DIRECTORY)):
            print(f"Opening file #{i+1}")
            for row in tqdm(jsonlines.open(os.path.join(S2ORC_METADATA_DIRECTORY, metadata_file))):
                if "paper_id" in row:
                    paper_id = row["paper_id"]
                    if paper_id in ids:
                        ids.remove(paper_id)
                        paper_year_mapping[paper_id] = row["year"]
                        matched_documents += 1
            print(f"Finished processing file. {matched_documents} documents matched so far.")
        json.dump(paper_year_mapping, open("/home/vijayv/pickle_backups/paper_year_mapping.json", "w"))

    filtered_paper_year_mapping = {}
    for paper, year in paper_year_mapping.items():
        # Filter out bad paper years with a little sanity check.
        if isinstance(year, int) and year > 1800 and year <= 2020:
            filtered_paper_year_mapping[paper] = year

    return filtered_paper_year_mapping

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

def draw_bar_chart(x_ticks, y_ticks, ylabel=None, xlabel=None, title=None, fname="/tmp/scratch.png", num_buckets=100, xlimit=None):
    fig, ax = plt.subplots()
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=90)
    if xlimit:
        ax.set_xlim(0, xlimit)
    if title:
        ax.set_title(title)

    ax.bar(x_ticks, y_ticks, color="lightblue", width=0.8)
    print(f"Wrote figure to {fname}")
    fig.savefig(fname, dpi=400, bbox_inches='tight')
    plt.tight_layout()
    del fig

def make_histogram_for(year_X, citation_graph, age_mapping, max_age = 100):
    citing_papers = list(citation_graph.keys())
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
    draw_bar_chart(ages, normalized_counts,
                   ylabel="Normalized frequency",
                   xlabel="Relative year of publication",
                   title=f"Relative year of publication of cited works in year {year_X}",
                   fname=os.path.join(PAPER_AGE_DIRECTORY, f"{year_X}.png"))

if __name__ == "__main__":
    out_edges = load_citation_graph()
    document_ids = set()
    for from_edge, to_edges in out_edges.items():
        document_ids.add(from_edge)
        document_ids.update(to_edges)
    print(f"{len(document_ids)} documents loaded from the citation graph.\n")
    age_mapping = calculate_ages(document_ids)
    
    years = [2020, 2014, 2007, 2000]
    for year in years:
        make_histogram_for(year, out_edges, age_mapping)
