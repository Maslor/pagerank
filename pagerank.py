import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    sum_sampling = 0
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
        sum_sampling += ranks[page]
    print(f"Total iteration (should be 1): {sum_sampling}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    transition_dictionary = {}

    linked_pages = corpus.get(page)

    probability_random = (1 - damping_factor) / len(corpus)

    if len(linked_pages) > 0:
        probability_linked = damping_factor / len(linked_pages)

        for page in linked_pages:
            transition_dictionary[page] = probability_linked + probability_random

    for page in corpus:
        if page not in transition_dictionary:
            transition_dictionary[page] = probability_random

    return transition_dictionary


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    sample_list = []

    random_page = random.sample(sorted(corpus), 1)[0]
    current_page = random_page

    i = 0
    while i < n:
        sample_list.append(current_page)
        transition_dict = transition_model(corpus, current_page, damping_factor)
        list_of_probabilities = []
        sorted_transition = sorted(transition_dict)
        for page in sorted_transition:
            list_of_probabilities.append(transition_dict.get(page))
        list_of_probabilities = transition_dict.values()
        current_page = random.choices(sorted_transition, list_of_probabilities)[0]
        i += 1

    sample_dictionary = {}
    for page in corpus:
        if page in sample_list:
            sample_dictionary[page] = sample_list.count(page) / n
        else:
            sample_dictionary[page] = 0

    return sample_dictionary


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
