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
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
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

        if len(corpus.get(current_page)) > 0:
            for page in sorted(corpus.get(current_page)):
                list_of_probabilities.append(transition_dict.get(page))
            current_page = random.choices(list(corpus.get(current_page)), list_of_probabilities)[0]
        else:
            for j in range(len(corpus)):
                list_of_probabilities.append((1 - damping_factor) / len(corpus))
            current_page = random.choices(sorted(corpus), list_of_probabilities)[0]
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

    page_rank_dictionary = {}
    threshold = 0.001
    n = len(corpus)
    is_over = n

    for page in corpus:
        page_rank_dictionary[page] = 1 / n

    while is_over > 0:
        for page_p in corpus:
            list_of_incoming_pages = []
            for page_i in corpus:
                if page_p in corpus.get(page_i) or len(corpus.get(page_i)) == 0:
                    list_of_incoming_pages.append(page_i)

            old_pagerank = page_rank_dictionary.get(page_p)

            summation = 0

            if len(list_of_incoming_pages) > 0:
                for incoming in list_of_incoming_pages:
                    num_links = len(corpus.get(incoming))
                    if num_links == 0:
                        num_links = len(corpus)
                    summation += page_rank_dictionary.get(incoming) / num_links
            else:
                for incoming in corpus:
                    if len(corpus.get(incoming)) == 0:
                        summation += page_rank_dictionary.get(incoming) / len(corpus)

            new_pagerank = (1 - damping_factor) / n + damping_factor * summation
            page_rank_dictionary[page_p] = new_pagerank

            if abs(new_pagerank - old_pagerank) <= threshold:
                is_over -= 1
            else:
                is_over = n

    return page_rank_dictionary


if __name__ == "__main__":
    main()
