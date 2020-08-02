import os

from tqdm import tqdm

dm_single_close_quote = u"\u2019"  # noqa: WPS302
dm_double_close_quote = u"\u201d"  # noqa: WPS302
END_TOKENS = [
    ".",
    "!",
    "?",
    "...",
    "'",
    "`",
    '"',
    dm_single_close_quote,
    dm_double_close_quote,
    ")",
]  # acceptable ways to end a sentence


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """
    Adds a period to a line that is missing a period.
    """
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    # print line[-1]
    return line + " ."


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            continue
            next_is_highlight = False
        else:
            index = line.find("(CNN) -- ")
            if index > -1:
                line = line[index + len("(CNN) -- ") :]
            article_lines.append(line)

    # Make article into a single string
    article = " ".join(article_lines)
    return article


if __name__ == "__main__":
    cnn_stories_dir = "data/raw_data/cnn/stories"
    dm_stories_dir = "data/raw_data/dailymail/stories"

    articles = []
    for dir in set([dm_stories_dir]):
        for root, dirs, files in os.walk(dir):
            for name in tqdm(files):
                articles.append(get_art_abs(os.path.join(root, name)))

    with open("data/cnn_dm.txt", "w") as f:
        for item in articles:
            f.write("%s\n" % item)
