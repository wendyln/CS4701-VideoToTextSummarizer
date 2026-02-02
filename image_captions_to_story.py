from transformers import pipeline
import nltk
import math
from wordfreq import word_frequency
import re
import numpy as np

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def captions_to_story_pretrained(captions):
    """
    Function to take in captions, and output a story (a summary of the 
    video, with overall insights)
    """
    long_text = '.'.join(captions)
    summarizer = pipeline("summarization")
    result = summarizer(long_text)
    print (result)


def preprocessing(captions):
    """
    Takes in a list of captions (or just a long string of text), returns 
    1. tokenized matrix of words in each sentence (list of lists)
    2. the actual sentences (list of strings)
    """
    # handles strings, splitting into list
    if type(captions) == str:
        captions = re.split(r"(?<=[.?!])", captions)
    # ensure every caption ends in a period 
    for i in range(len(captions)):
        if len(captions[i]) == 0:
          continue
        if captions[i][-1] not in  ".?!":
            captions[i] = captions[i] + "."
    paragraph = ' '.join(captions)
    sentences = nltk.sent_tokenize(paragraph)
    res = []
    for sentence in sentences:
        res.append(nltk.word_tokenize(sentence))
    return res, sentences

def build_frequency_table(tokens, stops):
    """
    Builds frequency table of words in a sentence (a mapping of words
    to the number of times they appear). Does not include "stop" words
    (e.g. , or . or ! or ?)
    """
    table = {}
    for word in tokens:
        if word not in stops:
            to_add = word.lower()
            if to_add in table:
                table[to_add] += 1
            else:
                table[to_add] = 1
    return table

def build_frequency_matrix(sentences, stops):
    """
    Builds frequency matrix. For each sentence, a mapping from words to the
    number of times they appear in the setence. Excludes words in "stops" (
    , or . or ! or ?)
    """
    matrix = {}
    for i in range(len(sentences)):
        table = build_frequency_table(sentences[i], stops)
        matrix[i] = table
    return matrix

def build_tf_matrix(matrix):
    """
    Takes in word frequency matrix, outputs TF matrix (a matrix of 
    word probabilities in each sentence). Essentially, normalizing by the 
    number of words in each sentence, so the sum of each row is 1
    """
    res = {}
    for key, table in matrix.items():
        tot = sum(table.values())
        freq_table = {}
        for word, count in table.items():
            freq_table[word] = count/tot
        res[key] = freq_table
    return res

def build_appearing_matrix(matrix):
    """
    Takes in word frequency (or tf) matrix - works with either - and produces
    a mapping from word to number of sentences it appears in. This is needed
    to produce idf matrix
    """
    appearances = {}
    for key, sentence in matrix.items():
        for word in sentence:
            if word in appearances:
                appearances[word] += 1
            else:
                appearances[word] = 1
    return appearances
            
def build_idf_matrix(freq_matrix, appearing_matrix):
    """
    Takes in frequence of each word in each sentence (freq matrix), and the 
    number of sentences each word appears in (appearing_matrix) and produces
    the idf matrix
    """
    idf_matrix = {}
    num_sentences = len(freq_matrix)
    for key, sentence in freq_matrix.items():
        idf_table = {}
        for word in sentence:
            idf_table[word] = math.log(num_sentences/appearing_matrix[word])
        idf_matrix[key] = idf_table
    return idf_matrix
            
def build_tf_idf_matrix(tf_matrix, idf_matrix):
    """
    Builds tf-idf matrix for tf-idf algorithm, using both tf and idf
    matrices
    """
    tf_idf_matrix = {}
    for key, sentence in tf_matrix.items():
        idf_sentence = idf_matrix[key]
        tf_idf_table = {}
        for word in idf_sentence:
            tf_idf_table[word] = sentence[word] * idf_sentence[word]
        tf_idf_matrix[key] = tf_idf_table
    return tf_idf_matrix
            
def sentence_scoring(tf_idf_matrix):
    """
    Returns average tf-idf score of words in a sentence. To be used 
    when selecting sentences in the tf-idf algorithm
    """
    scores = {}
    for key, sentence in tf_idf_matrix.items():
        avg_score = np.mean(list(sentence.values()))
        scores[key] = avg_score
    return scores

def choose_sentences_tfidf(sentence_scores, threshold):
    """
    Takes in sentences scores (as calculated by
    tf-idf algo) and a threshold for inclusion (e.g. 1.3). Returns
    sentences to be included in summary
    """
    scores = list(sentence_scores.values())
    average = np.mean(scores)
    sd = np.std(scores)
    res = []
    for key in sentence_scores:
        if sentence_scores[key] > (average + sd * threshold):
            res.append(key)
    return res

def choose_sentences_tfidf_modified(sentence_scores, idf_matrix, threshold):
    """
    Modified version of the tf-idf algorithm. Chooses sentences both according
    to tf-idf, and sentences with relatively uncommon english words that appear 
    frequently in the text
    """
    # first, find tf-idf sentences to include
    scores = list(sentence_scores.values())
    average = np.mean(scores)
    sd = np.std(scores)
    res = []
    for key in sentence_scores:
        if sentence_scores[key] > (average + threshold * sd):
            res.append(key)
    # now, find relatively unique words that appear frequently
    # include their sentences in the summary as well
    rare_words_used = set()
    for key, sentence in idf_matrix.items():
        average_idf = sum(sentence.values())/len(sentence)
        for word in sentence:
            if word not in rare_words_used:
              if word_frequency(word, "en") < 5e-5 and sentence[word] < 0.8 * average_idf:
                rare_words_used.add(word)
                res.append(key)
                break
    res.sort()
    return res   
    

def write_summary(sentences, lst):
    """
    Takes in the actual sentences, and a list of indices to be included, 
    creates text summary
    """
    if len(lst) == 0:
        return ""
    summary = ""
    for key in lst:
        summary += sentences[key]
        summary += " "
    return summary

def print_caption(caption):
    """
    Utility function fo printing captions
    """  
    if type(caption) == str:
        print (caption)
        return
    for i in range(len(caption)):
        if len(caption) == 0 or caption[i][-1] not in ".!?":
            caption[i] += "."
    res = ' '.join(caption)
    print (res)

def assert_dictionaries_equal(dict1, dict2):
    """
    Checks if two dictionaries of integers (or dictionaries of 
    dictionaries of integers, or dictionaries of dictionaries of dictionaries...)
    are equal, by utilizing assert statements
    """
    assert len(dict1) == len(dict2)
    for key, value in dict1.items():
        assert key in dict2
        if isinstance(value, dict):
            assert_dictionaries_equal(value, dict2[key])
        else:
          dict2_val = dict2[key]
          assert value - dict2_val < 0.001


def test_1():
    sample_caption = ["hello world world.",
                      "world hello test.",
                      "test test test hello."
                      ]
    tokens, sequence = preprocessing(sample_caption)
    freq_matrix = build_frequency_matrix(tokens, ',.!?')
    tf_matrix = build_tf_matrix(freq_matrix)
    target_tf = {0: {"hello": 0.333333, "world": 0.6666666},
              1: {"world": 0.333333, "hello": 0.333333, "test": 0.333333},
              2: {"hello": 0.25, "test": 0.75}}
    assert_dictionaries_equal(tf_matrix, target_tf)
    appearing_matrix = build_appearing_matrix(freq_matrix)
    idf_matrix = build_idf_matrix(freq_matrix, appearing_matrix)
    target_idf = {0: {"hello": math.log(3/3), "world": math.log(3/2)},
              1: {"world": math.log(3/2), "hello": math.log(3/3), "test": math.log(3/2)},
              2: {"hello": math.log(3/3), "test": math.log(3/2)}}
    assert_dictionaries_equal(idf_matrix, target_idf)
    tf_idf_matrix = build_tf_idf_matrix(tf_matrix, idf_matrix)
    sentence_scores = sentence_scoring(tf_idf_matrix)
    target_sentences = {0: 0.13515503603605478, 
                        1: 0.09010335735736985,
                        2: 0.15204941554056164}
    assert_dictionaries_equal(sentence_scores, target_sentences)
    print ("Test 1 passed")

def test_2():
    sample_caption = ["video image caption.",
                      "caption image caption.",
                      "image image image random."
                      ]
    tokens, sequence = preprocessing(sample_caption)
    freq_matrix = build_frequency_matrix(tokens, ',.!?')
    tf_matrix = build_tf_matrix(freq_matrix)
    target_tf = {0: {"video": 0.333333, "image": 0.333333, "caption": 0.3333333},
              1: {"image": 0.333333, "caption": 0.6666666},
              2: {"image": 0.75, "random": 0.25}}
    assert_dictionaries_equal(tf_matrix, target_tf)
    appearing_matrix = build_appearing_matrix(freq_matrix)
    idf_matrix = build_idf_matrix(freq_matrix, appearing_matrix)
    target_idf = {0: {"video": math.log(3), "image": math.log(3/3), "caption": math.log(3/2)},
              1: {"image": math.log(3/3), "caption": math.log(3/2)},
              2: {"image": math.log(3/3), "random": math.log(3)}}
    assert_dictionaries_equal(idf_matrix, target_idf)
    tf_idf_matrix = build_tf_idf_matrix(tf_matrix, idf_matrix)
    sentence_scores = sentence_scoring(tf_idf_matrix)
    target_sentences = {0: 0.16711971075291934, 
                        1: 0.13515503603605478,
                        2: 0.13732653608351372}
    assert_dictionaries_equal(sentence_scores, target_sentences)
    print ("Test 2 passed")


def test_3():
    sample_caption = ["cs 4701 4700.",
                      "47 47 4701 prac.",   
                      "artificial intelligence prac."
                      ]
    tokens, sequence = preprocessing(sample_caption)
    freq_matrix = build_frequency_matrix(tokens, ',.!?')
    tf_matrix = build_tf_matrix(freq_matrix)
    target_tf = {0: {"cs": 0.333333, "4700": 0.333333, "4701": 0.3333333},
              1: {"4701": 0.25, "47": 0.5, "prac": 0.25},
              2: {"artificial": 0.333333, "intelligence": 0.333333, "prac": 0.3333333}}
    assert_dictionaries_equal(tf_matrix, target_tf)
    appearing_matrix = build_appearing_matrix(freq_matrix)
    idf_matrix = build_idf_matrix(freq_matrix, appearing_matrix)
    target_idf = {0: {"cs": math.log(3), "4700": math.log(3), "4701": math.log(3/2)},
              1: {"4701": math.log(3/2), "47": math.log(3), "prac": math.log(3/2)},
              2: {"artificial": math.log(3), "intelligence": math.log(3), "prac": math.log(3/2)}}
    assert_dictionaries_equal(idf_matrix, target_idf)
    tf_idf_matrix = build_tf_idf_matrix(tf_matrix, idf_matrix)
    sentence_scores = sentence_scoring(tf_idf_matrix)
    target_sentences = {0: 0.28918774282715376, 
                        1: 0.25067956612937903,
                        2: 0.2891877428271537}
    assert_dictionaries_equal(sentence_scores, target_sentences)
    print ("Test 3 passed")


def full_pipeline(caption):
    """
    Takes a caption (either string or list of strings) and returns it summary,
    according to a modified version of the tf-idf algorithm
    """
    tokens, sentences = preprocessing(caption)
    # build relevant matrices
    freq_matrix = build_frequency_matrix(tokens, ",.!?")
    tf_matrix = build_tf_matrix(freq_matrix)
    appearing_matrix = build_appearing_matrix(freq_matrix)
    idf_matrix = build_idf_matrix(freq_matrix, appearing_matrix)
    tf_idf_matrix = build_tf_idf_matrix(tf_matrix, idf_matrix)
    # score sentences according to tf-idf metrics and modifications
    sentence_scores = sentence_scoring(tf_idf_matrix)
    # choose sentences to include, write summary
    to_include = choose_sentences_tfidf_modified(sentence_scores, idf_matrix, 0.25)
    summary = write_summary(sentences, to_include)
    return summary

def software_engineering_test_suite():
    """
    Tests that examine each function individually, ensuring results agree
    with expected values. These tests are meant to ensure correctness of the 
    algorithm, rather than performance metrics. 
    """
    print ("Starting software engineering style tests\n")
    test_1()
    test_2()
    test_3()


def machine_learning_test_suite(many_captions):
    """
    Runs the algorithm on multiple blocks of captions, and prints out the 
    results - used to ensure quality of the algorithm
    """
    for caption in many_captions:
        print ("ORIGINAL TEXT IS: \n")
        print_caption(caption)
        print ()
        # tokens, sentences = preprocessing(caption)
        # freq_matrix = build_frequency_matrix(tokens, ",.!?")
        # tf_matrix = build_tf_matrix(freq_matrix)
        # appearing_matrix = build_appearing_matrix(freq_matrix)
        # idf_matrix = build_idf_matrix(freq_matrix, appearing_matrix)
        # tf_idf_matrix = build_tf_idf_matrix(tf_matrix, idf_matrix)
        # sentence_scores = sentence_scoring(tf_idf_matrix)
        # to_include = choose_sentences_tfidf_modified(sentence_scores, idf_matrix, 0.25)
        # summary = write_summary(sentences, to_include)
        summary = full_pipeline(caption)
        print ("SUMMARIZED TEXT IS: \n")
        print (summary)
        print ()
    
"""
'Machine Learning' style test cases, generated both by hand (captioning images)
and via ChatGPT. Run each of these through modified tf-idf algorithm
"""

captions_robot = [
    "A man in black construction gear with a white hat stands on a platform, grabbing two metal bars",
    "A man in black construction gear with a white hat gear looks confused, and holds a red hammer",
    "A metal, modernized robot stands on a blue mat next to wood stairs",
    "A metal, modernized robot moving between stairs and a plank, connected to a gray table",
    "A metal, modernized robot carries a plank, with stairs in the background",
    "A metal, modernized robot stands next to a plank, which rests between stairs and metal scaffolding",
    "A metal, modernized robot stands with its hands open over a black and yellow bag",
    "A metal, modernized robot holding a black and yellow bag walking on stairs",
    "A metal, modernized robot holding a black and yellow bag standing on a platform",
    "A metal, modernized robot stands on a platform, in front of a large wood block",
    "A metal, modernized robot stands on a platform, with a large woord block resting below on a blue mat",
    "A metal, modernized robot stands on a large wood block on a blue mat, in a twisted body posture",
    "A metal, modernized robot stands on a blue mat, next to a large wood block",
    "A black background with BostonDynamics title and logo",
    "A black background with BostonDynamics title and logo",
]

captions_qr = [
    "man in gray shirt talking into camera, next to QR code with a smiling face",
    "a whiteboard with a 'helpful flow chart' to determine whether one should use a qr code",
    "a man in a gray button down shirt talking into a camera",
    "a woman with a gray shirt and umbrella, next to a green background with a qr code",
    "a futuristic robot, with a mini qr code coming out of their mouth",
    "a person walking through an airport holding a phone with a qr code on the screen",
    "a large mass of people travelling through a corridor",
    "a woman in pink clothing scanning a qr code with her phone",
    "a man sitting at his desk, writing on paper",
    "a colorful qr code being built out of game tiles on an othello-like board",
    "a room with old paintings, a window, and an opening into another room (with curtains)",

]


# below captions are all genderated by GPT as test cases

caption_gpt_gen = "The sun was setting over the city, casting a golden glow on the skyscrapers." \
"People hurried down the sidewalks, eager to get home after a long day at work. Sarah glanced " \
"at her watch and quickened her pace – she didn’t want to be late for dinner. Suddenly, "\
"she heard a loud bark and turned to see a small dog chasing after a frisbee. 'How cute!' she thought, "\
"smiling at the scene. As she approached the corner, she noticed a man struggling to carry a stack "\
"of books. Should she stop to help? Without hesitating, she walked over and offered a hand. 'Thank you so much!' "\
"the man said, looking relieved. They chatted for a moment before going their separate ways. "\
"As Sarah continued on her way, she felt a warm sense of satisfaction. Sometimes, the smallest "\
"gestures made the biggest difference."

next_caption = [
    "The sun was setting over the mountains, casting a warm glow across the valley.",
    "As Emma walked down the path, she could hear the distant sound of a river flowing.",
    "Birds chirped softly in the trees, adding to the peaceful atmosphere.",
    "Suddenly, she noticed a flash of color in the bushes nearby.",
    "Curious, she approached slowly, wondering what it could be.",
    "To her surprise, a small fox peeked out, its bright eyes watching her carefully.",
    "Emma smiled and crouched down, holding out a hand to show she meant no harm.",
    "The fox tilted its head, then cautiously took a step closer.",
    "For a few moments, they shared a quiet connection, both appreciating the calmness of the evening.",
    "Then, as quickly as it appeared, the fox turned and darted back into the forest, leaving Emma with a sense of wonder."
]

another_caption = [
    "It was a chilly morning as Jack set off on his daily jog through the park.",
    "The fog hung low, giving the trees a mysterious, almost magical appearance.",
    "As he ran, he spotted an old, abandoned bench that he hadn’t noticed before.",
    "Something about it seemed intriguing, so he stopped and walked over to take a closer look.",
    "On the bench, he found a small, dusty book with a faded leather cover.",
    "Curious, he picked it up and opened it to the first page.",
    "Inside, there was a note that read, To whoever finds this, may your journey be filled with wonder.",
    "Jack felt a thrill of excitement, wondering who had left the book and why.",
    "He tucked it under his arm and continued his run, feeling as if he’d stumbled into an adventure.",
    "Little did he know, this was just the beginning of an unexpected journey."
]

caption_story = [
    "The small diner at the edge of town buzzed with quiet conversation and the clatter of dishes.",
    "Ella sat in her usual booth, staring out the window as the rain streaked the glass.",
    "Her coffee had grown cold, untouched, as she nervously glanced at the clock on the wall.",
    "She wasn’t sure if he would come — it had been years since their last conversation.",
    "Just as she was about to leave, the door jingled, and a familiar figure stepped inside.",
    "James looked almost the same, though a little older, his hair flecked with gray.",
    "He spotted her immediately and gave a hesitant smile before making his way over.",
    "“I wasn’t sure you’d actually show up,” Ella said, her voice soft but steady.",
    "“I almost didn’t,” James admitted as he slid into the seat across from her.",
    "For a moment, they sat in silence, the sound of rain filling the space between them.",
    "“So,” Ella began, breaking the tension, “how’s life been treating you?”",
    "James chuckled nervously, rubbing the back of his neck. “Complicated. But I guess that’s why I’m here.”",
    "Their conversation flowed slowly at first, but soon the years between them seemed to melt away.",
    "When the rain finally stopped, Ella felt a strange sense of relief, like something heavy had lifted.",
    "As they stepped out of the diner together, she realized that some doors, once reopened, might not close again."
]

possible_image_captions = [
    "Starting the journey: A trailhead sign welcomes hikers at dawn.",
    "The first steps: A narrow dirt path winding through dense pine trees.",
    "Crossing the stream: Sunlight sparkles on the rippling water.",
    "A moment of rest: A hiker sits on a log, tying their boots.",
    "Into the clearing: A wide-open meadow surrounded by towering peaks.",
    "Wildflower bloom: Vibrant reds, yellows, and purples blanket the ground.",
    "Steady ascent: The trail begins to climb, with rocky steps leading higher.",
    "Catching the view: A hiker pauses, looking out over a distant valley.",
    "Lunch break: A spread of snacks on a flat boulder by the trail.",
    "Forest canopy: Sunlight filters through thick green leaves above.",
    "Rocky terrain: A tricky section of the trail with loose stones.",
    "Reaching the summit: A small flag marks the top of the peak.",
    "Panoramic vista: A breathtaking view of mountains stretching to the horizon.",
    "Heading back: Long shadows fall across the trail as the sun dips lower.",
    "Journey's end: The trailhead sign, now glowing in the golden hour light."
]


machine_learning_test_suite([captions_robot, captions_qr, caption_gpt_gen, next_caption,
                             another_caption, caption_story, possible_image_captions])

software_engineering_test_suite()
# tokens, sentences = preprocessing(possible_image_captions)
# freq_matrix = build_frequency_matrix(tokens, ",.!?")
# tf_matrix = build_tf_matrix(freq_matrix)
# appearing_matrix = build_appearing_matrix(freq_matrix)
# idf_matrix = build_idf_matrix(freq_matrix, appearing_matrix)
# tf_idf_matrix = build_tf_idf_matrix(tf_matrix, idf_matrix)
# sentence_scores = sentence_scoring(tf_idf_matrix)
# # to_include = choose_sentences_tfidf(sentence_scores, 1.1)
# to_include = choose_sentences_tfidf_modified(sentence_scores, idf_matrix, 0.25)
# summary = write_summary(sentences, to_include)
# print (summary)


