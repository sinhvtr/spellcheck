import random
import numpy as np
import json
import codecs
import logging

s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'

s3 = u'ẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẾếỀềỂểỄễỆệỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỨứỪừỬửỮữỰự'
s2 = u'ÂâÂâÂâÂâÂâĂăĂăĂăĂăĂăÊêÊêÊêÊêÊêÔôÔôÔôÔôÔôƠơƠơƠơƠơƠơƯưƯưƯưƯưƯư'
alphabet = 'abcdefghijklmnopqrstuvwxyz'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(format(obj, '.4f'))

        return json.JSONEncoder.default(self, obj)


def remove_accents(input_str):
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s


def generate_typos(token,
                   no_typo_prob=0.7,
                   asccents_prob=0.5,
                   lowercase_prob=0.5,
                   swap_char_prob=0.1,
                   add_chars_prob=0.1,
                   remove_chars_prob=0.1
                   ):
    
    if random.random() < no_typo_prob:
        # print("No typo prob")
        return token
    if random.random() < asccents_prob:
        if random.random() < 0.5:
            # print("asccents_prob < 0.5")
            token = remove_accents(token)
            # print(token)
        else:
            # print("asccents_prob >= 0.5")
            new_chars = []
            for cc in token:
                if cc in s3 and random.random() < 0.7:
                    cc = s2[s3.index(cc)]
                if cc in s1 and random.random() < 0.5:
                    cc = s0[s1.index(cc)]
                new_chars.append(cc)
            token = "".join(new_chars)
            # print(token)
    if random.random() < lowercase_prob:
        # print("lowercase_prob")
        token = token.lower()
        # print(token)
    if random.random() < swap_char_prob:
        # print("swap_char_prob")
        chars = list(token)
        n_swap = min(len(chars), np.random.poisson(0.5) + 1)
        index = np.random.choice(
            np.arange(len(chars)), size=n_swap, replace=False)
        swap_index = index[np.random.permutation(index.shape[0])]
        swap_dict = {ii: jj for ii, jj in zip(index, swap_index)}
        chars = [chars[ii] if ii not in index else chars[swap_dict[ii]]
                 for ii in range(len(chars))]
        token = "".join(chars)
        # print(token)
    if random.random() < remove_chars_prob:
        # print("remove_chars_prob")
        n_remove = min(len(token), np.random.poisson(0.005) + 1)
        for _ in range(n_remove):
            pos = np.random.choice(np.arange(len(token)), size=1)[0]
            token = token[:pos] + token[pos+1:]
        # print(token)
    if random.random() < add_chars_prob:
        # print("add_chars_prob")
        n_add = min(len(token), np.random.poisson(0.05) + 1)
        adding_chars = np.random.choice(
            list(alphabet), size=n_add, replace=True)
        for cc in adding_chars:
            pos = np.random.choice(np.arange(len(token)), size=1)[0]
            token = "".join([token[:pos], cc, token[pos:]])
        # print(token)
    print(token)
    return token


def generate_typos_for_text(texts):
    new_texts = []
    if len(texts) > 0:
        for s in texts:
            new_s = " ".join([generate_typos(t) for t in s.split()])
            new_texts.append(new_s)
    return new_texts


def load_json(path):
    with codecs.open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(o, saved):
    import unicodedata
    data = unicodedata.normalize('NFC', json.dumps(o, indent=4, ensure_ascii=False))
    with codecs.open(saved, mode="w", encoding="utf-8") as f:
        f.write(data)

def load_lines(path):
    with codecs.open(path, encoding="utf-8") as f:
        return [l.strip() for l in f.readlines()]


def load_resources(default_folder="resource"):
    def _load_teencode(path):
        lines = load_lines(path)
        teencode_dict = {}
        for l in lines:
            if l.strip() != "":
                if "=" in l:
                    items = l.split("=")
                    teencode_dict[items[0].lower()] = items[1]
        return teencode_dict

    def _load_abbr_dict(path):
        lines = load_lines(path)
        abbr_dict = {}
        for l in lines:
            if not l.startswith("#") and l.strip() != "":
                items = l.split(",")
                if len(items) > 0:
                    abbr_dict[items[0].lower()] = items[1]
        return abbr_dict

    teencode_dict = _load_teencode(default_folder + "/teencodeReplace.txt")
    logging.info("teencode_dict loaded ... %d", len(teencode_dict.keys()))

    abbr_dict = _load_abbr_dict(default_folder + "/zad_abb.txt")
    logging.info("abbr_dict loaded ... %d", len(abbr_dict.keys()))

    return {
        "teencode_dict": teencode_dict,
        "abbr_dict": abbr_dict
    }


SSS = set("0123456789`~.,;:!#$^&*()_+-{}[]<>?%\"' @aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ")
def text_normalize(t):
    t = t.replace("…", "...")

    inds = []
    for i in range(len(t) - 1):
        if t[i] in SSS and t[i + 1] not in SSS:
            inds.append(i)

    for i in range(len(t) - 1):
        if t[i] not in SSS and t[i + 1] in SSS:
            inds.append(i)

    inds.sort(reverse=True)

    for i in inds:
        t = t[:i + 1] + " " + t[i + 1:]

    return " ".join(t.split())

if __name__ == '__main__':
    text = ["Tuy quán không quá rộng nhưng không gian giữa các bàn khá thoải mái. Thêm vào đó bàn ghế rộng với chiều cao phù hợp để thoải mái kê laptop làm việc cũng là điểm cộng khiến nhiều freelance lựa chọn quán làm nơi ngồi làm việc."]
    print(text[0])
    typo_text = generate_typos_for_text(text)
    print(typo_text)
    # t = text_normalize("thôi … nhe")
    # s = "thôi … nhe ."
    # print(s[9:])
    # print(t)
