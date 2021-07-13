# Load rdrsegmenter from VnCoreNLP
# from vncorenlp import VnCoreNLP
# rdrsegmenter = VnCoreNLP("/home/local/Zalo/spellcheck_baomoi/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

# Load underthesea library for tokenizing
from underthesea import sent_tokenize, word_tokenize

# import re

# # Input 
# text = "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

# # To perform word (and sentence) segmentation
# sentences = rdrsegmenter.tokenize(text) 
# print("VnCoreNLP result: ", sentences)
# for sentence in sentences:
# 	print(" ".join(sentence))

# # Underthesea Tokenize
# u_sentences = sent_tokenize(text)
# u_words = word_tokenize(text)
# print("Underthesea sentences: ", u_sentences)
# print("Underthesea words: ", u_words)

# # regex tokenize
# re_text = re.sub(r'([?!,.]+)',r' \1 ', text) 
# re_text = re_text.split()
# print("Regex: ", re_text)

import logging
import json
import requests
import time
import re

url = "https://nlp.laban.vn/wiki/spelling_checker_api/"

loggers = [logging.getLogger(name) for name in logging.Logger.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

def get_spcheck(text):
    payload={'text': text, 'app_type': 'baomoi'}
    files=[]
    headers = {
		'Cache-Control': 'no-cache',
		'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return response

start_time = time.time()

text = """
Khi con bò tót gục ngã

Chúa sẽ nói tiếng Italy ư, như trận bán kết này? không ai biết, nhưg bây giờ, niềm hạnh phúc của chiến thắng nghet thở trên chấm luân lưu sau 120 phút khó khăn nhất của họ kể từ ngày khai mạc Eu ro cần được tận hưởng.

Tây Ban Nha đã đánh bại họ trên chấm 11m mùa hè 2008, đi vào chung kết, giành chiến thắng tiếp và mở ra một thời kỳ huy hoàng cho bóng đá xứ sở bò tót. Ai biết được, điều đẹp đẽ ấy có xảy ra với những người Thiên thanh nếu trận thắng ở bán kết này mở ra những điều tuyệt diệu nhất với họ, như Tây Ban Nha từng tận hưởng trong những năm đã qua?

Nhưng đó là câu chuyện của mấy ngày nữa, mấy tháng nữa, mấy năm nữa, cuộc đời đầy những chữ ngờ và đầy những biến đổi không thể biết trước được, nhưng trong cái đêm đẹp đẽ ở bán kết này, như Andrea Bocelli đã hát trong ngày khai mạc khúc “Nessun dorma" (không ai được ngủ), hàng triệu tifosi yêu Italy bằng cả trái tim cũng không ngủ được. Và họ muốn như thế nữa cả sau trận chung kết. Ở cuối của khúc “Nessun dorma” có câu hát, “vincerò”, nghĩa là “tôi sẽ chiến thắng”.

Nhưng chiến thắng ở trận bán kết chỉ có thể giành được theo một cách đau tim và căng thẳng nhất. Những hoài niệm về Euro 2000 khi Italy cũng phòng ngự chặt chẽ để rồi thắng Hà Lan bằng loạt luân lưu trong trận bán kết năm ấy lại hiện về. Cũng vất vả, cũng nhọc nhằn, cũng có những lúc chao đảo, đều là trước những đối thủ đã xông về phía họ như muốn tiêu diệt trong nháy mắt.

Hà Lan đã phải trả giá vì những quả penalty đá hỏng trong 120 phút và những quả luân lưu tiếp tục hỏng trước đôi tay dài của Francesco Toldo. Tây Ban Nha đã phải trả giá vì những cơ hội mà họ có được trong 120 phút thi đấu, trước một Italy không còn là họ ở 5 trận đấu trước, và rồi lại trả giá tiếp bằng 2 quả luân lưu không thành công, trớ trêu thay, từ 2 cái tên chói sáng bậc nhất của họ.

Dani Olmo, người làm khổ hàng thủ Italy trong cả trận, và Alvaro Morata, người gỡ hòa 1-1 ở phút 80, thắp lên cho Tây Ban Nha những hy vọng chiến thắng. Anh không thắng nổi Gigi Donnarumma, và con bò tót Tây Ban Nha, sau khi quần cho người đấu sĩ mệt lử mà anh ta không chết, đã khuỵu chân xuống và không thể đứng dậy nữa.
"""

response = get_spcheck(text)
result = response.json()['result']

for paragraph in result:
	for sentence in sent_tokenize(paragraph['text']):
		if sentence[0].islower():
			mistake_pos = paragraph['text'].find(sentence)
			mistake_text = sentence.split()[0]
			suggest = mistake_text.capitalize()
			mistake = {
				'text': mistake_text,
				'score': 1,
				'start_offset': mistake_pos,
				'suggest': [[suggest, 1]]
			}
			paragraph['mistakes'].append(mistake)
	if len(paragraph['mistakes']) > 0:
		for mistake in paragraph['mistakes']:
			mistake_pos = mistake['start_offset']
			mistake_len = len(mistake['text'])

	print(paragraph)

logging.info("Total time: --- %s seconds ---" % (time.time() - start_time))

print("******************")

print("Transforming output response")

def split_span(s):
    for match in re.finditer(r"\S+", s):
        span = match.span()
        yield match.group(0), span[0], span[1] - 1

all_tokens = []
for paragraph in result:
	para_tokens = split_span(paragraph['text'])

	mistake_positions = [mistake['start_offset'] for mistake in paragraph['mistakes']]
	
	for token in para_tokens:
		if token[1] not in mistake_positions:
			all_tokens.append(token[0])
		else:
			typo = dict()
			typo[token[0]] = []
			all_tokens.append(typo)

print(all_tokens)