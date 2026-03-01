#!/usr/bin/env python3
# extract_triples_mrebel_constant.py

from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ========= EDIT THESE CONSTANTS =========
TEXTS = [
    "James Dean\n\nJames Byron Dean (February 8, 1931\u00a0\u2013 September 30, 1955) was an American actor. He is remembered as a cultural icon\\ of teenage disillusionment and social estrangement, as expressed in the title of his most celebrated film, \"Rebel Without a Cause\\\" (1955), in which he starred as troubled teenager Jim Stark. The other two roles that defined his stardom were loner\\ Cal Trask in \"East of Eden\\\" (1955) and surly ranch hand Jett Rink in \"Giant\\\" (1956). Dean's premature death in a car crash\\ cemented his legendary status. He became the first actor to receive a posthumous Academy Award\\ nomination for Best Actor\\, and remains the only actor to have had two posthumous acting nominations. In 1999, the American Film Institute\\ ranked him the 18th best male movie star of Golden Age Hollywood in AFI's 100 Years...100 Stars\\ list. James Dean was born February 8, 1931, at the Seven Gables apartment on the corner of 4th Street and McClure Street in Marion, Indiana\\, the only child of Winton Dean (January 17, 1908 \u2013 February 21, 1995) and Mildred Marie Wilson (September 15, 1910 \u2013 July 14, 1940). His parents were of mostly English ancestry, with smaller amounts of German, Irish, Scottish, and Welsh. Six years after his father had left farming to become a dental technician, Dean and his family moved to Santa Monica, California\\. He was enrolled at Brentwood Public School in the Brentwood\\ neighborhood of Los Angeles\\, California\\, but transferred soon afterward to the McKinley Elementary School. The family spent several years there, and by all accounts, Dean was very close to his mother. According to Michael DeAngelis, she was \"the only person capable of understanding him\". In 1938, she was suddenly struck with acute stomach pain and quickly began to lose weight. She died of uterine cancer\\ when Dean was nine years old. Unable to care for his son, Dean's father sent him to live with Dean's aunt Ortense and her husband, Marcus Winslow, on a farm in Fairmount, Indiana\\, where he was raised in their Quaker\\ household. Dean's father served in World War II\\ and later remarried. In his adolescence, Dean sought the counsel and friendship of a local Methodist\\ pastor, the Rev. James DeWeerd, who seems to have had a formative influence upon Dean, especially upon his future interests in bullfighting\\, car racing, and theater. According to Billy J. Harbin, Dean had \"an intimate relationship with his pastor, which began in his senior year of high school and endured for many years\". Their alleged sexual relationship was suggested in the 1994 book \"Boulevard of Broken Dreams: The Life, Times, and Legend of James Dean\" by Paul Alexander\\. In 2011, it was reported that Dean once confided in Elizabeth Taylor\\ that he was sexually abused\\ by a minister approximately two years after his mother's death. Other reports on Dean's life also suggest that he was either sexually abused by DeWeerd as a child or had a sexual relationship with him as a late teenager. Dean's overall performance in school was exceptional and he was a popular student. He played on the baseball\\ and varsity basketball\\ teams, studied drama, and competed in public speaking\\ through the Indiana High School Forensic Association. After graduating from Fairmount High School in May 1949, Dean moved back to California with his dog, Max, to live with his father and stepmother. He enrolled in Santa Monica College\\ (SMC) and majored in pre-law\\. He transferred to UCLA\\ for one semester and changed his major to drama, which resulted in estrangement from his father. He pledged the Sigma Nu fraternity but was never initiated. While at UCLA, Dean was picked from a group of 350 actors to portray Malcolm in \"Macbeth\\\". At that time, he also began acting in James Whitmore\\'s workshop. In January 1951, he dropped out of UCLA to pursue a full-time career as an actor. Dean's first television appearance was in a Pepsi Cola\\ commercial. He quit college to act full-time and was cast in his first speaking part, as John the Beloved Disciple\\, in \"Hill Number One\", an Easter television special dramatizing the Resurrection of Jesus\\. Dean worked at the widely filmed Iverson Movie Ranch\\ in the Chatsworth\\ area of Los Angeles during production of the program, for which a replica of the tomb of Jesus was built on location at the ranch. Dean subsequently obtained three walk-on roles in movies: as a soldier in \"Fixed Bayonets!\\\", a boxing cornerman in \"Sailor Beware\\\", and a youth in \"Has Anybody Seen My Gal?\\\"\n\nWhile struggling to get jobs in Hollywood\\, Dean also worked as a parking lot attendant at CBS Studios\\, during which time he met Rogers Brackett, a radio director for an advertising agency, who offered him professional help and guidance in his chosen career, as well as a place to stay. In July 1951, Dean appeared on \"Alias Jane Doe\\\", which was produced by Brackett. In October 1951, following the encouragement of actor James Whitmore\\ and the advice of his mentor Rogers Brackett, Dean moved to New York City. There, he worked as a stunt tester for the game show\\ \"Beat the Clock\\\", but was subsequently fired for allegedly performing the tasks too quickly. He also appeared in episodes of several CBS television series \"The Web\", \"Studio One\\\", and \"Lux Video Theatre\\\", before gaining admission to the Actors Studio\\ to study method acting\\ under Lee Strasberg\\. Proud of this accomplishment, Dean referred to the Actors Studio in a 1952 letter to his family as \"the greatest school of the theater. It houses great people like Marlon Brando\\, Julie Harris\\, Arthur Kennedy\\, Mildred Dunnock\\, Eli Wallach\\... Very few get into it ... It is the best thing that can happen to an actor." # add more strings if you want to batch:
    # "The Red Hot Chili Peppers were formed in Los Angeles by Kiedis, Flea, Hillel Slovak and Jack Irons."
]
MODEL_ID = "Babelscape/mrebel-large"
USE_FP16 = True  # set False if your GPU doesn't support fp16
GEN_KWARGS = dict(
    max_length=1056,
    length_penalty=0.0,
    num_beams=3,
    num_return_sequences=3,
)
# =======================================


def parse_mrebel_output(text: str) -> List[Dict[str, str]]:
    """
    Parse mREBEL decoded text into a list of typed triplets:
    [{'head': ..., 'head_type': ..., 'type': ..., 'tail': ..., 'tail_type': ...}, ...]
    """
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, obj, obj_type, subj_type = '', '', '', '', ''
    for token in (
        text.replace("<s>", "")
            .replace("<pad>", "")
            .replace("</s>", "")
            .replace("tp_XX", "")
            .replace("__en__", "")
            .split()
    ):
        if token in ("<triplet>", "<relation>"):
            current = 't'
            if relation:
                triplets.append({
                    'head': subject.strip(),
                    'head_type': subj_type,
                    'type': relation.strip(),
                    'tail': obj.strip(),
                    'tail_type': obj_type
                })
                relation = ''
            subject = ''
        elif token.startswith("<") and token.endswith(">"):
            if current in ('t', 'o'):
                current = 's'
                if relation:
                    triplets.append({
                        'head': subject.strip(),
                        'head_type': subj_type,
                        'type': relation.strip(),
                        'tail': obj.strip(),
                        'tail_type': obj_type
                    })
                obj = ''
                subj_type = token[1:-1]
            else:
                current = 'o'
                obj_type = token[1:-1]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                obj += ' ' + token
            elif current == 'o':
                relation += ' ' + token

    if subject and relation and obj and obj_type and subj_type:
        triplets.append({
            'head': subject.strip(),
            'head_type': subj_type,
            'type': relation.strip(),
            'tail': obj.strip(),
            'tail_type': obj_type
        })
    seen = set()
    uniq = []
    for t in triplets:
        key = (t['head'], t['head_type'], t['type'], t['tail'], t['tail_type'])
        if key not in seen:
            uniq.append(t)
            seen.add(key)
    return uniq


class MRebelExtractor:
    def __init__(self, model_id: str = MODEL_ID, fp16: bool = USE_FP16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, src_lang="en_XX", tgt_lang="tp_XX"
        )
        dtype = torch.float16 if (fp16 and self.device == "cuda") else None
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(self.device)
        self.decoder_start_id = self.tokenizer.convert_tokens_to_ids("tp_XX")

    @torch.inference_mode()
    def extract(self, texts: List[str]) -> List[List[Dict[str, str]]]:
        enc = self.tokenizer(
            texts, max_length=256, padding=True, truncation=True, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_start_token_id=self.decoder_start_id,
            **GEN_KWARGS,
        )
        # group per input if returning multiple sequences
        per_input = (
            out.view(len(texts), -1, out.size(-1))
            if GEN_KWARGS.get("num_return_sequences", 1) > 1
            else out.unsqueeze(1)
        )
        results = []
        for i in range(len(texts)):
            decoded = self.tokenizer.batch_decode(per_input[i], skip_special_tokens=False)
            merged, seen = [], set()
            for s in decoded:
                for t in parse_mrebel_output(s):
                    key = (t['head'], t['head_type'], t['type'], t['tail'], t['tail_type'])
                    if key not in seen:
                        merged.append(t)
                        seen.add(key)
            results.append(merged)
        return results


if __name__ == "__main__":
    if not TEXTS:
        raise SystemExit("Please set TEXTS at the top of the file.")
    extractor = MRebelExtractor()
    batches = extractor.extract(TEXTS)
    for i, triples in enumerate(batches):
        print(f"# Input {i}: {TEXTS[i]}")
        for t in triples:
            print(f"- ({t['head']} : {t['head_type']})  --{t['type']}-->  ({t['tail']} : {t['tail_type']})")
