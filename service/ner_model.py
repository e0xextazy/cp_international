import re

from natasha import (
    AddrExtractor,
    DatesExtractor,
    Doc,
    MoneyExtractor,
    MorphVocab,
    NamesExtractor,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    NewsSyntaxParser,
    Segmenter,
)


class NER:
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        self.names_extractor = NamesExtractor(self.morph_vocab)
        self.addr_extractor = AddrExtractor(self.morph_vocab)
        self.dates_extractor = DatesExtractor(self.morph_vocab)
        self.money_extractor = MoneyExtractor(self.morph_vocab)
        self.phone_pattern = re.compile(r"\+?\d{1,4}?[-\s]??[-\s]?\d{1,4}[-\s]?\d{1,4}[-\s]?\d{1,9}")

    def _extract_phone_numbers(self, text):
        phone_numbers = self.phone_pattern.findall(text)

        return phone_numbers

    def get_tags(self, text):
        doc = Doc(text)

        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)

        for span in doc.spans:
            span.normalize(self.morph_vocab)
            span.extract_fact(self.names_extractor)

        # LOC, ORG, PER
        dt = {"LOC": [], "ORG": [], "PER": []}

        for span in doc.spans:
            dt[span.type].append(span.normal)

        for key in dt:
            dt[key] = list(set(dt[key]))

        # PHONE, MONEY, ADDRES, DATES
        dt["PHONE"] = self._extract_phone_numbers(text)

        dt["MONEY"] = [f"{_.fact.__dict__['amount']} {_.fact.__dict__['currency']}" for _ in self.money_extractor(text)]

        dt["ADDRES"] = [f"{_.fact.__dict__['type']} {_.fact.__dict__['value']}" for _ in self.addr_extractor(text)]

        date = [_.fact.__dict__ for _ in self.dates_extractor(text)]
        new_date = []
        for d in date:
            tmp = []
            for key in ["day", "month", "year"]:
                if d[key] is not None:
                    tmp.append(str(d[key]))
            new_date.append(".".join(tmp))
        dt["DATE"] = new_date

        return dt
