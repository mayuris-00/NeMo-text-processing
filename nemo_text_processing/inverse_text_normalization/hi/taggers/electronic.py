# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    ITN FST for classifying electronic expressions.
    Covers email, IP, URL (https/http), www-domains, plain domains.
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        # ── digits ────────────────────────────────────────────────────────────
        digit_words = pynini.string_map([
            ("एक","1"),("दो","2"),("तीन","3"),("चार","4"),("पाँच","5"),
            ("छह","6"),("सात","7"),("आठ","8"),("नौ","9"),("शून्य","0"),
        ])
        digit_glyphs = pynini.string_map([
            ("१","1"),("२","2"),("३","3"),("४","4"),("५","5"),
            ("६","6"),("७","7"),("८","8"),("९","9"),("०","0"),
        ])
        word_seq  = digit_words  + pynini.closure(delete_space + digit_words, 0)
        glyph_seq = digit_glyphs + pynini.closure(digit_glyphs, 0)
        digit_seq = pynutil.add_weight(glyph_seq, 0.8) | pynutil.add_weight(word_seq, 0.9)

        # ── TSV maps ──────────────────────────────────────────────────────────
        letter_map = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert()
        domain_map = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert()
        server_map = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert()
        common_map = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert()

        # ASCII TLD passthrough
        domain_map = domain_map | pynini.string_map([
            ("COM","com"),("NET","net"),("ORG","org"),
            ("BIZ","biz"),("INFO","info"),("IN","in"),
        ])

        # ── host-label prefixes (kept separate to avoid token conflicts) ──────
        host_prefix_map = pynini.string_map([
            ("एस आर वी",  "srv"),
            ("डी बी",      "db"),
            ("एल टी",      "lt"),
            ("वेब",        "web"),
            ("लैपटॉप",     "laptop"),
            ("डेस्कटॉप",  "desktop"),
            ("ई मेल",      "email"),
        ])

        # ── symbols ───────────────────────────────────────────────────────────
        dot = (
            delete_space
            + (pynutil.delete("डॉट") | pynutil.delete("DOT"))
            + delete_space
            + pynutil.insert(".")
        )
        slash      = delete_space + pynutil.delete("फॉरवर्ड स्लैश") + pynutil.insert("/")
        hyphen     = delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन")) + delete_space + pynutil.insert("-")
        underscore = delete_space + pynutil.delete("अंडर स्कोर") + delete_space + pynutil.insert("_")
        at_sign    = delete_space + pynutil.delete("एट") + delete_space

        # ── token primitives ──────────────────────────────────────────────────
        single_token = (
            pynutil.add_weight(server_map, 0.9)
            | pynutil.add_weight(common_map, 0.95)
            | pynutil.add_weight(letter_map, 1.0)
        )
        token_seq = single_token + pynini.closure(delete_space + single_token, 0)

        # ── domain label ──────────────────────────────────────────────────────
        first_label = (
            pynutil.add_weight(host_prefix_map,                         0.80)
            | pynutil.add_weight(digit_seq + delete_space + letter_map, 0.85)
            | pynutil.add_weight(digit_seq,                             0.87)
            | pynutil.add_weight(token_seq,                             1.0)
        )
        domain_body  = first_label + pynini.closure(hyphen + (digit_seq | token_seq), 0)
        compound_tld = domain_map + pynini.closure(dot + domain_map, 0, 2)
        full_domain  = pynini.closure(domain_body + dot, 0, 4) + domain_body + dot + compound_tld

        # ── EMAIL ─────────────────────────────────────────────────────────────
        uname_atom = (
            pynutil.add_weight(digit_words,   0.85)
            | pynutil.add_weight(digit_glyphs, 0.85)
            | pynutil.add_weight(server_map,   0.9)
            | pynutil.add_weight(common_map,   0.95)
            | pynutil.add_weight(letter_map,   1.0)
        )
        uname_sep = (
            (delete_space + pynutil.delete("डॉट") + delete_space + pynutil.insert("."))
            | (delete_space + (pynutil.delete("हाइफ़न") | pynutil.delete("हाइफन")) + delete_space + pynutil.insert("-"))
            | (delete_space + pynutil.delete("अंडर स्कोर") + delete_space + pynutil.insert("_"))
        )
        username = uname_atom + pynini.closure(
            (uname_sep + uname_atom) | (delete_space + uname_atom), 0
        )
        email_fst = (
            pynutil.insert("username: \"") + username + pynutil.insert("\"")
            + at_sign
            + pynutil.insert("domain: \"") + domain_body + dot + compound_tld + pynutil.insert("\"")
        )

        # ── IP ────────────────────────────────────────────────────────────────
        ip_fst = (
            pynutil.insert("ip: \"")
            + digit_seq + dot + digit_seq + dot + digit_seq + dot + digit_seq
            + pynutil.insert("\"")
        )

        # ── URL path segment ──────────────────────────────────────────────────
        # handles: /613  /domain-analysis  /adobe.com  /play.google.com
        inline_domain_seg = (
            pynini.closure(token_seq + dot, 0, 2)
            + token_seq
            + dot + domain_map
            + pynini.closure(dot + domain_map, 0, 1)
        )
        path_atom = (
            pynutil.add_weight(digit_seq + delete_space + letter_map + delete_space + digit_seq, 0.8)
            | pynutil.add_weight(digit_seq, 0.9)
            | pynutil.add_weight(token_seq, 1.0)
        )
        path_segment = (
            path_atom
            + pynini.closure(hyphen + (digit_seq | token_seq), 0)
            + pynini.closure(underscore + token_seq, 0)
            + pynini.closure(dot + token_seq, 0, 1)
        )
        slash_with_word = slash + delete_space + (
            pynutil.add_weight(
                pynutil.insert(".") + pynutil.delete("डॉट") + delete_space + token_seq,
                0.9
            )
            | pynutil.add_weight(inline_domain_seg, 0.95)
            | pynutil.add_weight(path_segment,      1.0)
        )

        # ── protocols ─────────────────────────────────────────────────────────
        https_prefix = (
            pynutil.delete("एच") + delete_space + pynutil.delete("टी") + delete_space
            + pynutil.delete("टी") + delete_space + pynutil.delete("पी") + delete_space
            + pynutil.delete("एस") + delete_space + pynutil.delete("कोलन") + delete_space
            + pynutil.delete("फॉरवर्ड स्लैश") + delete_space + pynutil.delete("फॉरवर्ड स्लैश")
            + pynutil.insert("https://")
        )
        http_prefix = (
            pynutil.delete("एच") + delete_space + pynutil.delete("टी") + delete_space
            + pynutil.delete("टी") + delete_space + pynutil.delete("पी") + delete_space
            + pynutil.delete("कोलन") + delete_space
            + pynutil.delete("फॉरवर्ड स्लैश") + delete_space + pynutil.delete("फॉरवर्ड स्लैश")
            + pynutil.insert("http://")
        )
        protocol = pynutil.add_weight(https_prefix, 1.0) | pynutil.add_weight(http_prefix, 1.01)

        # ── www prefix ────────────────────────────────────────────────────────
        www = (
            pynutil.delete("डब्ल्यू") + delete_space
            + pynutil.delete("डब्ल्यू") + delete_space
            + pynutil.delete("डब्ल्यू") + pynutil.insert("www")
        )

        # ── combined domain + path ─────────────────────────────────────────────
        domain_and_path = (
            full_domain
            + pynini.closure(slash_with_word, 0)
            + pynini.closure(slash, 0, 1)
        )

        url_fst   = pynutil.insert("domain: \"") + protocol + delete_space + pynini.closure(www + dot, 0, 1) + domain_and_path + pynutil.insert("\"")
        www_fst   = pynutil.insert("domain: \"") + www + dot + domain_and_path + pynutil.insert("\"")
        plain_fst = pynutil.insert("domain: \"") + domain_and_path + pynutil.insert("\"")

        graph = (
            pynutil.add_weight(ip_fst,     1.0)
            | pynutil.add_weight(email_fst, 1.05)
            | pynutil.add_weight(url_fst,   1.1)
            | pynutil.add_weight(www_fst,   1.2)
            | pynutil.add_weight(plain_fst, 1.3)
        )
        self.fst = self.add_tokens(graph).optimize()