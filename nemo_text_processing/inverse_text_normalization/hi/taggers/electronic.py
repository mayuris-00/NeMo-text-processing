# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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
    NEMO_SIGMA,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic addresses (ITN direction).
    Recognizes Hindi spoken forms and tags them into structured tokens.

    Examples:
        कुमार एट जीमेल डॉट कॉम   -> electronic { username: "kumar" domain: "gmail.com" }
        एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश गूगल डॉट कॉम
                                   -> electronic { protocol: "https" domain: "google.com" }
        गूगल डॉट कॉम              -> electronic { domain: "google.com" }
        एक नौ दो डॉट एक छह आठ डॉट एक डॉट एक  -> electronic { domain: "192.168.1.1" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transductions are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        # ==================== LOAD & INVERT DATA FILES ====================
        # Each file is inverted: Hindi spoken form -> original written form

        symbols_graph = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).invert().optimize()
        domain_graph = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert().optimize()
        server_name_graph = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert().optimize()
        common_words_graph = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert().optimize()
        protocols_graph = pynini.string_file(get_abs_path("data/electronic/protocols.tsv")).invert().optimize()
        file_ext_graph = pynini.string_file(get_abs_path("data/electronic/file_extensions.tsv")).invert().optimize()
        subscript_digit_graph = (
            pynini.string_file(get_abs_path("data/electronic/subscript_digit.tsv")).invert().optimize()
        )

        # Digit graphs (for IP address reconstruction digit-by-digit)
        hindi_digit_graph = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert().optimize()
        hindi_zero_graph = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert().optimize()
        digit_graph = hindi_digit_graph | hindi_zero_graph

        # Letters: Hindi letter name -> Latin letter (e.g. ए -> a, बी -> b)
        # Uses the same letters.tsv that TN verbalizer uses (inverted)
        latin_letters_graph = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()  # placeholder structure
        )
        # We build letter recognition from the symbols + common word graphs combined
        # A spoken "word segment" is either a known server/common word or letter-by-letter
        phonetic_word = server_name_graph | common_words_graph

        # ==================== BUILDING BLOCKS ====================

        # dot: डॉट -> "."
        spoken_dot = pynini.cross("डॉट", ".")
        # @ : एट -> "@"  (handled as a separator, not in symbols loop)
        spoken_at = pynini.cross("एट", "@")

        # A single spoken token: either a known phonetic word or a symbol
        # Spoken char: maps one Hindi spoken token to one Latin char/symbol
        spoken_char = pynutil.add_weight(phonetic_word, 0.9) | pynutil.add_weight(symbols_graph, 1.0)

        # spoken_char + delete_space allows consuming space-separated tokens
        # and concatenating their outputs without spaces
        spoken_char_seq = pynini.closure(spoken_char + delete_space, 1)

        # ==================== DOMAIN RECONSTRUCTION ====================
        # domain segment: e.g. "जीमेल" -> "gmail"  or letter-by-letter
        domain_segment = pynini.closure(spoken_char + delete_space, 0) + spoken_char

        # TLD: e.g. "कॉम" -> "com"
        tld = domain_graph | file_ext_graph

        # dot + TLD: "डॉट कॉम" -> ".com"
        dot_tld = spoken_dot + delete_space + tld

        # Full domain: "जीमेल डॉट कॉम" -> "gmail.com"
        # One or more segments separated by spoken dots
        domain_body = domain_segment + pynini.closure(dot_tld, 1)

        # Optional trailing slash
        optional_slash = pynini.closure(delete_space + pynini.cross("फॉरवर्ड स्लैश", "/"), 0, 1)

        domain_full = domain_body + optional_slash

        # Tagged domain field
        domain_tagged = pynutil.insert("domain: \"") + domain_full + pynutil.insert("\"")

        # ==================== USERNAME RECONSTRUCTION ====================
        # Username: everything before "एट" in an email
        # e.g. "कुमार" -> "kumar",  "जॉन डॉट स्मिथ" -> "john.smith"
        username_char = pynutil.add_weight(phonetic_word, 0.9) | pynutil.add_weight(symbols_graph, 1.0)
        username_segment = pynini.closure(username_char + delete_space, 0) + username_char
        username_tagged = pynutil.insert("username: \"") + username_segment + pynutil.insert("\"")

        # ==================== PROTOCOL RECONSTRUCTION ====================
        # e.g. "एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश" -> "https://"
        # protocols.tsv maps: https -> एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश
        # After invert: spoken -> "https" / "http" / "www" etc.
        protocol_tagged = pynutil.insert("protocol: \"") + protocols_graph + pynutil.insert("\"")

        # ==================== IP ADDRESS RECONSTRUCTION ====================
        # e.g. "एक नौ दो डॉट एक छह आठ डॉट एक डॉट एक" -> "192.168.1.1"
        # Digit-by-digit with spoken dots as separators
        ip_digit = digit_graph
        ip_octet = pynini.closure(ip_digit + delete_space, 0) + ip_digit
        dot_octet = delete_space + spoken_dot + delete_space + ip_octet
        ip_address = ip_octet + pynini.closure(dot_octet, 3, 3)

        ip_tagged = pynutil.insert("domain: \"") + ip_address + pynutil.insert("\"")

        # ==================== FILE WITH EXTENSION ====================
        # e.g. "रिपोर्ट डॉट पी डी एफ" -> "report.pdf"
        filename_stem = pynini.closure(spoken_char + delete_space, 0) + spoken_char
        file_ext_spoken = file_ext_graph
        file_with_ext = filename_stem + delete_space + spoken_dot + delete_space + file_ext_spoken
        file_tagged = pynutil.insert("domain: \"") + file_with_ext + pynutil.insert("\"")

        # ==================== COMBINED GRAPH ====================
        # Email: username + एट + domain
        email_graph = (
            username_tagged
            + delete_space
            + spoken_at
            + delete_space
            + pynutil.insert(" ")
            + domain_tagged
        )

        # URL with protocol: protocol + space + domain
        url_graph = protocol_tagged + delete_space + pynutil.insert(" ") + domain_tagged

        # Domain only (no protocol, no username)
        domain_only_graph = domain_tagged

        # Combined with weights (lower = higher priority)
        graph = (
            pynutil.add_weight(url_graph, 1.0)
            | pynutil.add_weight(email_graph, 1.01)
            | pynutil.add_weight(ip_tagged, 1.02)
            | pynutil.add_weight(file_tagged, 1.1)
            | pynutil.add_weight(domain_only_graph, 1.2)
        )

        self.graph = graph.optimize()
        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
