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

        # Letters: Hindi spoken letter name -> Latin letter
        # e.g. "पी" -> "p", "ई" -> "e", "जी" -> "g"
        letters_graph = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert().optimize()

        # Digit graphs for IP address: spoken -> Devanagari digit
        hindi_digit_graph = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert().optimize()
        hindi_zero_graph = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert().optimize()
        devanagari_digit = hindi_digit_graph | hindi_zero_graph

        # ASCII digit graph for usernames/domains: spoken -> ASCII digit (0-9)
        # Compose: spoken hindi -> devanagari digit -> ascii digit
        hi_to_ascii = pynini.string_map([
            ("०", "0"), ("१", "1"), ("२", "2"), ("३", "3"), ("४", "4"),
            ("५", "5"), ("६", "6"), ("७", "7"), ("८", "8"), ("९", "9"),
        ]).optimize()
        ascii_digit_graph = devanagari_digit @ hi_to_ascii

        # ==================== BUILDING BLOCKS ====================

        # token_sep: consume the space between consecutive spoken tokens
        token_sep = delete_space

        # word_token: one spoken Hindi token -> its Latin/ASCII equivalent
        # Priority: known phonetic word > digit > letter > symbol
        word_token = (
            pynutil.add_weight(server_name_graph | common_words_graph, 0.9)
            | pynutil.add_weight(ascii_digit_graph, 0.95)
            | pynutil.add_weight(letters_graph, 1.0)
            | pynutil.add_weight(symbols_graph, 1.1)
        )

        # spoken_dot: "डॉट" -> "."
        spoken_dot = pynini.cross("डॉट", ".")

        # delete_at: consume "एट" completely.
        # The verbalizer inserts "@" between username and domain fields.
        delete_at = pynutil.delete("एट")

        # ==================== DOMAIN RECONSTRUCTION ====================
        # One or more word tokens (space-separated) form a domain segment.
        # e.g. "जीमेल" -> "gmail", "पी ई टी ई आर" -> "peter"
        tld = domain_graph | file_ext_graph
        domain_segment = word_token + pynini.closure(token_sep + word_token)
        dot_tld = token_sep + spoken_dot + token_sep + tld
        domain_body = domain_segment + pynini.closure(dot_tld, 1)

        # Optional trailing forward-slash: "फॉरवर्ड स्लैश" -> "/"
        optional_slash = pynini.closure(token_sep + pynini.cross("फॉरवर्ड स्लैश", "/"), 0, 1)
        domain_full = domain_body + optional_slash
        domain_tagged = pynutil.insert("domain: \"") + domain_full + pynutil.insert("\"")

        # ==================== USERNAME RECONSTRUCTION ====================
        # Everything before "एट" in an email.
        # e.g. "पी ई टी ई आर शून्य आठ" -> "peter08"
        username_segment = word_token + pynini.closure(token_sep + word_token)
        username_tagged = pynutil.insert("username: \"") + username_segment + pynutil.insert("\"")

        # ==================== PROTOCOL RECONSTRUCTION ====================
        # e.g. "एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश" -> "https"
        protocol_tagged = pynutil.insert("protocol: \"") + protocols_graph + pynutil.insert("\"")

        # ==================== IP ADDRESS RECONSTRUCTION ====================
        # e.g. "एक नौ दो डॉट एक छह आठ डॉट एक डॉट एक" -> "192.168.1.1"
        # IP uses Devanagari digits (matching TN output)
        ip_octet = devanagari_digit + pynini.closure(token_sep + devanagari_digit)
        dot_octet = token_sep + spoken_dot + token_sep + ip_octet
        ip_address = ip_octet + pynini.closure(dot_octet, 3, 3)
        ip_tagged = pynutil.insert("domain: \"") + ip_address + pynutil.insert("\"")

        # ==================== COMBINED GRAPH ====================
        # Email: username + DELETE(एट) + domain
        email_graph = (
            username_tagged
            + token_sep
            + delete_at
            + token_sep
            + pynutil.insert(" ")
            + domain_tagged
        )

        # URL: protocol + domain
        url_graph = protocol_tagged + token_sep + pynutil.insert(" ") + domain_tagged

        # Domain only
        domain_only_graph = domain_tagged

        # Combined with weights (lower = higher priority)
        graph = (
            pynutil.add_weight(url_graph, 1.0)
            | pynutil.add_weight(email_graph, 1.01)
            | pynutil.add_weight(ip_tagged, 1.02)
            | pynutil.add_weight(domain_only_graph, 1.2)
        )

        self.graph = graph.optimize()
        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
