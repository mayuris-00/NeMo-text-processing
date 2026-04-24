# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    ITN Electronic classifier. Handles:
      - Emails:              अधीश एट जीमेल डॉट कॉम -> electronic { username: "adhish" domain: "gmail.com" }
      - URLs:                गूगल डॉट कॉम           -> electronic { domain: "google.com" }
      - Protocols:           एच टी टी पी एस ...     -> electronic { protocol: "https" domain: "..." }
      - IP addresses:        एक नौ दो डॉट ...       -> electronic { domain: "192.168.1.1" }
      - File paths:          सी कोलन बैकवर्ड ...    -> electronic { path: "C:\\..." }
      - Chemical (lookup):   ग्लूकोज                -> electronic { domain: "C6H12O6" }
      - Chemical (subscript):एच टू एस ओ फ़ोर       -> electronic { domain: "H₂SO₄" }
      - Alphanumeric codes:  ए जे एन एफ ... के      -> electronic { domain: "AJNFC3837K" }
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        # ==================== DATA FILES ====================
        symbols_graph      = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).invert().optimize()
        domain_graph       = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert().optimize()
        server_name_graph  = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert().optimize()
        common_words_graph = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert().optimize()
        protocols_graph    = pynini.string_file(get_abs_path("data/electronic/protocols.tsv")).invert().optimize()
        file_ext_graph     = pynini.string_file(get_abs_path("data/electronic/file_extensions.tsv")).invert().optimize()
        proper_names_graph = pynini.string_file(get_abs_path("data/electronic/proper_names.tsv")).invert().optimize()
        letters_graph      = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert().optimize()
        subscript_spoken   = pynini.string_file(get_abs_path("data/electronic/subscript_spoken.tsv")).invert().optimize()
        chemical_lookup    = pynini.string_file(get_abs_path("data/electronic/chemical_formulas.tsv")).invert().optimize()

        # Digit graphs
        hindi_digit  = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert().optimize()
        hindi_zero   = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert().optimize()
        devanagari_d = hindi_digit | hindi_zero
        teens_graph  = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert().optimize()

        hi_to_ascii = pynini.string_map([
            ("०","0"),("१","1"),("२","2"),("३","3"),("४","4"),
            ("५","5"),("६","6"),("७","7"),("८","8"),("९","9"),
        ]).optimize()
        ascii_digit     = devanagari_d @ hi_to_ascii
        ascii_two_digit = (teens_graph @ pynini.closure(hi_to_ascii, 1)).optimize()

        # Uppercase letter graph
        lower_to_upper = pynini.string_map([
            ("a","A"),("b","B"),("c","C"),("d","D"),("e","E"),("f","F"),
            ("g","G"),("h","H"),("i","I"),("j","J"),("k","K"),("l","L"),
            ("m","M"),("n","N"),("o","O"),("p","P"),("q","Q"),("r","R"),
            ("s","S"),("t","T"),("u","U"),("v","V"),("w","W"),("x","X"),
            ("y","Y"),("z","Z"),
        ]).optimize()
        upper_letters = letters_graph @ lower_to_upper

        token_sep   = delete_space
        spoken_dot  = pynini.cross("डॉट", ".")
        delete_at   = pynutil.delete("एट")
        tld         = domain_graph | file_ext_graph

        # ==================== WORD TOKEN (for domain/username/path) ====================
        # NO subscript here — avoids bleeding into alphanumeric context
        word_token = (
            pynutil.add_weight(proper_names_graph | server_name_graph | common_words_graph, 0.8)
            | pynutil.add_weight(ascii_two_digit, 0.92)
            | pynutil.add_weight(ascii_digit, 0.95)
            | pynutil.add_weight(letters_graph, 1.0)
            | pynutil.add_weight(symbols_graph, 1.1)
        )

        # ==================== ALPHANUMERIC TOKEN (uppercase, no subscript) ====================
        alnum_token = (
            pynutil.add_weight(ascii_two_digit, 0.92)
            | pynutil.add_weight(ascii_digit, 0.95)
            | pynutil.add_weight(upper_letters, 1.0)
        )

        # ==================== CHEMICAL TOKEN (subscript_spoken + uppercase) ====================
        chem_token = (
            pynutil.add_weight(subscript_spoken, 0.9)
            | pynutil.add_weight(upper_letters, 1.0)
            | pynutil.add_weight(ascii_digit, 1.1)
        )

        # ==================== DOMAIN ====================
        domain_segment = word_token + pynini.closure(token_sep + word_token)
        dot_tld        = token_sep + spoken_dot + token_sep + tld
        domain_body    = domain_segment + pynini.closure(dot_tld, 1)
        optional_slash = pynini.closure(token_sep + pynini.cross("फॉरवर्ड स्लैश", "/"), 0, 1)
        domain_full    = domain_body + optional_slash
        domain_tagged  = pynutil.insert("domain: \"") + domain_full + pynutil.insert("\"")

        # ==================== USERNAME ====================
        username_segment = word_token + pynini.closure(token_sep + word_token)
        username_tagged  = pynutil.insert("username: \"") + username_segment + pynutil.insert("\"")

        # ==================== PROTOCOL ====================
        protocol_tagged = pynutil.insert("protocol: \"") + protocols_graph + pynutil.insert("\"")

        # ==================== IP ADDRESS ====================
        ip_octet  = devanagari_d + pynini.closure(token_sep + devanagari_d)
        dot_octet = token_sep + spoken_dot + token_sep + ip_octet
        ip_tagged = pynutil.insert("domain: \"") + ip_octet + pynini.closure(dot_octet, 3, 3) + pynutil.insert("\"")

        # ==================== FILE PATH ====================
        # path uses word_token (lowercase letters) — uppercase handled by alnum_tagged
        path_token = (
            pynutil.add_weight(proper_names_graph | server_name_graph | common_words_graph, 0.8)
            | pynutil.add_weight(ascii_two_digit, 0.92)
            | pynutil.add_weight(ascii_digit, 0.95)
            | pynutil.add_weight(letters_graph, 1.0)
            | pynutil.add_weight(symbols_graph, 1.1)
        )
        path_segment = path_token + pynini.closure(token_sep + path_token)
        path_tagged  = pynutil.insert("path: \"") + path_segment + pynutil.insert("\"")

        # ==================== CHEMICAL FORMULAS ====================
        # Lookup: ग्लूकोज -> C6H12O6 (also covers FeCl₃, H₂SO₄ etc.)
        chemical_lookup_tagged = pynutil.insert("domain: \"") + chemical_lookup + pynutil.insert("\"")

        # Subscript fallback: एच टू एस ओ फ़ोर -> H₂SO₄
        chem_segment         = chem_token + pynini.closure(token_sep + chem_token)
        chemical_sub_tagged  = pynutil.insert("domain: \"") + chem_segment + pynutil.insert("\"")

        # ==================== ALPHANUMERIC CODES ====================
        # TN verbalizer maps: AJNFC3837K -> ए जे एन एफ सी तीन आठ तीन सात के
        # ITN mirrors this: one letter OR one digit per token, space-separated
        # letters_graph gives lowercase; we accept that for codes like ia32
        # For uppercase codes, upper_letters is used
        simple_alnum_token = (
            pynutil.add_weight(ascii_digit, 0.9)
            | pynutil.add_weight(upper_letters, 1.0)
            | pynutil.add_weight(letters_graph, 1.0)
        )
        simple_alnum_segment = simple_alnum_token + pynini.closure(token_sep + simple_alnum_token)
        alnum_tagged = pynutil.insert("domain: \"") + simple_alnum_segment + pynutil.insert("\"")

        # ==================== COMBINED ====================
        email_graph = username_tagged + token_sep + delete_at + token_sep + pynutil.insert(" ") + domain_tagged
        url_graph   = protocol_tagged + token_sep + pynutil.insert(" ") + domain_tagged

        graph = (
            pynutil.add_weight(url_graph,                1.0)
            | pynutil.add_weight(email_graph,            1.01)
            | pynutil.add_weight(ip_tagged,              1.02)
            | pynutil.add_weight(path_tagged,            1.15)
            | pynutil.add_weight(chemical_lookup_tagged, 1.04)
            | pynutil.add_weight(alnum_tagged,           1.05)
            | pynutil.add_weight(chemical_sub_tagged,    1.06)
            | pynutil.add_weight(domain_tagged,          1.2)
        )

        self.graph = graph.optimize()
        self.fst = self.add_tokens(self.graph).optimize()
