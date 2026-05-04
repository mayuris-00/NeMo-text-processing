# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import GraphFst, delete_space, NEMO_CHAR, convert_space
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


class ElectronicFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        symbols_graph      = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).invert().optimize()
        domain_tld         = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert().optimize()
        server_name_graph  = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert().optimize()
        common_words_graph = pynini.string_file(get_abs_path("data/electronic/common_words.tsv")).invert().optimize()
        file_ext_graph     = pynini.string_file(get_abs_path("data/electronic/file_extensions.tsv")).invert().optimize()
        proper_names_graph = pynini.string_file(get_abs_path("data/electronic/proper_names.tsv")).invert().optimize()
        letters_graph      = pynini.string_file(get_abs_path("data/electronic/letters.tsv")).invert().optimize()
        subscript_spoken   = pynini.string_file(get_abs_path("data/electronic/subscript_spoken.tsv")).invert().optimize()
        chemical_lookup    = pynini.string_file(get_abs_path("data/electronic/chemical_formulas.tsv")).invert().optimize()

        hindi_digit      = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert().optimize()
        hindi_zero       = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert().optimize()
        teens_graph      = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert().optimize()
        devanagari_digit = hindi_digit | hindi_zero

        hi_to_ascii = pynini.string_map([
            ("०","0"),("१","1"),("२","2"),("३","3"),("४","4"),
            ("५","5"),("६","6"),("७","7"),("८","8"),("९","9"),
        ]).optimize()

        ascii_digit     = devanagari_digit @ hi_to_ascii
        ascii_two_digit = (teens_graph @ pynini.cdrewrite(hi_to_ascii, "", "", pynini.closure(NEMO_CHAR))).optimize()

        lower_to_upper = pynini.string_map([
            ("a","A"),("b","B"),("c","C"),("d","D"),("e","E"),("f","F"),
            ("g","G"),("h","H"),("i","I"),("j","J"),("k","K"),("l","L"),
            ("m","M"),("n","N"),("o","O"),("p","P"),("q","Q"),("r","R"),
            ("s","S"),("t","T"),("u","U"),("v","V"),("w","W"),("x","X"),
            ("y","Y"),("z","Z"),
        ]).optimize()
        upper_letters = letters_graph @ lower_to_upper

        sep           = delete_space
        spoken_dot    = pynini.cross("डॉट", ".")
        spoken_slash  = pynini.cross("फॉरवर्ड स्लैश", "/")
        spoken_bslash = pynini.cross("बैकवर्ड स्लैश", "\\")
        spoken_colon  = pynini.cross("कोलन", ":")
        spoken_hyphen = pynini.union(
            pynini.cross("हाइफ़न", "-"),
            pynini.cross("हाइफन", "-"),
        )
        tilde = pynini.cross("~", "~")

        word_token = (
            pynutil.add_weight(proper_names_graph | server_name_graph | common_words_graph, 0.8)
            | pynutil.add_weight(ascii_two_digit, 0.9)
            | pynutil.add_weight(ascii_digit,     0.95)
            | pynutil.add_weight(letters_graph,   1.0)
            | pynutil.add_weight(symbols_graph,   1.1)
        )

        word_seq = word_token + pynini.closure(sep + word_token)

        word_hyph = word_seq + pynini.closure(sep + spoken_hyphen + word_seq)

        tld         = domain_tld | file_ext_graph
        dot_tld     = sep + spoken_dot + sep + tld
        dot_seg     = sep + spoken_dot + sep + word_hyph
        domain_body = word_hyph + pynini.closure(dot_tld, 1) + pynini.closure(dot_seg | dot_tld)

        # ================================================================
        # URL PATH
        # word_hyph handles both plain words AND digit sequences because
        # word_token already contains ascii_digit.
        # e.g. /613/956  -> word_hyph matches 613, then 956 as separate segments
        # e.g. /251x458  -> word_hyph matches 251x458 as mixed word_tokens
        # e.g. /about/   -> word_hyph matches 'about', trailing slash separate
        # e.g. /best-online-courses/ -> word_hyph matches via hyphen connector
        # ================================================================
        path_seg_with_ext = word_hyph + pynini.closure(
            sep + spoken_dot + sep + (file_ext_graph | tld | word_hyph)
        )

        # Mandatory content after slash: /613  /about  /best-online-courses
        url_slash_seg = sep + spoken_slash + sep + path_seg_with_ext

        # Trailing slash only: domain.com/
        url_slash_seg_trailing = sep + spoken_slash

        spoken_hash = pynini.cross("हैशटैग", "#")
        hash_seg    = sep + spoken_hash + pynini.closure(sep + word_hyph, 0, 1)

        url_path = (
            pynini.closure(url_slash_seg)
            + pynini.closure(url_slash_seg_trailing, 0, 1)
            + pynini.closure(hash_seg, 0, 1)
        )

        domain_with_path = domain_body + url_path

        protocol_map = pynini.string_map([
            ("https",    "एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश"),
            ("http",     "एच टी टी पी कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश"),
            ("httpswww", "एच टी टी पी एस कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश डब्ल्यू डब्ल्यू डब्ल्यू डॉट"),
            ("httpwww",  "एच टी टी पी कोलन फॉरवर्ड स्लैश फॉरवर्ड स्लैश डब्ल्यू डब्ल्यू डब्ल्यू डॉट"),
            ("www",      "डब्ल्यू डब्ल्यू डब्ल्यू डॉट"),
        ]).invert().optimize()

        ip_num    = ascii_two_digit | ascii_digit
        ip_octet  = ip_num + pynini.closure(sep + ip_num)
        dot_octet = sep + spoken_dot + sep + ip_octet
        ip_body   = ip_octet + pynini.closure(dot_octet, 3, 3)

        spoken_under_seq = pynini.cross(
            pynini.accep("अंडर") + pynini.accep(" ") + pynini.accep("स्कोर"),
            "_"
        )

        path_conn_hyphen = sep + spoken_hyphen
        path_conn_under  = sep + pynini.cross("अंडर", "") + sep + pynini.cross("स्कोर", "_")

        path_word_hyph = word_seq + pynini.closure(
            (path_conn_hyphen | path_conn_under) + word_seq
        )

        path_seg_with_ext_file = path_word_hyph + pynini.closure(
            sep + spoken_dot + sep + (file_ext_graph | path_word_hyph)
        )

        bslash_seg     = sep + spoken_bslash + pynini.closure(sep + path_seg_with_ext_file, 0, 1)
        drive_letter   = upper_letters
        win_path       = drive_letter + sep + spoken_colon + pynini.closure(bslash_seg, 1)
        fslash_seg     = spoken_slash + pynini.closure(sep + path_word_hyph, 0, 1)
        linux_path     = fslash_seg + pynini.closure(sep + fslash_seg)
        tilde_path     = tilde + sep + fslash_seg + pynini.closure(sep + fslash_seg)
        bare_bslash_path = spoken_bslash + pynini.closure(bslash_seg, 1)
        rel_path = word_hyph + sep + spoken_slash + pynini.closure(
            sep + word_hyph + pynini.closure(
                sep + spoken_slash + pynini.closure(sep + word_hyph, 0, 1)
            )
        )

        file_path_body = (
            pynutil.add_weight(win_path,           1.0)
            | pynutil.add_weight(bare_bslash_path, 1.0)
            | pynutil.add_weight(linux_path,       1.1)
            | pynutil.add_weight(tilde_path,       1.1)
            | pynutil.add_weight(rel_path,         1.2)
        )

        alnum_token = (
            pynutil.add_weight(ascii_digit,   0.85)
            | pynutil.add_weight(upper_letters, 1.0)
            | pynutil.add_weight(letters_graph, 1.0)
        )
        alnum_body = alnum_token + pynini.closure(sep + alnum_token, 1)

        chem_token = (
            pynutil.add_weight(subscript_spoken, 0.9)
            | pynutil.add_weight(upper_letters,  1.0)
            | pynutil.add_weight(ascii_digit,    1.1)
        )
        chem_body = chem_token + pynini.closure(sep + chem_token, 1)

        username_tagged     = pynutil.insert('username: "') + word_seq + pynutil.insert('"')
        domain_tagged       = pynutil.insert('domain: "') + domain_with_path + pynutil.insert('"')
        protocol_tagged     = pynutil.insert('protocol: "') + protocol_map + pynutil.insert('"')
        ip_tagged           = pynutil.insert('domain: "') + ip_body + pynutil.insert('"')
        path_tagged         = pynutil.insert('path: "') + file_path_body + pynutil.insert('"')
        chem_lookup_tagged  = pynutil.insert('domain: "') + chemical_lookup + pynutil.insert('"')
        chem_sub_tagged     = pynutil.insert('domain: "') + chem_body + pynutil.insert('"')
        alnum_tagged        = pynutil.insert('domain: "') + alnum_body + pynutil.insert('"')

        delete_at   = pynutil.delete("एट")
        email_graph = username_tagged + sep + delete_at + sep + pynutil.insert(' ') + domain_tagged
        url_graph   = protocol_tagged + sep + pynutil.insert(' ') + domain_tagged

        graph = (
            pynutil.add_weight(url_graph,           1.0)
            | pynutil.add_weight(email_graph,       1.01)
            | pynutil.add_weight(ip_tagged,         1.02)
            | pynutil.add_weight(path_tagged,       1.03)
            | pynutil.add_weight(chem_lookup_tagged,1.04)
            | pynutil.add_weight(chem_sub_tagged,   1.05)
            | pynutil.add_weight(alnum_tagged,      1.06)
            | pynutil.add_weight(domain_tagged,     1.2)
        )

        self.graph = graph.optimize()
        self.fst = self.add_tokens(self.graph).optimize()