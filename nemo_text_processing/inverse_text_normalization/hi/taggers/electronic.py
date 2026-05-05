# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

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
    Finite state transducer for classifying spoken Hindi electronic strings
    into structured token form (ITN direction: spoken → written).

    Fixes applied (v2):
      - digit_graph now produces ASCII digits (0-9), not Devanagari
      - हाइफ़न/हाइफन → - confirmed in symbol_graph (fixes codes like GSAT-18)
      - Bare backslash paths (बैकवर्ड स्लैश ...) now matched
      - www. inside URL paths no longer inserts a space
      - Cased word variants added (Downloads, Desktop, Zoom etc.)

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        # ── Load all 9 inverse lookup tables ────────────────────────────
        latin_graph = pynini.string_file(
            get_abs_path("data/electronic/hindi_to_latin.tsv")
        ).optimize()

        digit_graph = pynini.string_file(
            get_abs_path("data/electronic/hindi_to_digit.tsv")
        ).optimize()

        symbol_graph = pynini.string_file(
            get_abs_path("data/electronic/hindi_symbols_inv.tsv")
        ).optimize()

        domain_tld_graph = pynini.string_file(
            get_abs_path("data/electronic/hindi_domain_inv.tsv")
        ).optimize()

        word_graph = pynini.string_file(
            get_abs_path("data/electronic/hindi_words_inv.tsv")
        ).optimize()

        protocol_graph = pynini.string_file(
            get_abs_path("data/electronic/hindi_protocols_inv.tsv")
        ).optimize()

        subscript_graph = pynini.string_file(
            get_abs_path("data/electronic/hindi_subscript_inv.tsv")
        ).optimize()

        file_ext_graph = pynini.string_file(
            get_abs_path("data/electronic/hindi_file_ext_inv.tsv")
        ).optimize()

        chemical_graph = pynini.string_file(
            get_abs_path("data/electronic/hindi_chemical_inv.tsv")
        ).optimize()

        sp = delete_space

        # ── Atomic token weights ─────────────────────────────────────────
        chem_tok      = pynutil.add_weight(chemical_graph,  0.5)
        word_tok      = pynutil.add_weight(word_graph,      0.9)
        file_ext_tok  = pynutil.add_weight(file_ext_graph,  0.95)
        letter_tok    = pynutil.add_weight(latin_graph,     1.0)
        digit_tok     = pynutil.add_weight(digit_graph,     1.0)
        subscript_tok = pynutil.add_weight(subscript_graph, 1.05)
        symbol_tok    = pynutil.add_weight(symbol_graph,    1.1)

        any_char = (
            chem_tok | word_tok | file_ext_tok
            | letter_tok | digit_tok | subscript_tok | symbol_tok
        )

        # "डॉट <TLD>" → ".<tld>"
        dot_tld = pynini.cross("डॉट", ".") + sp + domain_tld_graph

        # ── FIX: www inside domain path — produce "www." without space ────
        # "डब्ल्यू डब्ल्यू डब्ल्यू डॉट" → "www." as a single domain unit
        www_inline = pynini.cross(
            "डब्ल्यू डब्ल्यू डब्ल्यू डॉट", "www."
        )

        domain_unit = (
            pynutil.add_weight(www_inline,     0.85)  # highest — www. wins
            | pynutil.add_weight(word_tok,       0.9)
            | pynutil.add_weight(file_ext_tok,   0.95)
            | pynutil.add_weight(dot_tld,        0.95)
            | pynutil.add_weight(letter_tok,     1.0)
            | pynutil.add_weight(digit_tok,      1.0)
            | pynutil.add_weight(symbol_tok,     1.1)
        )

        token_seq  = any_char  + pynini.closure(sp + any_char)
        domain_seq = domain_unit + pynini.closure(sp + domain_unit)

        # ── Field tags ───────────────────────────────────────────────────
        ins_username = pynutil.insert("username: \"")
        ins_domain   = pynutil.insert("domain: \"")
        ins_protocol = pynutil.insert("protocol: \"")
        ins_path     = pynutil.insert("path: \"")
        ins_close    = pynutil.insert("\"")

        # ── EMAIL ─────────────────────────────────────────────────────────
        username_seq = ins_username + token_seq + ins_close
        at_boundary  = sp + pynini.cross("एट", "") + sp
        domain_field = ins_domain + domain_seq + ins_close
        email_graph  = username_seq + at_boundary + domain_field

        # ── URL ───────────────────────────────────────────────────────────
        protocol_field = ins_protocol + protocol_graph + ins_close
        url_graph      = protocol_field + sp + domain_field

        # ── FILE PATH (Windows: C: or D: prefix) ─────────────────────────
        path_field      = ins_path + token_seq + ins_close
        windows_trigger = (
            pynini.cross("सी", "C") | pynini.cross("डी", "D")
            | pynini.cross("ई", "E") | pynini.cross("एफ", "F")
        )
        windows_colon = sp + pynini.cross("कोलन", ":") + sp
        windows_path = (
            ins_path
            + windows_trigger
            + windows_colon
            + pynini.cross("बैकवर्ड स्लैश", "\\")
            + pynini.closure(sp + any_char)
            + ins_close
        )

        # ── FIX: Bare backslash path (बैकवर्ड स्लैश ...) ─────────────────
        bare_backslash_path = (
            ins_path
            + pynini.cross("बैकवर्ड स्लैश", "\\")
            + pynini.closure(sp + any_char)
            + ins_close
        )

        # ── Unix path (फॉरवर्ड स्लैश ...) ────────────────────────────────
        unix_path = (
            ins_path
            + pynini.cross("फॉरवर्ड स्लैश", "/")
            + pynini.closure(sp + any_char)
            + ins_close
        )

        # Relative path (word + फॉरवर्ड स्लैश + ...) e.g. backups/temp
        relative_path = (
            ins_path
            + any_char
            + pynini.closure(
                sp + pynini.cross("फॉरवर्ड स्लैश", "/") + sp + any_char
            )
            + ins_close
        )

        path_graph = (
            pynutil.add_weight(windows_path,       1.0)
            | pynutil.add_weight(bare_backslash_path, 1.0)
            | pynutil.add_weight(unix_path,        1.0)
            | pynutil.add_weight(relative_path,    1.1)
            | pynutil.add_weight(path_field,       1.2)
        )

        # ── IP ADDRESS ────────────────────────────────────────────────────
        digit_run = digit_tok + pynini.closure(sp + digit_tok)
        dot_sep   = sp + pynini.cross("डॉट", ".") + sp
        ip_seq    = digit_run + dot_sep + digit_run + dot_sep + digit_run + dot_sep + digit_run
        ip_graph  = ins_domain + ip_seq + ins_close

        # ── CHEMICAL FORMULA ──────────────────────────────────────────────
        chem_seq    = (
            pynutil.add_weight(chemical_graph, 0.5)
            | pynutil.add_weight(token_seq,    1.0)
        )
        chem_tagged = ins_domain + chem_seq + ins_close

        # ── DOMAIN / ALPHANUMERIC CODE (fallback) ─────────────────────────
        domain_only_graph = ins_domain + domain_seq + ins_close

        # ── Combined graph ────────────────────────────────────────────────
        graph = (
            pynutil.add_weight(url_graph,           1.0)
            | pynutil.add_weight(email_graph,       1.0)
            | pynutil.add_weight(ip_graph,          1.0)
            | pynutil.add_weight(path_graph,        1.1)
            | pynutil.add_weight(chem_tagged,       1.15)
            | pynutil.add_weight(domain_only_graph, 1.2)
        )

        self.graph = graph.optimize()
        self.fst   = self.add_tokens(graph).optimize()
